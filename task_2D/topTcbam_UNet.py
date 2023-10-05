import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class TopTPercentChannelGate(nn.Module):
    def __init__(self, gate_channels, percent_t, reduction_ratio=16, pool_types=['avg', 'max']):
        super(TopTPercentChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.percent_t = percent_t
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        
    def forward(self, x):
        b, c, _, _ = x.size()
        x_flatten = x.view(b, c, -1)
        top_t = int(round(x_flatten.size(2) * self.percent_t))
        
        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                selected_values, _ = x_flatten.topk(top_t, dim=2)
                pool = selected_values.mean(dim=2, keepdim=True)
            elif pool_type == 'max':
                selected_values, _ = x_flatten.topk(top_t, dim=2)
                pool = selected_values.max(dim=2, keepdim=True)[0]
            else:
                raise ValueError("Invalid pool_type, choose between 'avg' and 'max'")
                
            channel_att_raw = self.mlp(pool)
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
                
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale
    

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


class Topt_CBAM(nn.Module):
    def __init__(self, gate_channels, percent_t, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Topt_CBAM, self).__init__()
        self.percent_t = percent_t
        self.reduction_ratio = reduction_ratio
        self.ChannelGate = TopTPercentChannelGate(gate_channels, percent_t=self.percent_t,reduction_ratio=self.reduction_ratio, pool_types=['avg', 'max'])
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class DoubleConv(nn.Module):
    """(convolution => [BN] => PReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.PReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class top_t_cbam_UNet(nn.Module):
    def __init__(self, n_channels, n_classes,percent_t, bilinear=False):
        super(top_t_cbam_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.percent_t = percent_t

        self.inc = (DoubleConv(n_channels, 64))
        self.tCbam0 = Topt_CBAM(64,percent_t=self.percent_t)

        self.down1 = (Down(64, 128))
        self.tCbam1 = Topt_CBAM(128,percent_t=self.percent_t)

        self.down2 = (Down(128, 256))
        self.tCbam2 = Topt_CBAM(256,percent_t=self.percent_t)

        self.down3 = (Down(256, 512))
        self.tCbam3 = Topt_CBAM(512,percent_t=self.percent_t)

        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, self.tCbam3(x4))
        x = self.up2(x, self.tCbam2(x3))
        x = self.up3(x, self.tCbam1(x2))
        x = self.up4(x, self.tCbam0(x1))
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    sample = torch.randn((4,1,512,512))
    model = top_t_cbam_UNet(n_channels=1,n_classes=1)

    print(model(sample).shape)

