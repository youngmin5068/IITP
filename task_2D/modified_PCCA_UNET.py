import torch
import torch.nn as nn
import torch.nn.functional as F

class PCA(nn.Module):
    def __init__(self, input_channels):
        super(PCA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 서로 다른 커널 크기를 가진 1D 컨볼루션 레이어들
        self.conv1 = nn.Conv1d(input_channels, input_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_channels, input_channels, kernel_size=5, padding=2)

    def forward(self, x):
        u = x.clone()
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)

        avg_pool = avg_pool.squeeze(-1)  # 2D 형태로 변환
        max_pool = max_pool.squeeze(-1) 

        avg_out1 = self.conv1(avg_pool)
        avg_out3 = self.conv3(avg_pool)
        avg_out5 = self.conv5(avg_pool)
        outs_avg = avg_out1 + avg_out3 + avg_out5

        max_out1 = self.conv1(max_pool)
        max_out3 = self.conv3(max_pool)
        max_out5 = self.conv5(max_pool)
        outs_max = max_out1 + max_out3 + max_out5

        out = outs_avg + outs_max

        return  u * out.unsqueeze(-1)
    

class CA(nn.Module):
    def __init__(self,dim,ratio=1):
        super(CA, self).__init__()
        self.avgpool_x = nn.AdaptiveAvgPool2d((1, None))  # X 축에 대한 pooling
        self.avgpool_y = nn.AdaptiveAvgPool2d((None, 1))  # Y 축에 대한 pooling

        self.conv = nn.Conv2d(dim,dim//ratio,1)
        self.bn = nn.BatchNorm2d(dim//ratio)
        self.siLU = nn.SiLU(inplace=True)

        self.conv_1 = nn.Conv2d(dim//ratio,dim,1)
        self.conv_2 = nn.Conv2d(dim//ratio,dim,1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        u = x.clone()
        x_avg = self.avgpool_x(x).permute(0,1,3,2)  # shape: (batch_size, 32, 1, 256) -> (batch_size, 32, 256, 1)
        y_avg = self.avgpool_y(x)  # shape: (batch_size, 32, 256, 1)
        # 두 결과를 높이 방향으로 연결
        combined = torch.cat([x_avg, y_avg], dim=2) # shape: (batch_size, 32, 512, 1)

        combined = self.siLU(self.bn(self.conv(combined)))
        
        split_1, split_2 = torch.split(combined, split_size_or_sections=combined.shape[2]//2, dim=2) 
        split_1 = self.sigmoid(self.conv_1(split_1.permute(0,1,3,2))) # batch, 32, 256, 1 -> batch, 32 , 1, 256
        split_2 = self.sigmoid(self.conv_2(split_2)) # batch, 32, 256, 1

        result = u * split_1 * split_2
        return result
    
class PCCA(nn.Module):
    def __init__(self,dim):
        super(PCCA, self).__init__()
        self.PCA = PCA(dim)
        self.CA = CA(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        u = x.clone()
        pca = self.PCA(x)
        ca = self.CA(x)

        return self.sigmoid(pca+ca) * u
    
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
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

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class pcca_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(pcca_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.pcca0 = PCCA(64)

        self.down1 = (Down(64, 128))
        self.pcca1 = PCCA(128)

        self.down2 = (Down(128, 256))
        self.pcca2 = PCCA(256)

        self.down3 = (Down(256, 512))
        self.pcca3 = PCCA(512)

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
        x = self.up1(x5, self.pcca3(x4))
        x = self.up2(x, self.pcca2(x3))
        x = self.up3(x, self.pcca1(x2))
        x = self.up4(x, self.pcca0(x1))
        logits = self.outc(x)
        return logits
