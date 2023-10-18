""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubPixel_DoubleConv(nn.Module):
    def __init__(self,in_channels=1,out_channels=8,mid_channels=None):
        super(SubPixel_DoubleConv,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=3,padding=1,bias=True),
            nn.GroupNorm(mid_channels, mid_channels),
            nn.PReLU(),
            nn.Conv2d(mid_channels, out_channels,kernel_size=3,padding=1,bias=True),
            nn.GroupNorm(out_channels, out_channels),
            nn.PReLU()
        )
        self.double_conv5_5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=5,padding=2,bias=True),
            nn.GroupNorm(mid_channels, mid_channels),
            nn.PReLU(),
            nn.Conv2d(mid_channels, out_channels,kernel_size=5,padding=2,bias=True),
            nn.GroupNorm(out_channels, out_channels),
            nn.PReLU()
        )
    def forward(self,x):
        x1 = self.double_conv3_3(x)
        x2 = self.double_conv5_5(x)
        result = torch.cat([x1,x2],dim=1)

        return result

class Depth2Space(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super(Depth2Space,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1)

    def forward(self,x):

        upscale_factor = 2
        x = F.pixel_shuffle(x, upscale_factor)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Space2Depth(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super(Space2Depth,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1)
    def forward(self,x):
        b, c, h, w = x.size()
        block_size=2
        unfolded_x = torch.nn.functional.unfold(x, kernel_size=block_size, stride=block_size)
        unfolded_x = unfolded_x.view(b, c * block_size * block_size, h // block_size, w // block_size)

        x = self.conv1(unfolded_x)
        x = self.conv2(x)

        return x

class SubPixelNetwork(nn.Module):
    def __init__(self):
        super(SubPixelNetwork,self).__init__()
        self.double_conv = SubPixel_DoubleConv(1,8) # output shape: 16x512x512
        self.d2s = Depth2Space(4,8,4) # output shape: 4x1024x1024 -> 4x1024x1024
        self.s2d = Space2Depth(16,32,32) # output shape: 32x512x512 -> 32x512x512

    def forward(self,x):
        x = self.double_conv(x)
        x1 = self.d2s(x)
        x2 = self.s2d(x1)

        return x1,x2



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
    

class SubPixelUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SubPixelUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.subpixelModule = SubPixelNetwork()

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 32, bilinear))
        self.double_conv = DoubleConv(64, 32)
        self.up5 = Depth2Space(8, 4, 4)

        self.d2s = Space2Depth(32, n_classes, n_classes)
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        sub_x1,sub_x2 = self.subpixelModule(x) #4x1024x1024 , 32x512x512
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.cat([x,sub_x2],dim=1) # 128x512x512
        x = self.double_conv(x) # 32x512x512
        x = self.up5(x) #4x1024x1024
        x = torch.cat([x,sub_x1],dim=1) #8x1024x1024

        logits = self.d2s(x)
        return logits



if __name__ == "__main__":

    sample = torch.randn(1,1,512,512)
    model = SubPixelUNet(n_channels=1,n_classes=1)
    output = model(sample)
    print(output.shape)