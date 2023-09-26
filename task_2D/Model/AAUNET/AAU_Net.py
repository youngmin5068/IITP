import torch.nn as nn
import torch

import sys
sys.path.append("/workspace/IITP/task_2D/train")

class AAU_channel(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(AAU_channel,self).__init__()

        self.conv3x3 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding=1)
        self.bn1 = nn.GroupNorm(out_channel,out_channel)

        self.conv5x5 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=5,padding=2)
        self.bn2 = nn.GroupNorm(out_channel,out_channel)

        self.dil_conv3x3 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,dilation=3,padding=3)
        self.bn3 = nn.GroupNorm(out_channel,out_channel)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=2*out_channel,out_features=out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=out_channel,out_features=out_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):  
        x1 = self.conv3x3(x)
        x1 = self.leaky_relu(self.bn1(x1))
        x2 = self.conv5x5(x)
        x2 = self.leaky_relu(self.bn2(x2))
        x3 = self.dil_conv3x3(x)
        x3 = self.leaky_relu(self.bn3(x3))  

        x4 = self.gap(torch.cat([x2,x3],dim=1))
        x4 = x4.view(x4.size(0), -1)
        x4 = self.fc1(x4)
        x4 = self.relu(x4)

        x4 = self.fc2(x4)
        x4 = self.sigmoid(x4)

        alpha = x4.view(x4.size(0), x4.size(1), 1, 1)
        x2 = x2 * alpha
        x3 = x3 * (1-alpha)

        att = x2 + x3

        return x1, att

class AAU_spatial(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(AAU_spatial,self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel,kernel_size=1)
        self.bn1 = nn.GroupNorm(out_channel,out_channel)

        self.conv2 = nn.Conv2d(in_channel,out_channel,kernel_size=1)
        self.bn2 = nn.GroupNorm(out_channel,out_channel)

        self.conv3 = nn.Conv2d(out_channel,1,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.conv4 = nn.Conv2d(out_channel,out_channel,kernel_size=1)
        self.bn3 = nn.GroupNorm(out_channel,out_channel)
        
        self.resample = nn.Conv2d(1,out_channel,kernel_size=1)

        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self,x,att):
        x = self.conv1(x)
        x = self.leaky_relu(self.bn1(x))

        att = self.conv2(att)
        att = self.leaky_relu(self.bn2(att))

        spatial = x+att
        spatial = self.relu(spatial)
        spatial = self.conv3(spatial)
        
        beta = self.sigmoid(spatial)
        beta_1 = 1-beta

        x = x * self.resample(beta_1)
        att = att * self.resample(beta)

        result = x+att

        result = self.conv4(result)
        # result = self.leaky_relu(self.bn3(result))

        return result

class HAAM(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(HAAM,self).__init__()

        self.channel_attention = AAU_channel(in_channel=in_channel,out_channel=out_channel)
        self.spatial_attention = AAU_spatial(in_channel=out_channel,out_channel=out_channel)
    
    def forward(self,x):
        x,att = self.channel_attention(x)
        result = self.spatial_attention(x,att)

        return result
    


class Double_HAAM(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Double_HAAM,self).__init__()
        self.haam = HAAM(in_channel,out_channel)
        self.haam2 = HAAM(out_channel,out_channel)
        self.bn = nn.GroupNorm(out_channel,out_channel)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self,x):
        x = self.haam(x)
        x = self.haam2(x)
        x = self.bn(x)
        x = self.leaky_relu(x)

        return x
    
class Up(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Up,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.double_haam = Double_HAAM(in_channel,out_channel)
    def forward(self,enc,x):
        x = self.upsample(x)
        x = self.double_haam(torch.cat([enc,x],dim=1))

        return x

class Outconv(nn.Module):
    def __init__(self,in_channel,out_channel,num_classes):
        super(Outconv,self).__init__()

        self.haam = HAAM(in_channel,out_channel)
        self.haam2 = HAAM(out_channel,num_classes)

    def forward(self,x):
        x = self.haam(x)
        x = self.haam2(x)

        return x

class AAU_Net(nn.Module):  
    def __init__(self,input_channels,num_classes):
        super(AAU_Net,self).__init__()

        self.maxpool = nn.MaxPool2d(2)

        self.enc1 = Double_HAAM(input_channels,32)
        self.enc2 = Double_HAAM(32,64)
        self.enc3 = Double_HAAM(64,128)
        self.enc4 = Double_HAAM(128,256)

        self.bottom = Double_HAAM(256,256)

        self.dec1 = Up(512,128)
        self.dec2 = Up(256,64)
        self.dec3 = Up(128,32)
        self.dec4 = Up(64,32)
        self.result = Outconv(32,32,num_classes=num_classes)

    def forward(self,x):
        x1 = self.enc1(x) # 64x512x512
        x2 = self.enc2(self.maxpool(x1)) # 128x256x256
        x3 = self.enc3(self.maxpool(x2)) # 256x128x128
        x4 = self.enc4(self.maxpool(x3)) # 512x64x64
        x5 = self.bottom(self.maxpool(x4)) # 512 x 32 x 32

        x6 = self.dec1(x4,x5)
        x7 = self.dec2(x3,x6)
        x8 = self.dec3(x2,x7)
        x9 = self.dec4(x1,x8)
        result = self.result(x9)

        return result

    

if __name__ == '__main__':
    # Sample input image with shape (1, 1, 512, 512)
    x = torch.rand(1, 1, 512, 512)


    # Creating an instance of the AAU_channel class
    net = AAU_Net(1,1)
    
    result = net(x)
    # Output shape
    print(result.shape)  # Exp
