import torch
import torch.nn as nn
import torch.nn.functional as F


class LKA(nn.Module):
    def __init__(self, in_channels,out_channels,out=False):
        super().__init__()

        if out:
            self.seq = nn.Sequential(
                                nn.Conv2d(in_channels,in_channels,3,padding=1),
                                nn.BatchNorm2d(in_channels),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 5, padding=2),
                                nn.BatchNorm2d(in_channels),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, out_channels, 7, stride=1, padding=9, dilation=3),
            )
        else:
            self.seq = nn.Sequential(
                                 nn.Conv2d(in_channels,out_channels,3,padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.PReLU(),
                                 nn.Conv2d(out_channels, out_channels, 5, padding=2),
                                 nn.BatchNorm2d(out_channels),
                                 nn.PReLU(),
                                 nn.Conv2d(out_channels, out_channels, 7, stride=1, padding=9, dilation=3),
                                 nn.BatchNorm2d(out_channels),
                                 nn.PReLU()
            )

    def forward(self, x):
   
        x = self.seq(x)
        return x


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


class LKA_enc(nn.Module):
    def __init__(self,in_channel,out_channel,first=False,out=False):
        super(LKA_enc,self).__init__()

        if out:
            self.lka_block = LKA(in_channel,out_channel,out)
        else:
            self.lka_block = LKA(in_channel, out_channel)
        self.pool2d = nn.MaxPool2d(2)
        self.first = first

    def forward(self,x):
        if not self.first:
            x = self.pool2d(x)
        x = self.lka_block(x)

        return x


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        
        # 전치 합성곱 (Transposed Convolution)
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.lka = LKA_enc(in_channel,out_channel,first=True)
        
        # 활성화 함수
        self.prelu = nn.PReLU()

    def forward(self, x, skip_connection):

        
        # 전치 합성곱을 사용하여 업샘플링
        x = self.deconv(x)
        x = torch.cat([skip_connection, x], dim=1)
        x = self.lka(x)
    
        return x
    

class Out(nn.Module):
    def __init__(self, in_channel, num_clases):
        super(Out, self).__init__()
        self.num_classes = num_clases

        self.deconv = nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=2, stride=2)
        self.lka = LKA_enc(in_channel,self.num_classes,first=True,out=True)

    def forward(self, x, skip_connection):

        x = self.deconv(x)
        x = torch.cat([skip_connection, x], dim=1)
        x = self.lka(x)
    
        return x


class LK_PC_UNet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super(LK_PC_UNet, self).__init__()
        features = [32, 64, 128, 256, 512]
        self.in_channel = in_channel
        self.num_classes = num_classes

        self.enc1 = LKA_enc(self.in_channel,features[0],first=True) # 32x512x512
        self.pcca1 = PCCA(features[0])
        self.enc2 = LKA_enc(features[0],features[1]) # 64x256x256
        self.pcca2 = PCCA(features[1])
        self.enc3 = LKA_enc(features[1],features[2]) # 128x128x128
        self.pcca3 = PCCA(features[2])
        self.enc4 = LKA_enc(features[2],features[3]) # 256x64x64
        self.pcca4 = PCCA(features[3])
        self.enc5 = LKA_enc(features[3],features[4]) # 512x32x32

        self.up1 = Upsample(features[4],features[3])
        self.up2 = Upsample(features[3],features[2])
        self.up3 = Upsample(features[2],features[1])

        self.out = Out(features[1],self.num_classes)

    def forward(self,x):
        
        x1 = self.enc1(x) 
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        

        x6 = self.up1(x5,self.pcca4(x4))
        x7 = self.up2(x6,self.pcca3(x3))
        x8 = self.up3(x7,self.pcca2(x2))

        out = self.out(x8,self.pcca1(x1))

        return out


sample = torch.randn((4,1,512,512))

model = LK_PC_UNet(1,1)

print(model(sample).shape)