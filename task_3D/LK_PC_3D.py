import torch
import torch.nn as nn
import torch.nn.functional as F

class LKA3D(nn.Module):
    def __init__(self, in_channels, out_channels, out=False):
        super().__init__()

        if out:
            self.seq = nn.Sequential(
                                nn.Conv3d(in_channels, in_channels, 3, padding=1),
                                nn.GroupNorm(4, in_channels),
                                nn.PReLU(),
                                nn.Conv3d(in_channels, in_channels, 5, padding=2),
                                nn.GroupNorm(4, in_channels),
                                nn.PReLU(),
                                nn.Conv3d(in_channels, out_channels, 7, stride=1, padding=3, dilation=1), # Adjusted padding and dilation for 3D
            )
        else:
            self.seq = nn.Sequential(
                                 nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                 nn.GroupNorm(4, out_channels),
                                 nn.PReLU(),
                                 nn.Conv3d(out_channels, out_channels, 5, padding=2),
                                 nn.GroupNorm(4, out_channels),
                                 nn.PReLU(),
                                 nn.Conv3d(out_channels, out_channels, 7, stride=1, padding=3, dilation=1), # Adjusted padding and dilation for 3D
                                 nn.GroupNorm(4, out_channels),
                                 nn.PReLU()
            )

    def forward(self, x):
        x = self.seq(x)
        return x


class PCA3D(nn.Module):
    def __init__(self, input_channels):
        super(PCA3D, self).__init__()
        self.avg_pool3d  = nn.AdaptiveAvgPool3d((1,1,1))
        self.max_pool3d = nn.AdaptiveMaxPool3d((1,1,1))
        # Define 3D convolution layers with different kernel sizes
        self.conv1 = nn.Conv1d(input_channels, input_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_channels, input_channels, kernel_size=5, padding=2)

    def forward(self, x):
        u = x.clone()
        avg_pool = self.avg_pool3d(x).squeeze(-1).squeeze(-1)
        max_pool = self.max_pool3d(x).squeeze(-1).squeeze(-1)

        avg_out1 = self.conv1(avg_pool)
        avg_out3 = self.conv3(avg_pool)
        avg_out5 = self.conv5(avg_pool)
        outs_avg = avg_out1 + avg_out3 + avg_out5

        max_out1 = self.conv1(max_pool)
        max_out3 = self.conv3(max_pool)
        max_out5 = self.conv5(max_pool)
        outs_max = max_out1 + max_out3 + max_out5

        out = outs_avg + outs_max

        return u * out.unsqueeze(-1).unsqueeze(-1)


class CA3D(nn.Module):
    def __init__(self, dim, ratio=1):
        super(CA3D, self).__init__()

        self.avgpool_x = nn.AdaptiveAvgPool3d((1, None, None))  # X-axis pooling
        self.avgpool_y = nn.AdaptiveAvgPool3d((None, 1, None))  # Y-axis pooling
        self.avgpool_z = nn.AdaptiveAvgPool3d((None, None, 1))  # Z-axis pooling

        self.conv = nn.Conv3d(dim, dim // ratio, 1)
        self.bn = nn.BatchNorm3d(dim // ratio)
        self.siLU = nn.SiLU(inplace=True)

        self.conv_1 = nn.Conv3d(dim // ratio, dim, 1)
        self.conv_2 = nn.Conv3d(dim // ratio, dim, 1)
        self.conv_3 = nn.Conv3d(dim // ratio, dim, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        u = x.clone()
        x_avg = self.avgpool_x(x).permute(0,1,3,2,4)
        y_avg = self.avgpool_y(x)
        z_avg = self.avgpool_z(x).permute(0,1,2,4,3)


        combined = torch.cat([x_avg, y_avg, z_avg], dim=2)

        combined = self.siLU(self.bn(self.conv(combined)))

        split_1, split_2, split_3 = torch.split(combined, split_size_or_sections=combined.shape[2] // 3, dim=2)

        split_1 = self.sigmoid(self.conv_1(split_1).permute(0,1,3,2,4))
        split_2 = self.sigmoid(self.conv_2(split_2))
        split_3 = self.sigmoid(self.conv_3(split_3).permute(0,1,2,4,3))

        result = u * split_1 * split_2 * split_3
        return result

class PCCA3D(nn.Module):
    def __init__(self, dim):
        super(PCCA3D, self).__init__()
        self.PCA = PCA3D(dim)
        self.CA = CA3D(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        pca = self.PCA(x)
        ca = self.CA(x)
        return self.sigmoid(pca + ca) * u

class LKA_enc3D(nn.Module):
    def __init__(self, in_channel, out_channel, first=False, out=False):
        super(LKA_enc3D, self).__init__()

        if out:
            self.lka_block = LKA3D(in_channel, out_channel, out)
        else:
            self.lka_block = LKA3D(in_channel, out_channel)
            
        self.pool3d = nn.MaxPool3d(2)
        self.first = first

    def forward(self, x):
        if not self.first:
            x = self.pool3d(x)
        x = self.lka_block(x)
        return x
    
class Upsample3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample3D, self).__init__()

        self.deconv = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=2, stride=2)
        self.lka = LKA_enc3D(in_channel, out_channel, first=True)
        
    def forward(self, x, skip_connection):
        x = self.deconv(x)
        x = torch.cat([skip_connection, x], dim=1)
        x = self.lka(x)
        return x

class Out3D(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(Out3D, self).__init__()
        self.num_classes = num_classes

        self.deconv = nn.ConvTranspose3d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.lka = LKA_enc3D(in_channel, self.num_classes, first=True, out=True)

    def forward(self, x, skip_connection):
        x = self.deconv(x)
        x = torch.cat([skip_connection, x], dim=1)
        x = self.lka(x)
        return x

class LK_PC_UNet3D(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(LK_PC_UNet3D, self).__init__()
        features = [32, 64, 128, 256, 512]
        self.in_channel = in_channel
        self.num_classes = num_classes

        self.enc1 = LKA_enc3D(self.in_channel, features[0], first=True)
        self.pcca1 = PCCA3D(features[0])
        self.enc2 = LKA_enc3D(features[0], features[1])
        self.pcca2 = PCCA3D(features[1])
        self.enc3 = LKA_enc3D(features[1], features[2])
        self.pcca3 = PCCA3D(features[2])
        self.enc4 = LKA_enc3D(features[2], features[3])
        self.pcca4 = PCCA3D(features[3])
        self.enc5 = LKA_enc3D(features[3], features[4])

        self.up1 = Upsample3D(features[4], features[3])
        self.up2 = Upsample3D(features[3], features[2])
        self.up3 = Upsample3D(features[2], features[1])

        self.out = Out3D(features[1], self.num_classes)

    def forward(self, x):
        x1 = self.enc1(x)

        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        # x5 = self.enc5(x4)

        # x6 = self.up1(x5, self.pcca4(x4))
        x7 = self.up2(x4, self.pcca3(x3))
        x8 = self.up3(x7, self.pcca2(x2))
        out = self.out(x8, self.pcca1(x1))
        return out
    

    
sample = torch.randn((1,1,64,64,64))

model = LK_PC_UNet3D(1,1)

print(model(sample).shape)