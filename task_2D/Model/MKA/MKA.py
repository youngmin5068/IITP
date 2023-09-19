import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import math

import torch
import math
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
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

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
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
    

####################################################################
    
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
 

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, dim,ratio=1):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim//ratio, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(dim//ratio)
        self.proj_2 = nn.Conv2d(dim//ratio, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKA_Block(nn.Module):
    def __init__(self, in_channel,out_channel, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.PReLU(),
            nn.Conv2d(out_channel,out_channel,3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.PReLU()               
            )

        self.attn = Attention(out_channel)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(out_channel)
        mlp_hidden_dim = int(out_channel * mlp_ratio)
        self.mlp = Mlp(in_features=out_channel, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((out_channel)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((out_channel)), requires_grad=True)



    def forward(self, x):
        x = self.double_conv(x)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class AAU_channel(nn.Module):
    def __init__(self,dim):
        super(AAU_channel,self).__init__()

        self.conv3x3 = nn.Conv2d(dim,dim,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(dim)

        self.conv5x5 = nn.Conv2d(dim,dim,kernel_size=5,padding=2)
        self.bn2 = nn.BatchNorm2d(dim)

        self.dil_conv3x3 = nn.Conv2d(dim,dim,kernel_size=3,dilation=3,padding=3)
        self.bn3 = nn.BatchNorm2d(dim)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=2*dim,out_features=dim)

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=dim,out_features=dim)
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
    def __init__(self,dim,ratio=16):
        super(AAU_spatial,self).__init__()

        self.conv1 = nn.Conv2d(dim, dim//ratio,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(dim//ratio)

        self.conv2 = nn.Conv2d(dim,dim//ratio,kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dim//ratio)

        self.conv3 = nn.Conv2d(dim//ratio,1,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        
        self.resample = nn.Conv2d(1,dim//ratio,kernel_size=1)
        self.conv4 = nn.Conv2d(dim//ratio,dim,kernel_size=1)
        
        self.bn3 = nn.BatchNorm2d(dim)
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
    def __init__(self,dim):
        super(HAAM,self).__init__()

        self.channel_attention = AAU_channel(dim)
        self.spatial_attention = AAU_spatial(dim)
    
    def forward(self,x):
        x,att = self.channel_attention(x)
        result = self.spatial_attention(x,att)

        return result
    
class PCA(nn.Module):
    def __init__(self, input_channels):
        super(PCA, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        # 서로 다른 커널 크기를 가진 1D 컨볼루션 레이어들
        self.conv1d_1 = nn.Conv1d(input_channels, out_channels=input_channels, kernel_size=1)
        self.conv1d_3 = nn.Conv1d(input_channels, out_channels=input_channels, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(input_channels, out_channels=input_channels, kernel_size=5, padding=2)
    
    def forward(self, x):
        u = x.clone()
        x = self.GAP(x)
        x = x.squeeze(-1)  # 2D 형태로 변환

        out1 = self.conv1d_1(x)
        out3 = self.conv1d_3(x)
        out5 = self.conv1d_5(x)

        outs = out1 + out3 + out5
        result = u * outs.unsqueeze(-1)

        return result
    

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
        x_avg = self.avgpool_x(x)  # shape: (batch_size, 32, 1, 256)
        y_avg = self.avgpool_y(x).permute(0,1,3,2)  # shape: (batch_size, 32, 1, 256)
        # 두 결과를 높이 방향으로 연결
        combined = torch.cat([x_avg, y_avg], dim=3).permute(0,1,3,2)  # shape: (batch_size, 32, 512, 1)

        combined = self.siLU(self.bn(self.conv(combined)))
        
        split_1, split_2 = torch.split(combined, split_size_or_sections=combined.shape[2]//2, dim=2) 
        split_1 = self.sigmoid(self.conv_1(split_1)) # batch, 32, 256, 1
        split_2 = self.sigmoid(self.conv_2(split_2.permute(0,1,3,2))) # batch, 32, 1, 256

        result = u * split_1 * split_2
        return result
    
class PCCA(nn.Module):
    def __init__(self,dim):
        super(PCCA, self).__init__()
        self.PCA = PCA(dim)
        self.CA = CA(dim)

    def forward(self,x):
        pca = self.PCA(x)
        ca = self.CA(x)

        return pca+ca
    

class MKA(nn.Module):
    def __init__(self,dim):
        super(MKA, self).__init__()
        
        self.cbam = CBAM(dim)
        self.pcca = PCCA(dim)
        self.lka = LKA(dim)
        self.haam = HAAM(dim)

        self.conv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        u = x.clone()
        cbam_x = self.conv(self.cbam(x))
        pcca_x = self.conv(self.pcca(x))
        lka_x = self.conv(self.lka(x))
        haam_x = self.conv(self.haam(x))

        sum_x = cbam_x+pcca_x+lka_x+haam_x

        #concat_x = torch.cat([cbam_x, pcca_x, lka_x,haam_x],dim=1)

        output = self.sigmoid(self.bn(sum_x))

        return output * u
    

sample = torch.randn(1,1,256,256)

model = LKA_Block(1,32)

print(model(sample).shape)