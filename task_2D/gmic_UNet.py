import torch
import torch.nn as nn
import numpy as np
import tools 
import modules as m
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


class GMIC_UNet(nn.Module):
    def __init__(self, parameters):
        super(GMIC_UNet, self).__init__()

        self.swin_unetr = SwinUNETR(img_size=(512,512),spatial_dims=2,in_channels=1,out_channels=1,depths=(2,2,2,2))

        # save parameters
        self.experiment_parameters = parameters

        self.global_network = m.GlobalNetwork(self.experiment_parameters)
        self.aggregation_function = m.TopTPercentAggregationFunction(self.experiment_parameters)

    def forward(self,x_original):

        swin_output = self.swin_unetr(x_original)
        h_g, self.saliency_map = self.global_network(x_original)
        saliency_resized = F.interpolate(self.saliency_map, size=(512,512),mode="bilinear",align_corners=True)
        
        self.y_global = self.aggregation_function(self.saliency_map)
        
        att_res_x = swin_output * saliency_resized
    
        return att_res_x, self.y_global
    
# sample = torch.randn((1,1,512,512))

# parameters = {
#         "percent_t":0.1,
#         # "device_type":"gpu",
#         # "gpu_number":0,
#         "post_processing_dim": 256,
#         "num_classes": 1
#     }

# model = GMIC_UNet(parameters)

# print(torch.max(model(sample)[2]))