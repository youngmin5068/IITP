import numpy as np
from PIL import Image
import torch as torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import cv2
from einops import rearrange
from custom_transforms import *
import pydicom as dcm
from pydicom.pixel_data_handlers.util import apply_voi_lut

class lung_Dataset(Dataset):
    def __init__(self,path):
        self.path = path

        self.train_path_list = []
        self.train_list = []

        self.label_path_list = []
        self.label_list = []

        self.train_path = path + "/input_dcm"
        self.label_path = path + "/target"

        
        for file in os.listdir(self.train_path):
            self.train_path_list.append(os.path.join(self.train_path,file))
        self.train_path_list.sort()
                
        for file in os.listdir(self.label_path):
            self.label_path_list.append(os.path.join(self.label_path,file))           
        self.label_path_list.sort()


    def __len__(self):
        return len(self.label_path_list)
        
    def __getitem__(self,idx):


        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((512,512)),
                                            #transforms.CenterCrop(size=384),
                                            customRandomHorizontalFlip(SEED=idx,p=0.5), 
                                            customRandomRotate(degrees=180,SEED=idx),
                                            #customRandomResizedCrop(SEED=idx,size=(384,384)),
                                             ])
        
        image_path = self.train_path_list[idx]

        slice = dcm.read_file(image_path)
        image = slice.pixel_array.astype(np.float32)
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)
        image = apply_voi_lut(image, slice)
        image = Image.fromarray(image)

        label_path = self.label_path_list[idx]
        label = np.array(Image.open(label_path).convert("L"))
        label = Image.fromarray(label)


        input_image = self.transform(image)
        target_image = self.transform(label)


        thresh = np.zeros_like(target_image)
        thresh[target_image > 0.5] = 1

        return input_image, thresh



class tumor_Dataset(Dataset):
    def __init__(self,path, train=True):
        self.path = path
        self.train = train
        self.train_path_list = []
        self.train_list = []

        self.label_path_list = []
        self.label_list = []

        self.train_path = path + "/input"
        self.label_path = path + "/target"

        
        for file in os.listdir(self.train_path):
            self.train_path_list.append(os.path.join(self.train_path,file))
        self.train_path_list.sort()
                
        for file in os.listdir(self.label_path):
            self.label_path_list.append(os.path.join(self.label_path,file))           
        self.label_path_list.sort()


    def __len__(self):
        return len(self.label_path_list)
    
    def get_label(self,image):
     if np.any(image == 1):
        return 1
     else:
        return 0

    def __getitem__(self,idx):
        if self.train:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((512,512)),
                                                transforms.CenterCrop((250, 512)),
                                                transforms.Resize((250, 512))
                                                #customRandomRotate(degrees=180,SEED=idx),
                                                #customRandomResizedCrop(SEED=idx,size=(512,512))
                                                ])
            
            
        image_path = self.train_path_list[idx]

        slice = dcm.read_file(image_path)
        image = slice.pixel_array
        image = apply_voi_lut(image, slice)
        epsilon = 1e-10
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val+epsilon)
        
        image = Image.fromarray(image)

        label_path = self.label_path_list[idx]
        label = np.array(Image.open(label_path).convert("L"))
        label = Image.fromarray(label)


        input_image = self.transform(image)
        target_image = self.transform(label)


        thresh = np.zeros_like(target_image)
        thresh[target_image > 0.5] = 1

        # class_label = self.get_label(thresh)

        return input_image, thresh

# class tumor_Dataset(Dataset):
#     def __init__(self,root,input_path,target_path):
#         self.path = root

#         self.train_path_list = []
#         self.train_list = []

#         self.target_path_list = []
#         self.target_list = []

#         self.train_path = root + input_path
#         self.target_path = root + target_path
        
#         for file in os.listdir(self.train_path):
#             self.train_path_list.append(os.path.join(self.train_path,file))
#         self.train_path_list.sort()

#         for file in os.listdir(self.target_path):
#             self.target_path_list.append(os.path.join(self.target_path,file))           
#         self.target_path_list.sort()

#     def __len__(self):
#         return len(self.train_path_list)
        
#     def get_label(self,image):
#         if np.any(image == 1.0):
#             return 1
#         else:
#             return 0
        
#     def __getitem__(self,idx):
#         self.transform = transforms.Compose([transforms.ToTensor(),
#                                             transforms.Resize((512,512)),
#                                             customRandomRotate(degrees=180,SEED=idx),
#                                             #customRandomResizedCrop(SEED=idx,size=(256,256))
#                                              ])
#         image_path = self.train_path_list[idx]
#         # dicom 파일일때만
#         # slice = dcm.read_file(image_path)
#         # image = slice.pixel_array
#         # image = apply_voi_lut(image, slice)
#         image = np.array(Image.open(image_path))
#         epsilon = 1e-10
#         min_val = np.min(image)
#         max_val = np.max(image)
#         image = (image - min_val) / (max_val - min_val+epsilon)
#         image = Image.fromarray(image)

#         target_path = self.target_path_list[idx]
#         target = np.array(Image.open(target_path).convert("L"))

#         target = Image.fromarray(target)

#         input_image = self.transform(image)
#         target_image = self.transform(target)


#         target_thresh = np.zeros_like(target_image)
#         target_thresh[target_image > 0.5] = 1.0

#         class_label = float(self.get_label(target_thresh))
        
#         return input_image, target_thresh, class_label
    
if __name__ == "__main__":
    dataset_path = "/mount_folder"
    dataset = tumor_Dataset(dataset_path,"/roi_input","/target")
    print(len(dataset))
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

    sample = next(iter(dataloader))

    print((torch.min(sample[0])))


    