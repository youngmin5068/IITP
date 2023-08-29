import torch
import torchio as tio
from torch.utils.data import random_split
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import os
import nibabel as nib

def data_dicts(image_path,label_path):
    image_subpaths = os.listdir(image_path)
    label_subpaths = os.listdir(label_path)
    image_list = [image_path + "/" + f for f in image_subpaths]
    label_list = [label_path + "/" + f for f in label_subpaths]
    data_dicts = [{'image': img, 'label': lbl} for img, lbl in zip(image_list, label_list)]

    return data_dicts
