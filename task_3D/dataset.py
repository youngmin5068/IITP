import torch
import torchio as tio
from torch.utils.data import random_split
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import os
import nibabel as nib

class BreastDataset(Dataset):
    def __init__(self, image_paths, breast_paths,transform=None):
        self.image_paths = image_paths
        self.breast_paths = breast_paths
        self.transform = transform

        self.image_list = []
        self.breast_list = []

        for file in os.listdir(self.image_paths):
            self.image_list.append(os.path.join(self.image_paths, file))
        self.image_list.sort()

        for file in os.listdir(self.breast_paths):
            self.breast_list.append(os.path.join(self.breast_paths,file))
        self.breast_list.sort()


        
    def __len__(self):
        assert len(self.image_list) == len(self.breast_list)

        return len(self.image_list)

    def __getitem__(self, idx):

        image_path = self.image_list[idx]
        breast_path = self.breast_list[idx]

        nifti_img = nib.load(image_path)
        nifti_breast = nib.load(breast_path)

        if self.transform:
            nifti_img = self.transform(nifti_img)
            nifti_breast = self.transform(nifti_breast)

        voxel_data = (torch.tensor(nifti_img.get_fdata()).unsqueeze(dim=0))



        voxel_breast = torch.tensor(nifti_breast.get_fdata()).unsqueeze(dim=0)


        thresh_breast = torch.zeros_like(voxel_breast)
        thresh_breast[voxel_breast>0.5] = 1
    
        

        return voxel_data, thresh_breast
    
def Dataset_load(image_paths, breast_paths,transforms=None):

    image_list = []
    breast_list = []

    for file in os.listdir(image_paths):
        image_list.append(os.path.join(image_paths, file))
    image_list.sort()

    for file in os.listdir(breast_paths):
        breast_list.append(os.path.join(breast_paths,file))
    breast_list.sort()

    #print(len(image_list), len(breast_list))

    assert len(image_list) == len(breast_list)

    MRI = 'mri'
    LABEL = 'label'
    subjects = []
    for (image_path, label_path) in zip(image_list, breast_list):
        subject = tio.Subject(
            MRI = tio.ScalarImage(image_path),
            LABEL = tio.LabelMap(label_path),
        )

        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects,transform=transforms)
    return dataset

if __name__ == "__main__":

    image_path = "/workspace/breast_mri/3d_train/input"
    label_path = "/workspace/breast_mri/3d_train/breast"

    transforms = tio.Compose([
        #tio.Resize(target_shape=(32, 512, 512)),
        tio.CropOrPad(target_shape=(32, 32, 32)),
        #tio.RescaleIntensity((0, 1)),  
        tio.RandomAffine(),  
    ])

    datasets = Dataset_load(image_path,label_path,transforms=transforms)
    


    # dataset = BreastDataset(image_path, label_path,transform=transforms)

    # train_size = int(len(dataset)*0.8)
    # validation_size = len(dataset)-(train_size)
    # train_dataset, val_dataset = random_split(dataset,[train_size, validation_size]) 

    train_loader = DataLoader(datasets,batch_size=1,shuffle=True)

    sample = next(iter(train_loader))
    print(sample["MRI"][tio.DATA].shape)
    print(sample["LABEL"][tio.DATA].shape)

    #print(torch.max(sample[0]))



