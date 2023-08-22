import os
import torch
import torch.nn as nn
import logging
import torchio as tio
import torch.optim as optim
from trainer import train
from validation import validation
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataset_3d import BreastDataset,Dataset_load
from losses import DiceLoss
from util import set_seed
from config import *
import torchio as tio
import logging
from config import *
from losses import BCEDiceLoss
from monai.networks.nets import SwinUNETR




def main(net,device):

    image_paths = "/workspace/breast_mri/3d_train/input/"
    label_paths = "/workspace/breast_mri/3d_train/breast/"
    dir_checkpoint = "/workspace/breast_mri/dir_checkpoint_Breast_3D_Model"
    
    transforms = tio.Compose([
        tio.CropOrPad(target_shape=(32, 256, 256)),
        tio.RandomAffine(),
        #tio.RandomAffine(degrees=(5)),
    ])


    dataset = Dataset_load(image_paths,label_paths,transforms=transforms)


    train_size = int(TRAIN_RATIO *len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset,[train_size, validation_size])  


    logging.info(f'''Starting training:
        Epochs:          {EPOCH}
        Batch size:      {BATCH_SIZE}
        Train size:      {len(train_dataset)}
        Test size:       {len(val_dataset)}
        Learning rate:   {LearningRate}        
        Device:          {device}
    ''')   


    optimizer = optim.AdamW(net.parameters(),betas=(0.9,0.999),lr=LearningRate,weight_decay=5e-4) # weight_decay : prevent overfitting
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,150],gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20,T_mult=1,eta_min=0.000001,last_epoch=-1)
    loss = {"BCEDICE_loss": BCEDiceLoss(alpha=1.0,beta=1.0)}
    best_dice = 0.0
    best_epoch = 1

    for epoch in range(EPOCH):
        net.train()
        i=1
        train(train_dataset,net=net,optimizer=optimizer,loss_funcs=loss,epoch = epoch,device=device)


        print("----------validation start----------")
        net.eval()
        dice_score = validation(val_dataset=val_dataset,net=net,device=device)

        if dice_score > best_dice:
            best_dice = dice_score
            best_epoch = epoch+1
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError :
                pass
        torch.save(net.state_dict(), dir_checkpoint+ "/best_model_SwinUNETR.pth")
        logging.info(f'Checkpoint {epoch + 1} saved !')

        print("epoch : {} , best_dice : {:.4f}".format(best_epoch, best_dice))
        scheduler.step()


 
if __name__ == "__main__":
    set_seed(MODEL_SEED)

    net = SwinUNETR(img_size=(32,256,256),spatial_dims=3,in_channels=1,out_channels=1)
    #net = UNet3D(input_channels=1,num_classes=1)
    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net,device_ids=[1,2,3]) 
        print("data parallel ON")
    net.to(device=device)

    main(net=net,device=device)
