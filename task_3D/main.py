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
from dataset import data_dicts
from util import set_seed
from config import *
import torchio as tio
import logging
from config import *
from losses import BCEDiceLoss
from monai.networks.nets import SwinUNETR
from monai.data import CacheDataset
from transforms import train_transforms,val_transforms

def main(net,device):

    train_image_paths = "/workspace/breast_mri/tumor_train_3d/input"
    train_label_paths = "/workspace/breast_mri/tumor_train_3d/tumor"
    val_image_paths = "/workspace/breast_mri/tumor_tuning_3d/input"
    val_label_paths = "/workspace/breast_mri/tumor_tuning_3d/tumor"
    dir_checkpoint = "/workspace/IITP/task_3D/dir_checkpoint"

    # dicts = data_dicts(train_image_paths,train_label_paths)
    # train_dicts = dicts[:int(len(dicts)*0.95)]
    # val_dicts = dicts[int(len(dicts)*0.95):]


    train_dataset = CacheDataset(data=data_dicts(train_image_paths,train_label_paths),
                            transform = train_transforms,
                            cache_num = 24,
                            cache_rate=1.0,
                            num_workers=8)
    
    val_dataset = CacheDataset(data=data_dicts(val_image_paths,val_label_paths),
                            transform = val_transforms,
                            cache_num = 6,
                            cache_rate=1.0,
                            num_workers=4)

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
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=100,T_mult=1,eta_min=0.000001,last_epoch=-1)
    
    best_dice = 0.0
    best_epoch = 1

    for epoch in range(EPOCH):
        train(train_dataset,net=net,optimizer=optimizer,epoch=epoch,device=device)
        print("----------validation start----------")
        dice_score = validation(val_dataset=val_dataset,net=net,device=device)

        if dice_score > best_dice:
            best_dice = dice_score
            best_epoch = epoch+1
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError :
                pass
        torch.save(net.state_dict(), dir_checkpoint+ "/just_testing.pth")
        logging.info(f'Checkpoint {epoch + 1} saved !')

        print("epoch : {} , best_dice : {:.4f}".format(best_epoch, best_dice))
        scheduler.step()


 
if __name__ == "__main__":

    set_seed(MODEL_SEED)
    torch.backends.cudnn.benchmark = True
    net = SwinUNETR(img_size=IMAGE_SIZE,spatial_dims=len(IMAGE_SIZE),in_channels=1,out_channels=1,depths=(2,2,2,2))
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net,device_ids=[0,1,2,3]) 
        print("data parallel ON")
    net.to(device=device)

    main(net=net,device=device)
