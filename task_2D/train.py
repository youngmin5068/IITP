import logging
from timm.models.vision_transformer import _cfg
import os
import torch
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchio as tio
from custom_transforms import *
from dataset import breast_Dataset
from Model.AAUNET.AAU_Net import AAU_Net
from Model.TRANSUNET.transunet import TransUNet
from loss import *
from metric import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



dataset_path = "/workspace/breast_mri/2d_train"

def train_net(net,                       
              device,     
              epochs=100,
              batch_size=32,
              lr=0.001,
              save_cp=True
              ):
    
    dataset = breast_Dataset(dataset_path)
    train_ratio = 0.95
    train_length = int(train_ratio * len(dataset))
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset, [train_length, test_length])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=12,pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False,num_workers=12,pin_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=12,pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False,num_workers=12,pin_memory=True)

    dir_checkpoint = '/IITP/task_2D/dir_checkpoint'

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Train size:      {len(train_dataset)}
        Test size:       {len(test_dataset)}
        Learning rate:   {lr}        
        Checkpoints:     {save_cp}
        Device:          {device}
    ''')    

    optimizer = optim.AdamW(net.parameters(),betas=(0.9,0.999),lr=lr,weight_decay=5e-4) # weight_decay : prevent overfitting
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=0.000001,last_epoch=-1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[70],gamma=0.1)
    l1 = DiceLoss()
    l2 = nn.BCEWithLogitsLoss()

    best_dice = 0.0
    best_epoch = 0
    best_precision = 0.0
    best_recall = 0.0


    for epoch in range(epochs):

        net.train()
        i=1
        for imgs,true_masks in train_loader:

            imgs = imgs.to(device=device,dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)

            #optimizer.zero_grad()
            for param in net.parameters():
                param.grad = None

            masks_preds = net(imgs)

            loss1 = l1(torch.sigmoid(masks_preds),true_masks)
            loss2 = l2(masks_preds,true_masks)

            loss = loss1+loss2
            loss.backward()

            nn.utils.clip_grad_value_(net.parameters(), 0.1)     

            optimizer.step()

            
            if i*batch_size%1600 == 0:
                print('epoch : {}, index : {}/{},dice loss : {:.4f},bce loss : {:.4f}, loss (batch) : {:.4f}'.format(
                                                                                                                        epoch+1, 
                                                                                                                        i*batch_size,
                                                                                                                        len(train_dataset),
                                                                                                                        loss1.detach(),
                                                                                                                        loss2.detach(),
                                                                                                                        loss.detach())) 
            i += 1

        #when train epoch end
        print("--------------Validation start----------------")
        net.eval()      
        dice = 0.0
        recall = 0.0
        precision = 0.0

        for imgs, true_masks in val_loader:
            imgs = imgs.to(device=device,dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)
            with torch.no_grad():
                mask_pred = net(imgs)
                mask_pred = torch.sigmoid(mask_pred)

            thresh = torch.zeros_like(mask_pred)
            thresh[mask_pred > 0.5] = 1

            precision += precision_score(thresh,true_masks)
            recall += recall_score(thresh,true_masks)
            dice += dice_score(thresh,true_masks)
            

        print("dice score : {:.4f}, len(val_loader) : {:.4f}".format(dice, len(val_loader)))
        print("dice score : {:.4f}, recall score : {:.4f}, precision score : {:.4f}".format(dice/len(val_loader), recall/len(val_loader),precision/len(val_loader)) )
        scheduler.step()
        net.train()
        
        if dice/len(val_loader) > best_dice:
            best_dice = dice/len(val_loader)
            best_recall = recall/len(val_loader)
            best_precision = precision/len(val_loader)
            best_epoch = epoch+1

            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                torch.save(net.state_dict(), dir_checkpoint + f'/AAU_Net.pth')
                
                logging.info(f'Checkpoint {epoch + 1} saved !')

        print("epoch : {} , best_dice : {:.4f}, best_recall : {:.4f}, best_precision : {:.4f}".format(best_epoch, best_dice,best_recall,best_precision))
        

if __name__ == '__main__':
    Model_SEED = 980118
    set_seed(Model_SEED)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    net = AAU_Net()


    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net,device_ids=[0,1,2]) 
    net.to(device=device)

    train_net(net=net,device=device)

