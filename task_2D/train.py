import logging
from timm.models.vision_transformer import _cfg
from PIL import Image
import os
import torch
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchio as tio
from custom_transforms import *
from dataset import tumor_Dataset
from Model.AAUNET.AAU_Net import AAU_Net
from Model.TRANSUNET.transunet import TransUNet
from loss import *
from metric import *
from monai.networks.nets import SwinUNETR
from Model.MKA.LK_PC import LK_PC_UNet
from gmic_UNet import GMIC_UNet
from modified_PCCA_UNET import pcca_UNet
from topTcbam_UNet import top_t_cbam_UNet
from subpixel_swin import subpixel_SwinUNetr
from subpixelUNet import SubPixelUNet
from cbam_swin import SwinUNETR_

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def train_net(net,
              roi_model,          
              device,     
              epochs=100,
              batch_size=32,
              lr=0.001,
              save_cp=True
              ):
    dir_checkpoint = '/workspace/IITP/task_2D/dir_checkpoint_tumorResult'
    train_dataset_path = "/mount_folder/sampling/train/balance"
    tuning_dataset_path = "/mount_folder/sampling/tuning/balance"
    test_dataset_path = "/mount_folder/sampling/test/balance"

    train_dataset = tumor_Dataset(train_dataset_path)
    tuning_dataset = tumor_Dataset(tuning_dataset_path)
    test_dataset = tumor_Dataset(test_dataset_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=12,pin_memory=True)
    val_loader = DataLoader(tuning_dataset, batch_size=batch_size,shuffle=False,num_workers=12,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False,num_workers=12,pin_memory=True)
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Train size:      {len(train_dataset)}
        Test size:       {len(test_dataset)}
        Learning rate:   {lr}        
        Checkpoints:     {save_cp}
        Device:          {device}
    ''')    

    optimizer = optim.AdamW(net.parameters(),betas=(0.9,0.999),lr=lr,weight_decay=1e-4) # weight_decay : prevent overfitting
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2,eta_min=0.0001,last_epoch=-1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[40,80],gamma=0.1)
    diceloss = DiceLoss()
    bceloss = nn.BCEWithLogitsLoss()
    classifyloss = nn.BCELoss()

    best_dice = 0.0
    best_epoch = 0
    best_precision = 0.0
    best_recall = 0.0

    for epoch in range(epochs):

        net.train()
        roi_model.eval()
        i=1
        for imgs,true_masks,true_class in train_loader:

            imgs = imgs.to(device=device,dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)
            true_class = true_class.unsqueeze(1).to(device=device,dtype=torch.float32)
            #optimizer.zero_grad()
            for param in net.parameters():
                param.grad = None

            #masks_preds,cl_preds,saliency_map = net(imgs)

            with torch.no_grad():
                roi_preds = torch.sigmoid(roi_model(imgs))
                roi_thresh = torch.zeros_like(roi_preds)
                roi_thresh[roi_preds>0.5] = 1.0
                roi_results = imgs * roi_thresh

            masks_preds,class_preds = net(imgs)
            loss1 = diceloss(torch.sigmoid(masks_preds),true_masks)
            loss2 = bceloss(masks_preds,true_masks)
            loss3 = classifyloss(torch.sigmoid(class_preds),true_class)

            loss = loss1+loss2+loss3
            loss.backward()

            nn.utils.clip_grad_value_(net.parameters(), 0.1)     

            optimizer.step()

            if i*batch_size%800 == 0:
                print('epoch : {}, index : {}/{}, dice loss : {:.4f}, bce loss : {:.4f}, class loss : {:.4f},total loss : {:.4f}'.format(
                                                                                epoch+1, 
                                                                                i*batch_size,
                                                                                len(train_dataset),
                                                                                loss1.detach(),
                                                                                loss2.detach(),
                                                                                loss3.detach(),
                                                                                loss.detach())) 
            i += 1

        #when train epoch end
        print("--------------Validation start----------------")
        net.eval()      
        dice = 0.0
        recall = 0.0
        precision = 0.0

        for imgs, true_masks,_ in val_loader:
            imgs = imgs.to(device=device,dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)
            with torch.no_grad():
                roi_preds = torch.sigmoid(roi_model(imgs))
                roi_thresh = torch.zeros_like(roi_preds)
                roi_thresh[roi_preds>0.5] = 1.0
                roi_results = imgs * roi_thresh

                mask_pred,_ = net(imgs)
                mask_pred = torch.sigmoid(mask_pred)

            thresh = torch.zeros_like(mask_pred)
            thresh[mask_pred > 0.5] = 1.0

            precision += precision_score(thresh,true_masks)
            recall += recall_score(thresh,true_masks)
            dice += dice_score(thresh,true_masks)
            

        print("dice score : {:.4f}, len(val_loader) : {:.4f}".format(dice, len(val_loader)))
        print("dice score : {:.4f}, recall score : {:.4f}, precision score : {:.4f}".format(dice/len(val_loader), recall/len(val_loader),precision/len(val_loader)) )
        scheduler.step()
        
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
                torch.save(net.state_dict(), dir_checkpoint + f'/SubPixel_SwinUNetr_dataedited.pth')
                
                logging.info(f'Checkpoint {epoch + 1} saved !')

        print("epoch : {} , best_dice : {:.4f}, best_recall : {:.4f}, best_precision : {:.4f}".format(best_epoch, best_dice,best_recall,best_precision))

    print("--------------------------TEST------------------------")

    net.eval()
    precision=0.0
    recall=0.0
    dice=0.0

    for imgs, true_masks,_ in test_loader:
        imgs = imgs.to(device=device,dtype=torch.float32)
        true_masks = true_masks.to(device=device,dtype=torch.float32)

        with torch.no_grad():
            # roi_preds = torch.sigmoid(roi_model(imgs))
            # roi_thresh = torch.zeros_like(roi_preds)
            # roi_thresh[roi_preds>0.5] = 1.0

            # roi_results = imgs * roi_thresh

            mask_pred,_ = net(imgs)
            mask_pred = torch.sigmoid(mask_pred)
        thresh = torch.zeros_like(mask_pred)
        thresh[mask_pred > 0.5] = 1.0

        precision += precision_score(thresh,true_masks)
        recall += recall_score(thresh,true_masks)
        dice += dice_score(thresh,true_masks)

    print("dice score : {:.4f}, len(testloader) : {:.4f}".format(dice, len(test_loader)))
    print("dice score : {:.4f}, recall score : {:.4f}, precision score : {:.4f}".format(dice/len(test_loader), recall/len(test_loader),precision/len(test_loader)) )
        

if __name__ == '__main__':
    Model_SEED = 7777777
    set_seed(Model_SEED)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # parameters = {
    #         "percent_t":0.1,
    #         "device_type":"gpu",
    #         "gpu_number":2,
    #         "post_processing_dim": 256,
    #         "num_classes": 1
    #     }

    # net = GMIC_UNet(parameters).to(device=device)
    #net = pcca_UNet(1,1).to(device=device)
    #net = SPi_UNet(1,1).to(device=device)
    #net = top_t_cbam_UNet(1,1,0.02).to(device=device)
    #net = SubPixelUNet(1,1).to(device=device) 
    #net = LK_PC_UNet(1, 1).to(device=device)
    net = subpixel_SwinUNetr(img_size=(512,512),spatial_dims=2,in_channels=1,out_channels=1,depths=(2,2,2,2)).to(device=device)
    #net = SwinUNETR((512,512), spatial_dims=2,in_channels=1, out_channels=1,depths=(2,2,2,2)).to(device=device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net,device_ids=[0,1,2])


    model_path = '/workspace/IITP/task_2D/dir_checkpoint_breast_ROI/ROI_cbam-swinUNetr.pth'

    roi_model = SwinUNETR_(img_size=(512,512),spatial_dims=2,in_channels=1,out_channels=1,depths=(2,2,2,2)).to(device=device)
    if torch.cuda.device_count() > 1:
        roi_model = nn.DataParallel(roi_model,device_ids=[0,1,2]) 

    roi_model.load_state_dict(torch.load(model_path))

    train_net(net=net,roi_model=roi_model,device=device)



