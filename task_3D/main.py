import os
import torch
import numpy as np
import torch.nn as nn
import logging
import torch.optim as optim
# from trainer import train
# from validation import validation
# from torch.utils.data import random_split
from dataset import data_dicts
from util import set_seed
from config import *
from losses import BCEDiceLoss
from monai.networks.nets import SwinUNETR
from monai.data import decollate_batch, DataLoader, Dataset
from monai.data.utils import pad_list_data_collate
from transforms import train_transforms,val_transforms
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric

def main(net,device):

    train_image_paths = "/workspace/breast_mri/tumor_train_3d/input"
    train_label_paths = "/workspace/breast_mri/tumor_train_3d/tumor"
    val_image_paths = "/workspace/breast_mri/tumor_tuning_3d/input"
    val_label_paths = "/workspace/breast_mri/tumor_tuning_3d/tumor"
    dir_checkpoint = "/workspace/IITP/task_3D/dir_checkpoint"

    #dicts = data_dicts(train_image_paths,train_label_paths)
    # train_dicts = dicts[370:]
    # val_dicts = dicts[0:50]
    # train_dicts = dicts[:int(len(dicts)*0.9)]
    # val_dicts = dicts[int(len(dicts)*0.9):]


    train_dataset = Dataset(data=data_dicts(train_image_paths,train_label_paths),
                            transform = train_transforms,
                            # cache_num=24,
                            # cache_rate=1.0,
                            # num_workers=8
                            )
    
    val_dataset = Dataset(data=data_dicts(val_image_paths,val_label_paths),
                            transform = val_transforms,
                            # cache_num=8,
                            # cache_rate=1.0,
                            # num_workers=4
                        )
    
    train_loader = DataLoader(train_dataset, num_workers=16, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True,collate_fn=pad_list_data_collate)
    val_loader =  DataLoader(val_dataset, num_workers=16, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True,collate_fn=pad_list_data_collate)
    
    logging.basicConfig(level=logging.INFO)

    logging.info(f'''Starting training:
        Epochs:          {EPOCH}
        Batch size:      {BATCH_SIZE}
        Train size:      {len(train_dataset)}
        Test size:       {len(val_dataset)}
        Learning rate:   {LEARNING_RATE}        
        Device:          {device}
    ''')

    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # weight_decay : prevent overfitting
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200],gamma=0.1)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=1e-5,last_epoch=-1)
    
    best_dice = 0.0
    best_epoch = 1

    validation_interval = 10
    best_dice = 0.0
    best_epoch = 1

    post_label = AsDiscrete(threshold=0.5) #threshold
    post_pred = AsDiscrete(threshold=0.5)
    diceCoeff = DiceMetric(include_background=True,reduction="mean")
    
    post_sigmoid = Activations(sigmoid=True)
    criterion = BCEDiceLoss(alpha=1.0,beta=1.0)
    losses = []
    dices = []
    for epoch in range(EPOCH):
        #train(train_loader,net=net,optimizer=optimizer,epoch=epoch,device=device)
        print("----------train start----------")
        net.train()
        i=0
        for batch in train_loader:

            imgs = batch['image'].to(device=device)
            masks = batch['label'].to(device=device)
            scaled_image = masks / 255.0
            true_masks = post_label(scaled_image) #TRUE Target 0 or 1

            optimizer.zero_grad()

            preds = net(imgs)
            loss = criterion(preds, true_masks)
            
            loss.backward()
            optimizer.step()

            nn.utils.clip_grad_value_(net.parameters(),0.1)

            if BATCH_SIZE*i % 100 == 0 :
                print('epoch : {}, index : {}/740, loss (batch) : {:.4f}'.format(
                                                                                epoch+1, 
                                                                                (i+1)*BATCH_SIZE,
                                                                                loss.detach())
                                                                                )
                losses.append(loss.detach())
            i += 1

        #if (epoch+1) % validation_interval:
        print("----------validation start----------")
            #dice_score = validation(val_loader,net=net,device=device)
        net.eval()

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device=device)
                true_masks = batch["label"].to(device=device)
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(imgs,IMAGE_SIZE, NUM_SLIDING, net)
                val_labels_list = decollate_batch(true_masks)
                val_labels_convert = [post_label(val_label_tensor/255.0) for val_label_tensor in val_labels_list]

                #val_outputs = net(imgs)
                val_outputs_list = decollate_batch(val_outputs) 
                val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]

                diceCoeff(y_pred=val_output_convert, y=val_labels_convert)
                
            mean_dice_val = diceCoeff.aggregate().item()
            dices.append(mean_dice_val)
            diceCoeff.reset()

        print("mean_dice_val : {:.4f}".format(mean_dice_val))
        
        if mean_dice_val > best_dice:
            best_dice = mean_dice_val
            best_epoch = epoch+1
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError :
                pass
            torch.save(net.state_dict(), dir_checkpoint+ "/best_model_2023_09_07.pth")
            logging.info(f'Checkpoint {epoch + 1} saved !')
        print("BEST EPOCH : {} , BEST_DICE : {:.4f}".format(best_epoch, best_dice))
            
        scheduler.step()
    return losses,dices



 
if __name__ == "__main__":

    set_seed(MODEL_SEED)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    torch.backends.cudnn.benchmark = True
    net = SwinUNETR(img_size=IMAGE_SIZE,spatial_dims=len(IMAGE_SIZE),in_channels=1,out_channels=1)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net,device_ids=[0,1,2,3]) 
        print("data parallel ON")
    net.to(device=device)

    losses, dices = main(net=net,device=device)

    np.save("/workspace/IITP/task_3D/dir_checkpoint/swinunetr_losses.npy",losses)
    np.save("/workspace/IITP/task_3D/dir_checkpoint/swinunetr_dices.npy",dices)
