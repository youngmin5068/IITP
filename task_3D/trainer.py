import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import DataLoader
from util import normalize_minmax
from config import *
from losses import BCEDiceLoss
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.data import ThreadDataLoader
from monai.data.utils import pad_list_data_collate 

def train(train_dataset,net,optimizer,device,epoch,scheduler=None):
    net.train()
    post_label = AsDiscrete(threshold=0.5) #threshold
    train_loader = ThreadDataLoader(train_dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True,collate_fn=pad_list_data_collate)
    criterion = DiceCELoss(sigmoid=True)
    scaler = torch.cuda.amp.GradScaler()

    i=0
    for batch in train_loader:

        imgs = batch['image'].to(device=device)
        true_masks = batch['label'].to(device=device)
        masks = post_label(true_masks) #TRUE Target

        with torch.cuda.amp.autocast():
            preds = net(imgs)
            loss = criterion(preds, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        nn.utils.clip_grad_value_(net.parameters(),0.1)

        if BATCH_SIZE*i % 20== 0 :
            print('epoch : {}, index : {}/{}, loss (batch) : {:.4f}'.format(
                                                                            epoch+1, 
                                                                            (i+1)*BATCH_SIZE,
                                                                            len(train_dataset),
                                                                            loss.detach())
                                                                            )
        i += 1
            
            

