import torch
import torch.nn as nn
import torchio as tio
import logging
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataset_3d import BreastDataset
from losses import DiceLoss
from util import set_seed, normalize_minmax
from config import *

def train(train_dataset,net,optimizer,loss_funcs,device,epoch,scheduler=None):

    #patch_dataset = tio.SubjectsDataset(train_dataset,transform=transform)
    # patch_size = (64,64,64)
    # sampler = tio.data.UniformSampler(patch_size=patch_size)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True,num_workers=12,pin_memory=True)

    i=0
    for batch in train_loader:

        imgs = normalize_minmax(batch["MRI"][tio.DATA].to(device=device,dtype=torch.float32))
        true_masks = batch["LABEL"][tio.DATA].to(device=device,dtype=torch.float32)
        true_thresh = torch.zeros_like(true_masks)
        true_thresh[true_masks>0.5] = 1.0

        for param in net.parameters():
            param.grad = None

        masks_preds = net(imgs)

        criterion = loss_funcs["BCEDICE_loss"]
        loss = criterion(masks_preds,true_thresh)


        loss.backward()

        nn.utils.clip_grad_value_(net.parameters(),0.1)

        optimizer.step()

        if BATCH_SIZE*i % 20== 0 :
            print('epoch : {}, index : {}/{}, loss (batch) : {:.4f}'.format(
                                                                            epoch+1, 
                                                                            (i+1)*BATCH_SIZE,
                                                                            len(train_dataset),
                                                                            loss.detach())
                                                                            )
        i += 1
            
            

