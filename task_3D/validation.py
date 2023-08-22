import os
import torch
import torchio as tio
from torch.utils.data import DataLoader
from metrics import DiceCoefficient
from util import normalize_minmax
from config import *


def validation(val_dataset,net,device):
   
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=False,num_workers=12,pin_memory=True)
    diceCoeff = DiceCoefficient()
    dice = 0.0
    for batch in val_loader:

        imgs = normalize_minmax(batch["MRI"][tio.DATA].to(device=device,dtype=torch.float32))
        true_masks = batch["LABEL"][tio.DATA].to(device=device,dtype=torch.float32)
        true_thresh = torch.zeros_like(true_masks)
        true_thresh[true_masks>0.5] = 1.0

        with torch.no_grad():
            mask_pred = net(imgs)

        mask_pred = torch.sigmoid(mask_pred)
        thresh = torch.zeros_like(mask_pred)
        thresh[mask_pred>0.5] = 1
        dice += diceCoeff(thresh,true_thresh)

    print("dice : {:.4f}, length(val_loader) : {:.4f}".format(dice,len(val_loader)))
    print("dice score : {:.4f}".format(dice/len(val_loader)))

    return dice/len(val_loader)

