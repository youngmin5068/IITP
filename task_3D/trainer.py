import torch.nn as nn
from util import normalize_minmax
from config import *
from losses import BCEDiceLoss
from monai.transforms import AsDiscrete

def train(train_loader,net,optimizer,device,epoch,scheduler=None):
    net.train()
    post_label = AsDiscrete(threshold=0.5) #threshold
    criterion = BCEDiceLoss(alpha=1.0,beta=1.0)
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
        i += 1
            
            