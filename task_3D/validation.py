import os
import torch
import torchio as tio
from torch.utils.data import DataLoader
from monai.data.utils import pad_list_data_collate 
from metrics import DiceCoefficient
from util import normalize_minmax
from config import *
from monai.data import ThreadDataLoader,decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

def validation(val_dataset,net,device):

    net.eval()
    post_label = AsDiscrete(threshold=0.5) #threshold
    post_pred = AsDiscrete(threshold=0.5)
    val_loader =  ThreadDataLoader(val_dataset, num_workers=0, batch_size=BATCH_SIZE,collate_fn=pad_list_data_collate)
    diceCoeff = DiceMetric(include_background=False,reduction="mean", get_not_nans=False)

    with torch.no_grad():
        for batch in val_loader:

            imgs = batch['image'].to(device=device)
            true_masks = batch["label"].to(device=device)
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(imgs,IMAGE_SIZE, NUM_SLIDING, net)
            val_labels_list = decollate_batch(true_masks)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]

            val_outputs_list = decollate_batch(val_outputs) # check...
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            diceCoeff(y_pred=val_output_convert, y=val_labels_convert)
            
        mean_dice_val = diceCoeff.aggregate().item()
        diceCoeff.reset()
    #print("dice : {:.4f}, length(val_loader) : {:.4f}".format(dice,len(val_loader)))
    print("dice score : {:.4f}".format(mean_dice_val))

    return mean_dice_val

