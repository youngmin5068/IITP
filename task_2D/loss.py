import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):

        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        dice = (2 * intersection) / (union + 1e-8)  # Adding a small epsilon to avoid division by zero
        dice_loss = 1 - dice
        return dice_loss