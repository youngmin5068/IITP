import torch
from config import *

from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    ResizeWithPadOrCropd
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = Compose(
    [   
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        #ResizeWithPadOrCropd(keys=["image","label"],spatial_size=[192,512,512],method="symmetric", mode="constant"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-29722.0,
            a_max=32767.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes=AXCODES),
        Spacingd(
            keys=["image", "label"],
            pixdim=PIX_DIM,
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=IMAGE_SIZE,
            pos=1,
            neg=1,
            num_samples=NUM_SAMPLES,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)

val_transforms = Compose(
    [   
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        #ResizeWithPadOrCropd(keys=["image","label"],spatial_size=[192,512,512],method="symmetric", mode="constant"),
        ScaleIntensityRanged(keys=["image"], a_min=-29722.0, a_max=32767.0, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes=AXCODES),
        Spacingd(
            keys=["image", "label"],
            pixdim=PIX_DIM,
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ]
)


