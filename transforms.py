"""
Module for defining image transformation pipelines using Albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any

# CONFIG is imported from the shared config module
from config import CONFIG

def get_validation_transforms(img_height: int, img_width: int) -> A.Compose:
    """
    Defines the Albumentations validation/test transformation pipeline.
    This pipeline resizes the image and normalizes its pixel values
    using ImageNet's mean and standard deviation, then converts it to a PyTorch tensor.

    Args:
        img_height (int): The target height for resizing.
        img_width (int): The target width for resizing.

    Returns:
        albumentations.Compose: An Albumentations composition of transformations.
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    return A.Compose([
        A.Resize(img_height, img_width),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])

