import torch
import segmentation_models_pytorch as smp
from typing import Any

# Import CONFIG from the main config file
from config import CONFIG, logger

# ------------------- MODEL BUILDING -------------------
def build_model(encoder_name: str = CONFIG.DEFAULT_ENCODER, device: torch.device = CONFIG.DEVICE) -> smp.Unet:
    """
    Builds a U-Net segmentation model with a specified encoder.
    This function is used specifically for the training pipeline to create
    an *untrained* model instance.

    Args:
        encoder_name (str): The name of the encoder backbone to use (e.g., 'resnet34').
        device (torch.device): The device (e.g., 'cpu' or 'cuda') to load the model onto.

    Returns:
        smp.Unet: An untrained U-Net model in training mode.
    """
    logger.info(f"Building model with encoder: {encoder_name} on device: {device}")
    model = smp.Unet(
        encoder_name=encoder_name,      # Choose encoder backbone
        encoder_weights="imagenet",     # Use pre-trained weights for encoder
        in_channels=3,                  # Input image channels (RGB)
        classes=CONFIG.NUM_CLASSES,     # Number of output classes (defects)
        activation=None,                # No activation here, sigmoid will be applied in loss/inference
    )
    model.to(device)
    model.train() # Set model to training mode initially
    logger.info(f"Model built and set to training mode on {device}.")
    return model

# ------------------- LOSS FUNCTION -------------------
# Define a combined Dice and BCE loss function.
# This is a common and effective loss for segmentation tasks,
# balancing pixel-wise accuracy with region-based overlap.
class DiceBCELoss(torch.nn.Module):
    """
    Combines Dice Loss and Binary Cross-Entropy (BCE) Loss.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
        self.bce_loss = torch.nn.BCEWithLogitsLoss() # Applies sigmoid internally

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the combined Dice and BCE loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits, before sigmoid).
                                   Shape: (N, C, H, W)
            targets (torch.Tensor): Ground truth masks (binary, 0 or 1).
                                    Shape: (N, C, H, W)

        Returns:
            torch.Tensor: The calculated scalar loss value.
        """
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return dice + bce # Simple sum, can be weighted if needed
