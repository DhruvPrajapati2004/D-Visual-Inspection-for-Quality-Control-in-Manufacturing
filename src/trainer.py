import torch
import time
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# Import CONFIG and logger from the main config file
from config import CONFIG, logger

# ------------------- TRAINING ENGINE -------------------
class Trainer:
    """
    Manages the training and validation loops for a PyTorch model.
    Includes functionality for logging, early stopping, and model checkpointing.
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: Any, criterion: torch.nn.Module):
        """
        Initializes the Trainer.

        Args:
            model (torch.nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): The optimizer (e.g., Adam, SGD).
            scheduler (Any): Learning rate scheduler (e.g., ReduceLROnPlateau).
            criterion (torch.nn.Module): The loss function.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = CONFIG.DEVICE # Get device from global config
        
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'valid_dice': [],
            'valid_iou': [],
            'lr': []
        }
        self.best_valid_score = -np.inf # Initialize with negative infinity for 'max' mode
        self.epochs_no_improve = 0 # Counter for early stopping patience
        logger.info("Trainer initialized.")

    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        """
        Runs a single training epoch.

        Args:
            train_loader (DataLoader): DataLoader for the training set.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train() # Set model to training mode
        running_loss = 0.0
        
        # Use tqdm for a progress bar during training
        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for images, masks in train_loader_tqdm:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad() # Zero the gradients
            
            outputs = self.model(images) # Forward pass
            loss = self.criterion(outputs, masks) # Calculate loss
            
            loss.backward() # Backward pass
            self.optimizer.step() # Update weights
            
            running_loss += loss.item() * images.size(0) # Accumulate loss
            train_loader_tqdm.set_postfix(loss=loss.item()) # Update progress bar postfix
            
        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f"Training Epoch Loss: {epoch_loss:.4f}")
        return epoch_loss

    def _validate_one_epoch(self, valid_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Runs a single validation epoch.

        Args:
            valid_loader (DataLoader): DataLoader for the validation set.

        Returns:
            Tuple[float, float, float]: Average validation loss, Dice score, and IoU score for the epoch.
        """
        self.model.eval() # Set model to evaluation mode
        running_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        
        # Use tqdm for a progress bar during validation
        valid_loader_tqdm = tqdm(valid_loader, desc="Validating", leave=False)
        with torch.no_grad(): # Disable gradient calculation for validation
            for images, masks in valid_loader_tqdm:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item() * images.size(0)

                # Calculate metrics (Dice and IoU)
                # Apply sigmoid to logits to get probabilities, then threshold to get binary masks
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float() # Binary predictions
                
                # Calculate Dice and IoU for each image in the batch and average
                batch_dice = 0.0
                batch_iou = 0.0
                for i in range(images.size(0)):
                    # For multi-label, iterate over classes for metrics
                    for c in range(CONFIG.NUM_CLASSES):
                        dice = self._dice_score(preds[i, c], masks[i, c])
                        iou = self._iou_score(preds[i, c], masks[i, c])
                        batch_dice += dice
                        batch_iou += iou
                
                total_dice += batch_dice / (images.size(0) * CONFIG.NUM_CLASSES)
                total_iou += batch_iou / (images.size(0) * CONFIG.NUM_CLASSES)

                valid_loader_tqdm.set_postfix(loss=loss.item())
                
        epoch_loss = running_loss / len(valid_loader.dataset)
        avg_dice = total_dice / len(valid_loader)
        avg_iou = total_iou / len(valid_loader)
        
        logger.info(f"Validation Epoch Loss: {epoch_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
        return epoch_loss, avg_dice, avg_iou

    def _dice_score(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Calculates the Dice coefficient for a single binary mask."""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()

    def _iou_score(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Calculates the IoU (Jaccard) coefficient for a single binary mask."""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Trains the model for a specified number of epochs.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            valid_loader (DataLoader): DataLoader for the validation set.

        Returns:
            Dict[str, List[float]]: A dictionary containing training history (loss, metrics, LR).
        """
        logger.info(f"Starting training for {CONFIG.EPOCHS} epochs.")
        os.makedirs(CONFIG.CHECKPOINT_DIR, exist_ok=True) # Ensure checkpoint directory exists
        
        for epoch in range(CONFIG.EPOCHS):
            logger.info(f"Epoch {epoch+1}/{CONFIG.EPOCHS}")
            
            train_loss = self._train_one_epoch(train_loader)
            valid_loss, valid_dice, valid_iou = self._validate_one_epoch(valid_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_dice'].append(valid_dice)
            self.history['valid_iou'].append(valid_iou)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Scheduler step (if using ReduceLROnPlateau, it needs a metric)
            if self.scheduler:
                # Assuming scheduler is ReduceLROnPlateau and monitors validation metric
                self.scheduler.step(valid_dice) # Monitor Dice score for scheduling

            # Check for best model and early stopping
            if valid_dice > self.best_valid_score:
                self.best_valid_score = valid_dice
                self.epochs_no_improve = 0
                checkpoint_path = os.path.join(CONFIG.CHECKPOINT_DIR, CONFIG.BEST_MODEL_NAME)
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"Model improved! Saving best model to {checkpoint_path} with Dice: {self.best_valid_score:.4f}")
            else:
                self.epochs_no_improve += 1
                logger.info(f"Model did not improve. Epochs without improvement: {self.epochs_no_improve}")
                if self.epochs_no_improve >= CONFIG.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break # Stop training

        logger.info("Training finished.")
        return self.history
