import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from PIL import Image

# Import CONFIG from the main config file
from config import CONFIG, logger

# ------------------- PLOTTING TRAINING HISTORY -------------------
def plot_history(history: Dict[str, List[float]]):
    """
    Plots the training and validation loss, Dice score, IoU score, and learning rate over epochs.

    Args:
        history (Dict[str, List[float]]): A dictionary containing training history.
                                          Expected keys: 'train_loss', 'valid_loss',
                                          'valid_dice', 'valid_iou', 'lr'.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 10))

    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=4)
    plt.plot(epochs, history['valid_loss'], label='Valid Loss', marker='o', markersize=4)
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Dice Score
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['valid_dice'], label='Valid Dice Score', marker='o', markersize=4, color='green')
    plt.title('Validation Dice Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    # Plot IoU Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['valid_iou'], label='Valid IoU Score', marker='o', markersize=4, color='orange')
    plt.title('Validation IoU Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True)

    # Plot Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['lr'], label='Learning Rate', marker='o', markersize=4, color='red')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    logger.info("Training history plots displayed.")


# ------------------- VISUALIZING PREDICTIONS -------------------
def visualize_predictions(model: torch.nn.Module, dataset: torch.utils.data.Dataset, num_samples: int = 5):
    """
    Visualizes model predictions against ground truth masks for a few samples.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        dataset (torch.utils.data.Dataset): The dataset to sample from (e.g., validation dataset).
        num_samples (int): Number of samples to visualize.
    """
    model.eval() # Set model to evaluation mode
    
    # Ensure num_samples does not exceed dataset size
    num_samples = min(num_samples, len(dataset))
    
    logger.info(f"Visualizing {num_samples} predictions.")

    plt.figure(figsize=(20, num_samples * 4)) # Adjust figure size dynamically
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get a random sample from the dataset
            idx = np.random.randint(0, len(dataset))
            image_tensor, mask_true_tensor = dataset[idx]
            
            # Move image to device and add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(CONFIG.DEVICE)
            
            # Get model output (logits)
            output = model(image_tensor)
            
            # Convert logits to probabilities and then to binary masks
            mask_pred = (torch.sigmoid(output) > 0.5).squeeze(0).cpu().numpy() # Remove batch dim, to CPU, to NumPy
            mask_true = mask_true_tensor.cpu().numpy() # To CPU, to NumPy

            # Convert image tensor back to displayable format (HWC, 0-255)
            # Assuming image was normalized with ImageNet mean/std
            image_vis = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() # CHW to HWC
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_vis = image_vis * std + mean # Denormalize
            image_vis = np.clip(image_vis, 0, 1) # Clip to 0-1 range
            
            # Create subplots for original image + each defect class mask
            # +1 for the original image column
            num_cols = CONFIG.NUM_CLASSES + 1
            
            # Original Image
            ax = plt.subplot(num_samples, num_cols, i * num_cols + 1)
            ax.imshow(image_vis)
            ax.set_title(f"Sample {idx+1}\nOriginal Image")
            ax.axis('off')
            
            # Predicted vs. Ground Truth Masks for each class
            for c in range(CONFIG.NUM_CLASSES):
                ax = plt.subplot(num_samples, num_cols, i * num_cols + c + 2)
                
                # Create a combined mask for visualization:
                # Red channel for prediction, Green channel for ground truth
                combined_mask_vis = np.zeros((CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH, 3), dtype=np.float32)
                
                # Predicted mask (red)
                if mask_pred.shape[0] > c: # Ensure class index exists
                    combined_mask_vis[..., 0] = mask_pred[c, :, :] # Red channel for prediction
                
                # Ground truth mask (green)
                if mask_true.shape[0] > c: # Ensure class index exists
                    combined_mask_vis[..., 1] = mask_true[c, :, :] # Green channel for ground truth
                
                ax.imshow(combined_mask_vis)
                ax.set_title(f"{CONFIG.CLASSES[c]}\n(Green: True, Red: Pred)")
                ax.axis('off')
                
    plt.tight_layout()
    plt.show()
    logger.info("Prediction visualizations displayed.")
