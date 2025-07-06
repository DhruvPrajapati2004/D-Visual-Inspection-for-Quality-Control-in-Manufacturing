import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Dict, Any, Tuple

# Import CONFIG from the main config file
from config import CONFIG, logger

# ------------------- RLE DECODING UTILITY -------------------
# This function is used to convert Run-Length Encoding masks back into binary images.
# It's a common format for storing sparse segmentation masks efficiently.
def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): The RLE string (e.g., '1 2 3 4').
        shape (Tuple[int, int]): The (height, width) of the mask.

    Returns:
        numpy.ndarray: A 2D binary NumPy array representing the decoded mask.
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T # Reshape and transpose to get correct orientation


# ------------------- DATAFRAME PREPARATION -------------------
def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Prepares the dataframe from the CSV, handling RLE decoding and
    creating a 'has_defect' column for stratified splitting.

    Args:
        csv_path (str): Path to the CSV file containing image IDs and RLE masks.

    Returns:
        pd.DataFrame: Processed DataFrame with 'ImageId', 'ClassId', 'EncodedPixels',
                      and 'has_defect' columns.
    """
    logger.info(f"Loading dataframe from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Split 'ImageId_ClassId' into 'ImageId' and 'ClassId'
    df[['ImageId', 'ClassId']] = df['ImageId_ClassId'].str.split('_', expand=True)
    
    # Create a column to indicate if an image has any defect
    # This is crucial for stratified splitting to ensure each fold has a representative
    # number of defected and non-defected images.
    df['has_defect'] = df['EncodedPixels'].notna().astype(int)
    
    logger.info(f"DataFrame prepared. Total entries: {len(df)}")
    logger.info(f"Unique images: {df['ImageId'].nunique()}")
    logger.info(f"Images with defects: {df[df['has_defect'] == 1]['ImageId'].nunique()}")
    return df


# ------------------- PYTORCH DATASET CLASS -------------------
class SteelDataset(Dataset):
    """
    Custom PyTorch Dataset for loading steel images and their corresponding masks.
    Applies transformations (augmentations) to images and masks.
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, transforms: Any = None):
        """
        Initializes the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing 'ImageId', 'ClassId', 'EncodedPixels'.
            img_dir (str): Directory where image files are stored.
            transforms (albumentations.Compose, optional): Albumentations compose object for data augmentation.
        """
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Get unique image IDs
        self.image_ids = self.df['ImageId'].unique()
        logger.info(f"SteelDataset initialized with {len(self.image_ids)} unique images.")

    def __len__(self) -> int:
        """Returns the total number of unique images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves an image and its corresponding multi-class mask.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed image tensor
                                               and the multi-class mask tensor.
        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.img_dir, image_id)
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB")) # Ensure RGB format
        
        # Initialize an empty mask for all classes
        # The mask will have shape [num_classes, height, width]
        masks = np.zeros((CONFIG.NUM_CLASSES, CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH), dtype=np.float32)
        
        # Filter dataframe for the current image and iterate through its defect classes
        image_df = self.df[self.df['ImageId'] == image_id]
        for i, class_id in enumerate(CONFIG.CLASSES):
            # Find the RLE mask for the current class_id
            class_mask_row = image_df[image_df['ClassId'] == class_id]
            if not class_mask_row.empty and pd.notna(class_mask_row['EncodedPixels'].iloc[0]):
                rle_mask = class_mask_row['EncodedPixels'].iloc[0]
                decoded_mask = rle_decode(rle_mask, (image.shape[0], image.shape[1]))
                masks[i, :, :] = decoded_mask # Assign decoded mask to the correct class channel
        
        # Apply transformations (augmentations) to both image and masks
        if self.transforms:
            # Albumentations expects masks as a list of 2D arrays if multi-label,
            # or a single 3D array if multi-channel. Here, we pass the 3D array directly.
            # It will apply transformations consistently to all channels.
            augmented = self.transforms(image=image, masks=masks.transpose(1, 2, 0)) # Convert to HWC for Albumentations
            image = augmented['image']
            masks = augmented['masks'].transpose(2, 0, 1) # Convert back to CHW for PyTorch
        
        return image, torch.from_numpy(masks) # Return image and masks as PyTorch tensors
