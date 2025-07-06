import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os

# Import CONFIG and logger from the main config file
from config import CONFIG, logger

# Import functions/classes from the new training package
from src.training.data_preparation import prepare_dataframe, SteelDataset
from src.training.model_utils import build_model, DiceBCELoss
from src.training.trainer import Trainer
from src.training.visualize import plot_history, visualize_predictions

# Import transformations from the main transforms module
from transforms import get_train_augs, get_test_transforms # get_test_transforms is used for validation augs


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility.")


if __name__ == '__main__':
    set_seed(CONFIG.SEED)
    logger.info("Starting model training execution block.")
    logger.info(f"Training images directory: {CONFIG.TRAIN_IMG_DIR}")
    logger.info(f"CSV path: {CONFIG.CSV_PATH}")

    # Ensure necessary directories exist
    os.makedirs(CONFIG.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG.CSV_PATH), exist_ok=True) # Ensure data dir exists if not already

    # --- 1. Prepare Data ---
    df = prepare_dataframe(CONFIG.CSV_PATH)
    logger.info(f"DataFrame prepared. Total unique images: {len(df)}")

    # --- 2. Stratified K-Fold Split ---
    # StratifiedKFold ensures that the proportion of samples with defects
    # is roughly the same in each fold as in the whole dataset.
    skf = StratifiedKFold(n_splits=CONFIG.N_SPLITS, shuffle=True, random_state=CONFIG.SEED)
    
    train_df, valid_df = None, None
    # Iterate through folds. For this example, we'll use only the first fold for training/validation.
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df['ImageId'], df['has_defect'])):
        if fold == 0: # Use the first fold for training and validation
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[valid_idx]
            break
    
    if train_df is None or valid_df is None:
        logger.error("Failed to create train/validation split. Ensure CSV and data are correctly formatted.")
        raise RuntimeError("Training split could not be created.")

    logger.info(f"Training set size: {len(train_df)} unique images")
    logger.info(f"Validation set size: {len(valid_df)} unique images")

    # --- 3. Create Datasets and Dataloaders ---
    # Get transformations for training (with augmentations) and validation (deterministic)
    train_augs = get_train_augs(CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH)
    valid_augs = get_test_transforms(CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH) # Using test transforms for validation

    train_dataset = SteelDataset(train_df, CONFIG.TRAIN_IMG_DIR, train_augs)
    valid_dataset = SteelDataset(valid_df, CONFIG.TRAIN_IMG_DIR, valid_augs)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.TRAIN_BATCH_SIZE, shuffle=True, num_workers=CONFIG.NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.VALID_BATCH_SIZE, shuffle=False, num_workers=CONFIG.NUM_WORKERS, pin_memory=True)
    logger.info("DataLoaders created.")

    # --- 4. Initialize Model and Training Components ---
    logger.info(f"Building model with encoder: {CONFIG.DEFAULT_ENCODER} on device: {CONFIG.DEVICE}")
    model = build_model(encoder_name=CONFIG.DEFAULT_ENCODER, device=CONFIG.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR, weight_decay=CONFIG.WEIGHT_DECAY)
    
    # Using ReduceLROnPlateau as per original notebook
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=CONFIG.SCHEDULER_PATIENCE, verbose=True
    )
    
    criterion = DiceBCELoss() # Using the custom combined loss

    # --- 5. Train the model ---
    logger.info("Starting model training...")
    trainer = Trainer(model, optimizer, scheduler, criterion)
    history = trainer.fit(train_loader, valid_loader)
    logger.info("Model training finished.")

    # --- 6. Post-training Visualization ---
    logger.info("Generating training history plots.")
    plot_history(history)

    # Load the best model for visualization
    best_model_path = os.path.join(CONFIG.CHECKPOINT_DIR, CONFIG.BEST_MODEL_NAME)
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path} for visualization.")
        best_model = build_model(encoder_name=CONFIG.DEFAULT_ENCODER, device=CONFIG.DEVICE) # Re-build model structure
        best_model.load_state_dict(torch.load(best_model_path, map_location=CONFIG.DEVICE))
        best_model.eval() # Set to evaluation mode for visualization
        
        logger.info("Visualizing predictions on validation data.")
        visualize_predictions(best_model, valid_dataset, num_samples=5)
    else:
        logger.warning(f"Best model not found at {best_model_path}. Skipping prediction visualization.")

    logger.info("Training script execution complete.")
