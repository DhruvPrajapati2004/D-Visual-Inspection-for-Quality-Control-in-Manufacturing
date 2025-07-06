"""
Centralized configuration for the Streamlit AI Surface Defect Detection app.
This module defines global constants, paths, settings, and state initializers.
"""

import torch
import logging
import logging.handlers
import os
import streamlit as st
import pandas as pd

# ------------------- LOGGING SETUP -------------------
# Configure logging for the entire application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------- APPLICATION CONFIGURATION -------------------
class AppConfig:
    """
    Defines all configuration parameters for the Streamlit application.
    """
    # Device Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image Dimensions (for model input and consistent display)
    IMG_HEIGHT, IMG_WIDTH = 128, 800

    # Defect Classes and their Colors (RGB tuples)
    CLASSES = ['Defect 1', 'Defect 2', 'Defect 3', 'Defect 4']
    COLORS = {
        'Defect 1': (235, 235, 0),   # Yellowish
        'Defect 2': (0, 210, 0),     # Greenish
        'Defect 3': (0, 150, 255),   # Bluish
        'Defect 4': (255, 50, 50)    # Reddish
    }

    # Model Configuration
    MODEL_PATH = "unet_qc_trained.pth"
    DEFAULT_ENCODER = "resnet34"
    SUPPORTED_ENCODERS = [
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "resnext50_32x4d", "resnext101_32x8d",
        "timm-resnest14d", "timm-resnest26d", "timm-resnest50d",
        "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3",
        "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7",
        "mobilenet_v2", "xception", "se_resnet50", "se_resnext50_32x4d",
        "densenet121", "densenet169", "densenet201",
        "vgg11", "vgg13", "vgg16", "vgg19"
    ]

    # Thresholds and Visualization Defaults
    DEFAULT_THRESHOLD = 0.5
    CONTOUR_WIDTH = 2
    DEFAULT_HEATMAP_MIN = 0.0
    DEFAULT_HEATMAP_MAX = 1.0

    # Supported Image Extensions
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']

    # Model Metadata
    MODEL_INFO = {
        "Name": "U-Net Segmentation Model for Surface Defects",
        "Version": "1.1.0",
        "Description": "A deep learning model for semantic segmentation of surface defects.",
        "Training Data": "Proprietary Industrial Dataset",
        "Input Size": f"{IMG_HEIGHT}x{IMG_WIDTH} (H x W)",
        "Output Classes": CLASSES,
        "Framework": "PyTorch",
        "Library": "segmentation_models.pytorch"
    }

# Instantiate the configuration
CONFIG = AppConfig()

# ------------------- SESSION STATE INITIALIZATION -------------------
def initialize_session_state():
    """
    Initializes all session state variables.
    This function is called once at the start of the app (in app.py)
    to ensure all keys are present in st.session_state.
    """
    defaults = {
        'current_model_path': CONFIG.MODEL_PATH,
        'selected_encoder_name': CONFIG.DEFAULT_ENCODER,
        'class_thresholds': {cls: CONFIG.DEFAULT_THRESHOLD for cls in CONFIG.CLASSES},
        'uploaded_batch': [],
        'model_loaded': False,
        'visible_classes': CONFIG.CLASSES,
        'file_uploader_key': 0,
        'camera_input_key': 0,
        'last_uploaded_files_hash': None,
        'batch_analysis_df': pd.DataFrame(),
        'heatmap_cmap_name': 'viridis',
        'last_uploaded_model_hash': None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

# ------------------- FILE-BASED LOGGING SETUP -------------------
def setup_file_logger():
    """Configures a rotating file handler for logging."""
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")

    root_logger = logging.getLogger()
    # Avoid adding duplicate handlers on Streamlit reruns
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == LOG_FILE_PATH for h in root_logger.handlers):
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
        logger.info(f"File logger configured at: {LOG_FILE_PATH}")
