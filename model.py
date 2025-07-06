"""
Module for loading and performing inference with the segmentation model.
"""

import os
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import logging
from typing import Any, Tuple
import io
import time

# Assuming CONFIG and logger are available from app's context or shared config
from config import CONFIG, logger
import streamlit as st # Imported for st.cache_resource and st.cache_data decorators

@st.cache_resource(show_spinner="Loading deep learning model... This may take a moment.")
def load_segmentation_model(model_path: str, device: torch.device, encoder_name: str) -> smp.Unet:
    """
    Loads the trained U-Net segmentation model.
    This function is cached to prevent re-loading the model on every Streamlit rerun.
    It attempts to load the model as a state_dict, and if that fails, as a full model object.

    Args:
        model_path (str): The file path to the saved PyTorch model state dictionary (.pth).
        device (torch.device): The device (e.g., 'cpu' or 'cuda') to load the model onto.
        encoder_name (str): The name of the encoder backbone to use (e.g., 'resnet34').

    Returns:
        smp.Unet: The loaded U-Net model in evaluation mode.

    Raises:
        FileNotFoundError: If the model file does not exist at the specified path.
        RuntimeError: If there's an issue loading the model (e.g., corrupt file, wrong architecture).
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure it's in the correct directory.")

    model = None
    try:
        # Attempt 1: Load as state_dict with ImageNet weights (most common scenario)
        try:
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=len(CONFIG.CLASSES),
                activation=None
            )
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded model state_dict with 'imagenet' weights from {model_path}.")
            else:
                raise TypeError("Loaded object is not a state_dict (expected dict).")
        except (RuntimeError, TypeError) as e:
            logger.warning(f"Attempt 1 failed ({e}). Trying to load state_dict without 'imagenet' weights or as full model object.")

            # Attempt 2: Load as state_dict without ImageNet weights
            try:
                model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=None, # Try without pretrained weights
                    in_channels=3,
                    classes=len(CONFIG.CLASSES),
                    activation=None
                )
                state_dict = torch.load(model_path, map_location=device)
                if isinstance(state_dict, dict):
                    model.load_state_dict(state_dict)
                    logger.info(f"Successfully loaded model state_dict with None weights from {model_path}.")
                else:
                    raise TypeError("Loaded object is not a state_dict (expected dict).")
            except (RuntimeError, TypeError) as e:
                logger.warning(f"Attempt 2 failed ({e}). Trying to load as a full model object.")

                # Attempt 3: Load the entire model object
                full_model_object = torch.load(model_path, map_location=device)
                if isinstance(full_model_object, torch.nn.Module):
                    model = full_model_object
                    logger.info(f"Successfully loaded full model object from {model_path}.")
                else:
                    raise TypeError(f"Loaded object from {model_path} is not a recognized PyTorch model type (torch.nn.Module or state_dict). Type: {type(full_model_object)}.")

    except Exception as e:
        logger.exception(f"Critical error during model loading from {model_path} with encoder '{encoder_name}':")
        raise RuntimeError(f"Failed to load the model from '{model_path}'. Please ensure it's a valid PyTorch model file compatible with the '{encoder_name}' encoder. Error: {e}")

    if model is None:
        raise RuntimeError("Model could not be initialized or loaded after all attempts.")

    model.to(device)
    model.eval() # Set model to evaluation mode for consistent predictions
    logger.info(f"Model {encoder_name} on {device} is ready.")
    return model

@st.cache_data(show_spinner=False)
def run_inference(image_bytes: bytes, _model_ref: Any, _transform: Any, device: torch.device) -> Tuple[np.ndarray, float]:
    """
    Performs inference on the input image using the loaded model.
    Takes image bytes as input for robust caching. The '_model_ref' and '_transform'
    arguments are prefixed with an underscore to tell Streamlit's caching mechanism
    not to hash them, as they are unhashable objects.

    Args:
        image_bytes (bytes): The raw bytes of the input image.
        _model_ref (Any): The loaded segmentation model (not hashed for caching). Can be smp.Unet or other torch.nn.Module.
        _transform (albumentations.Compose): The image transformation pipeline (not hashed for caching).
        device (torch.device): The device to perform inference on.

    Returns:
        Tuple[numpy.ndarray, float]:
            - A 3D NumPy array of probabilities [C, H, W] for each class.
            - The inference time in seconds.
    """
    try:
        _image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(_image.convert("RGB"))
        
        augmented = _transform(image=image_np)
        tensor = augmented['image'].unsqueeze(0).to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            logits = _model_ref(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        
        logger.info(f"Prediction complete. Probabilities shape: {probs.shape}, Inference time: {inference_time:.4f} seconds")
        return probs, inference_time
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise RuntimeError(f"Inference failed: {e}")

def get_model_summary_info(model: torch.nn.Module, dummy_input: torch.Tensor) -> str:
    """
    Generates a string summary of the PyTorch model using torchinfo.
    
    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
        dummy_input (torch.Tensor): A dummy input tensor matching the model's expected input shape.
        
    Returns:
        str: A string containing the model summary.
    """
    try:
        from torchinfo import summary
        model_summary_str = io.StringIO()
        summary(model, input_data=dummy_input, verbose=0, print_fn=lambda x: model_summary_str.write(x + '\n'))
        return model_summary_str.getvalue()
    except ImportError:
        return "torchinfo not installed. Please install it (`pip install torchinfo`) to view model summary."
    except Exception as e:
        logger.error(f"Error generating model summary: {e}")
        return f"Failed to generate model summary: {e}"

