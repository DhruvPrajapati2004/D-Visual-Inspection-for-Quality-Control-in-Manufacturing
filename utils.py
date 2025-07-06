"""
Utility functions for image processing, mask manipulation, and Streamlit helpers.
"""

import numpy as np
from PIL import Image, UnidentifiedImageError
import hashlib
from io import BytesIO
import logging
from typing import List, Tuple, Dict, Any
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

from config import CONFIG, logger

@st.cache_data
def mask2contour_cv2(mask: np.ndarray) -> np.ndarray:
    """
    Generates a contour from a binary mask using OpenCV's findContours.

    Args:
        mask (numpy.ndarray): A 2D binary mask (0s and 1s, uint8 type).

    Returns:
        numpy.ndarray: A 2D binary array (0s and 1s) representing the contour of the mask.
    """
    # Ensure mask is uint8 and binary (0 or 255) for findContours
    mask_for_cv2 = mask.astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask_for_cv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty image to draw contours on
    contour_image = np.zeros_like(mask, dtype=np.uint8)
    
    # Draw contours. -1 means draw all contours.
    cv2.drawContours(contour_image, contours, -1, 1, CONFIG.CONTOUR_WIDTH) # Draw with CONFIG.CONTOUR_WIDTH thickness
    
    return contour_image

@st.cache_data
def get_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    """
    Calculates the bounding box (x, y, width, height) for a binary mask.

    Args:
        mask (numpy.ndarray): A 2D binary mask (0s and 1s).

    Returns:
        Tuple[int, int, int, int] | None: (x, y, width, height) of the bounding box,
                                        or None if the mask is empty.
    """
    if np.sum(mask) == 0:
        return None # No defect detected
    
    # Find coordinates of all active pixels
    coords = np.argwhere(mask == 1)
    if coords.size == 0: # Should not happen if sum > 0, but as a safeguard
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # OpenCV format: (x, y, width, height)
    x = int(x_min)
    y = int(y_min)
    width = int(x_max - x_min + 1)
    height = int(y_max - y_min + 1)
    
    return (x, y, width, height)

@st.cache_data
def create_confidence_heatmap(
    probabilities: np.ndarray, # Raw probabilities [C, H, W]
    original_image_np: np.ndarray, # Original image as NumPy array (H, W, 3) - already resized
    alpha: float = 0.5, # Transparency of the heatmap overlay
    cmap_name: str = 'viridis' # Colormap to use (now dynamic)
) -> np.ndarray:
    """
    Generates an overlay that blends the original image with a composite confidence heatmap
    for all defect classes. Higher probabilities result in more intense color overlay.

    Args:
        probabilities (numpy.ndarray): A 3D NumPy array of raw probability masks [C, H, W].
        original_image_np (numpy.ndarray): The original image as a NumPy array (H, W, 3).
        alpha (float): Transparency of the heatmap overlay (0.0 to 1.0).
        cmap_name (str): Name of the matplotlib colormap to use.

    Returns:
        numpy.ndarray: A 3D NumPy array representing the image with a blended confidence heatmap.
    """
    if probabilities.size == 0:
        logger.warning("Empty probabilities array for confidence heatmap.")
        return original_image_np # Return original if no probabilities

    # Take the maximum probability across all classes for each pixel
    # This creates a single 2D heatmap representing overall "defectness"
    overall_confidence_map = np.max(probabilities, axis=0) # (H, W)

    # Normalize to 0-1 for colormap application
    norm_confidence_map = (overall_confidence_map - overall_confidence_map.min()) / \
                          (overall_confidence_map.max() - overall_confidence_map.min() + 1e-8) # Add epsilon for stability

    # Apply colormap
    cmap = cm.get_cmap(cmap_name) # Use the dynamic cmap_name
    heatmap_colored = cmap(norm_confidence_map)[:, :, :3] # Get RGB, discard alpha from cmap

    # Convert to 0-255 uint8
    heatmap_colored_uint8 = (heatmap_colored * 255).astype(np.uint8)

    # Blend with original image
    blended_img = cv2.addWeighted(original_image_np, 1 - alpha, heatmap_colored_uint8, alpha, 0)
    return blended_img

@st.cache_data
def create_class_heatmap(
    probability_map: np.ndarray, # Raw probability map for a single class [H, W]
    original_image_np: np.ndarray, # Original image as NumPy array (H, W, 3) - already resized
    alpha: float = 0.5, # Transparency of the heatmap overlay
    cmap_name: str = 'viridis', # Colormap to use (now dynamic)
    min_val: float = 0.0, # Min probability for color mapping
    max_val: float = 1.0 # Max probability for color mapping
) -> np.ndarray:
    """
    Generates a heatmap for a single class's probability map and blends it with the original image.

    Args:
        probability_map (numpy.ndarray): A 2D NumPy array of probabilities for a single class.
        original_image_np (numpy.ndarray): The original image as a NumPy array (H, W, 3).
        alpha (float): Transparency of the heatmap overlay (0.0 to 1.0).
        cmap_name (str): Name of the matplotlib colormap to use.
        min_val (float): Minimum probability value for color mapping.
        max_val (float): Maximum probability value for color mapping.

    Returns:
        numpy.ndarray: A 3D NumPy array representing the image with a blended class-specific heatmap.
    """
    if probability_map.size == 0:
        logger.warning("Empty probability map for class heatmap.")
        return original_image_np # Return original if no probabilities

    # Normalize probabilities to the [min_val, max_val] range for colormapping
    # Clip to ensure values are within [0, 1] after normalization
    norm_prob_map = np.clip((probability_map - min_val) / (max_val - min_val + 1e-8), 0, 1)

    # Apply colormap
    cmap = cm.get_cmap(cmap_name) # Use the dynamic cmap_name
    heatmap_colored = cmap(norm_prob_map)[:, :, :3] # Get RGB, discard alpha from cmap

    # Convert to 0-255 uint8
    heatmap_colored_uint8 = (heatmap_colored * 255).astype(np.uint8)

    # Blend with original image
    blended_img = cv2.addWeighted(original_image_np, 1 - alpha, heatmap_colored_uint8, alpha, 0)
    return blended_img


@st.cache_data
def overlay_predictions_on_image(
    image_bytes: bytes,
    predictions: np.ndarray, # Raw probabilities [C, H, W]
    colors: Dict[str, Tuple[int, int, int]],
    img_width: int,
    img_height: int,
    selected_defects_for_display: List[str], # Classes selected by user for visual display
    class_thresholds: Dict[str, float] # Per-class thresholds from sidebar
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Overlays predicted defect masks as colored contours and semi-transparent fills
    on the original image. Contours and fills are drawn only for defect types
    selected by the user. Also calculates bounding boxes and returns detailed defect info.

    Args:
        image_bytes (bytes): The raw bytes of the original input image.
        predictions (numpy.ndarray): A 3D NumPy array of raw probability masks [C, H, W].
        colors (Dict[str, tuple[int, int, int]]): A dictionary mapping class names to RGB colors.
        img_width (int): The target width for the overlay image.
        img_height (int): The target height for the overlay image.
        selected_defects_for_display (list[str]): A list of defect class names selected by the user for visual overlay.
        class_thresholds (Dict[str, float]): Dictionary of per-class thresholds.

    Returns:
        tuple[numpy.ndarray, list[Dict[str, Any]]]:
            - A 3D NumPy array representing the image with overlaid contours and fills.
            - A list of dictionaries, each containing detailed information about a detected defect
              (class_name, detected, area_pixels, bounding_box, avg_confidence, threshold_used).
    """
    # Re-open the image from bytes and resize to display dimensions
    _original_image_pil = Image.open(io.BytesIO(image_bytes))
    overlay_img = np.array(_original_image_pil.resize((img_width, img_height)).convert("RGB")).copy()
    
    # Create a blank overlay image for semi-transparent fills
    fill_overlay = np.zeros((img_height, img_width, 4), dtype=np.uint8) # RGBA

    detailed_defect_info = [] # List to store structured defect info

    for i, class_name in enumerate(CONFIG.CLASSES):
        # Determine which threshold to use for THIS specific defect's detection
        current_threshold_for_detection = class_thresholds.get(class_name, CONFIG.DEFAULT_THRESHOLD)
        
        # Binarize the mask based on the determined threshold
        bin_mask = (predictions[i] > current_threshold_for_detection).astype(np.uint8)
        
        area = np.sum(bin_mask) # Calculate the total area of the detected defect
        
        bounding_box = get_bounding_box(bin_mask) # Get bounding box
        
        # Calculate average confidence within the detected area
        confidence = 0.0
        if area > 0:
            # Only consider probabilities where the binary mask is 1
            relevant_probs = predictions[i][bin_mask == 1]
            if relevant_probs.size > 0:
                confidence = np.mean(relevant_probs)

        defect_entry = {
            'class_name': class_name,
            'detected': area > 0,
            'area_pixels': int(area),
            'area_percentage': (area / (img_width * img_height) * 100) if (img_width * img_height) > 0 else 0.0,
            'bounding_box': bounding_box, # (x, y, width, height) or None
            'avg_confidence': float(confidence), # Average probability in detected region
            'threshold_used': current_threshold_for_detection # Store the threshold actually used for this detection
        }
        detailed_defect_info.append(defect_entry)

        # Only draw overlay if defect is detected AND it's selected for display
        if area > 0 and class_name in selected_defects_for_display:
            # Get color for current defect
            color_rgb = colors[class_name]
            
            # --- Apply semi-transparent fill ---
            fill_mask = bin_mask * 255 # Scale to 0 or 255 for alpha channel
            fill_color_rgba = (*color_rgb, 100) # 100 is alpha (out of 255) for semi-transparency
            
            fill_overlay[fill_mask == 255, :3] = fill_color_rgba[:3] # Set RGB channels
            fill_overlay[fill_mask == 255, 3] = fill_color_rgba[3]   # Set Alpha channel

            # --- Apply contour ---
            contour = mask2contour_cv2(bin_mask) # Use the new cv2-based contour function
            
            # Apply the defect-specific color to the contour pixels on the overlay image
            overlay_img[contour == 1] = color_rgb

            # --- Draw Bounding Box and Label on overlay_img ---
            if bounding_box:
                x, y, w, h = bounding_box
                # Draw rectangle (image, start_point, end_point, color, thickness)
                cv2.rectangle(overlay_img, (x, y), (x + w, y + h), color_rgb, 2) # 2 pixels thick
                # Put text (class name) near the bounding box
                # Adjust text position to be above the box if possible, otherwise below
                text_pos_y = y - 10
                if text_pos_y < 15: # If too close to top, put below
                    text_pos_y = y + h + 20
                cv2.putText(overlay_img, class_name, (x, text_pos_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_rgb, 2, cv2.LINE_AA)

    # Blend the fill_overlay with the original image
    # Convert overlay_img to RGBA for blending
    overlay_img_rgba = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2RGBA)
    
    # Alpha blend: output = (foreground * alpha) + (background * (1 - alpha))
    alpha_fill = fill_overlay[:, :, 3] / 255.0 # Alpha channel of fill_overlay
    alpha_fill = np.expand_dims(alpha_fill, axis=2) # Expand to 3 dimensions for broadcasting
    
    # Blend RGB channels
    blended_rgb = (fill_overlay[:, :, :3] * alpha_fill + overlay_img_rgba[:, :, :3] * (1 - alpha_fill)).astype(np.uint8)
    
    # Keep the original image's alpha channel (if any), or just make it opaque
    blended_alpha = np.maximum(fill_overlay[:, :, 3], overlay_img_rgba[:, :, 3]) # Max of alpha channels
    blended_img_rgba = np.concatenate((blended_rgb, np.expand_dims(blended_alpha, axis=2)), axis=2)

    # Convert back to RGB for Streamlit display
    final_output_img = cv2.cvtColor(blended_img_rgba, cv2.COLOR_RGBA2RGB)

    return final_output_img, detailed_defect_info

def get_defect_summary_strings(defect_info_list: List[Dict[str, Any]], selected_classes: List[str]) -> List[str]:
    """
    Generates formatted summary strings for detected defects.

    Args:
        defect_info_list (List[Dict[str, Any]]): List of defect info dictionaries.
        selected_classes (List[str]): List of classes selected for display.

    Returns:
        List[str]: Formatted strings for detected defects.
    """
    summary_strings = []
    for info in defect_info_list:
        if info['detected'] and info['class_name'] in selected_classes:
            bbox_str = f"({info['bounding_box'][0]}, {info['bounding_box'][1]}, {info['bounding_box'][2]}, {info['bounding_box'][3]})" if info['bounding_box'] else "N/A"
            summary_strings.append(
                f"<b>{info['class_name']}</b>: Area {info['area_pixels']} px² ({info['area_percentage']:.2f}%), "
                f"Confidence {info['avg_confidence']:.4f}, Bounding Box {bbox_str}"
            )
    return summary_strings

def calculate_defect_metrics(defect_info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates overall defect metrics from a list of defect information.

    Args:
        defect_info_list (List[Dict[str, Any]]): List of defect info dictionaries.

    Returns:
        Dict[str, Any]: Dictionary containing overall metrics.
    """
    total_defects = sum(1 for d in defect_info_list if d['detected'])
    total_area_pixels = sum(d['area_pixels'] for d in defect_info_list if d['detected'])
    
    # Calculate overall average confidence for detected defects
    detected_confidences = [d['avg_confidence'] for d in defect_info_list if d['detected']]
    avg_overall_confidence = np.mean(detected_confidences) if detected_confidences else 0.0

    return {
        "total_defects_count": total_defects,
        "total_defected_area_pixels": total_area_pixels,
        "average_confidence_detected": avg_overall_confidence
    }

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Loads a PIL Image from raw image bytes.

    Args:
        image_bytes (bytes): The raw bytes of an image file.

    Returns:
        PIL.Image.Image: The loaded PIL Image object.

    Raises:
        IOError: If the image bytes cannot be opened as an image (e.g., corrupted, wrong format).
    """
    if not image_bytes:
        logger.error("Attempted to load an empty image byte stream.")
        raise IOError("Image data is empty. Please provide a valid image file.")
    try:
        return Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as e:
        logger.error(f"Pillow could not identify image format: {e}")
        raise IOError(f"Could not identify image format. Please ensure it's a valid JPG, JPEG, or PNG file. Details: {e}")
    except Exception as e:
        logger.error(f"Failed to load image from bytes due to generic error: {e}")
        raise IOError(f"Could not open image from bytes. Please check file format and integrity. Details: {e}")

