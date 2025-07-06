"""
Streamlit page for Batch Image Analysis and Single Image Camera Detection.
Allows users to upload multiple images/ZIP files or capture a single image via webcam
for defect detection.
"""

import streamlit as st
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple
import hashlib
import io
import uuid
import pandas as pd
import concurrent.futures
import os
import zipfile
import plotly.express as px
import torch
import logging
import datetime # Import datetime for camera image naming

# Import modules from our structured project
from config import CONFIG, logger
from model import load_segmentation_model, run_inference
from transforms import get_validation_transforms
from utils import (
    create_class_heatmap,
    create_confidence_heatmap,
    overlay_predictions_on_image,
    load_image_from_bytes,
    get_defect_summary_strings
)

# ------------------- SESSION STATE INITIALIZATION -------------------
# Initialize session state variables if they don't exist
st.session_state.setdefault('current_model_path', CONFIG.MODEL_PATH)
st.session_state.setdefault('selected_encoder_name', CONFIG.DEFAULT_ENCODER)
st.session_state.setdefault('class_thresholds', {cls: CONFIG.DEFAULT_THRESHOLD for cls in CONFIG.CLASSES})
st.session_state.setdefault('uploaded_batch', []) # List of dicts: {'id', 'name', 'bytes', 'predictions', 'inference_time'}
st.session_state.setdefault('single_image_report_data', None) # Data for single image CSV report
st.session_state.setdefault('model_loaded', False) # Track if model is loaded
st.session_state.setdefault('min_heatmap_prob', CONFIG.DEFAULT_HEATMAP_MIN)
st.session_state.setdefault('max_heatmap_prob', CONFIG.DEFAULT_HEATMAP_MAX)
st.session_state.setdefault('visible_classes', CONFIG.CLASSES) # Default to all visible classes
st.session_state.setdefault('file_uploader_batch_key', 0) # Key for the batch file uploader widget
st.session_state.setdefault('camera_input_single_key', 0) # Key for the single camera input widget
st.session_state.setdefault('camera_image_bytes', None)
st.session_state.setdefault('single_image_predictions', None) # Raw predictions for single camera image
st.session_state.setdefault('single_image_inference_time', None)
st.session_state.setdefault('last_uploaded_files_hash_batch', None) # To prevent re-processing same batch
st.session_state.setdefault('batch_analysis_df', pd.DataFrame()) # For PDF report generation
st.session_state.setdefault('heatmap_cmap_name', 'Greens') # Changed: Default colormap for heatmaps to 'Greens'


# ------------------- MODEL LOADING (CACHED) -------------------
@st.cache_resource
def get_model(model_path, device, encoder_name):
    """Caches the model loading to avoid reloading on every rerun."""
    try:
        model = load_segmentation_model(model_path, device, encoder_name)
        st.session_state.model_loaded = True
        logger.info(f"Model {encoder_name} from {model_path} loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please check model path and encoder name on 'Model Management' page.")
        st.session_state.model_loaded = False
        logger.error(f"Failed to load model: {e}")
        return None

# Load the model and transforms
model = get_model(st.session_state.current_model_path, CONFIG.DEVICE, st.session_state.selected_encoder_name)
validation_transform = get_validation_transforms(CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH)


# ------------------- CLEAR FUNCTIONS (Defined at top level) -------------------
def clear_batch_data():
    """Clears all session state data related to uploaded batch images and reports."""
    st.session_state.uploaded_batch = []
    st.session_state.batch_analysis_df = pd.DataFrame() # Clear batch analysis DataFrame
    st.session_state.last_uploaded_files_hash_batch = None
    st.session_state.file_uploader_batch_key += 1 # Increment key to reset file uploader
    st.cache_data.clear() # Clear cached data for overlays, etc.
    logger.info("Batch data cleared.")
    # No st.rerun() here, as it's called by the button's on_click which handles the rerun.
    # The current script run will still complete, but the state is now empty.


def clear_single_image_data():
    """Clears session state data related to the single camera image."""
    st.session_state.camera_image_bytes = None
    st.session_state.single_image_predictions = None
    st.session_state.single_image_inference_time = None
    st.session_state.single_image_report_data = None
    st.session_state.camera_input_single_key += 1 # Increment key to reset camera input
    st.cache_data.clear() # Clear cached data for overlays, etc.
    logger.info("Single camera image data cleared.")
    st.rerun()


# ------------------- HELPER FUNCTIONS FOR PROCESSING & DISPLAY -------------------

def process_single_image_for_display(image_bytes: bytes, image_name: str, model: Any, transform: Any, device: torch.device, class_thresholds: Dict[str, float], colors: Dict[str, Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]], float, np.ndarray]:
    """
    Processes a single image through the model and generates all visualizations.
    Returns original_img_np (resized), overlay_img_np, confidence_heatmap_np, class_heatmaps, defect_info_list, inference_time, raw_predictions.
    """
    if model is None:
        st.error("Model is not loaded. Please go to 'Model Management' page to load the model.")
        return None, None, None, {}, [], 0.0, None

    try:
        img_pil = load_image_from_bytes(image_bytes)
        
        # IMPORTANT: Resize original image to match model input/output dimensions for consistent blending
        img_pil_resized = img_pil.resize((CONFIG.IMG_WIDTH, CONFIG.IMG_HEIGHT), Image.LANCZOS)
        original_img_np_resized = np.array(img_pil_resized.convert("RGB"))

        # Run inference using the cached function from model.py
        predictions, inference_time = run_inference(image_bytes, model, transform, device)

        # Generate visualizations using cached utility functions
        # Note: overlay_predictions_on_image handles its own resizing internally from image_bytes
        overlay_img_np, defect_info_list = overlay_predictions_on_image(
            image_bytes, predictions, colors, CONFIG.IMG_WIDTH, CONFIG.IMG_HEIGHT,
            st.session_state.visible_classes, class_thresholds
        )
        
        # Pass the consistently resized original_img_np_resized to heatmap functions
        confidence_heatmap_np = create_confidence_heatmap(
            predictions, original_img_np_resized, 0.5, st.session_state.heatmap_cmap_name # Use dynamic cmap_name
        )

        class_heatmaps = {}
        for i, class_name in enumerate(CONFIG.CLASSES):
            if class_name in st.session_state.visible_classes:
                class_heatmaps[class_name] = create_class_heatmap(
                    predictions[i], original_img_np_resized, 0.5, st.session_state.heatmap_cmap_name,
                    st.session_state.min_heatmap_prob, st.session_state.max_heatmap_prob
                )

        return original_img_np_resized, overlay_img_np, confidence_heatmap_np, class_heatmaps, defect_info_list, inference_time, predictions

    except Exception as e:
        st.error(f"Error processing image '{image_name}': {e}")
        logger.exception(f"Error processing image '{image_name}':")
        return None, None, None, {}, [], 0.0, None

def display_image_results(original_img_np, overlay_img_np, confidence_heatmap_np, class_heatmaps, defect_info_list, image_name, inference_time, raw_predictions_for_masks: np.ndarray):
    """Displays the original, overlaid, and heatmap images in columns with defect summary."""
    
    st.subheader(f"Results for: {image_name}")

    cols = st.columns(2)
    with cols[0]:
        st.image(original_img_np, caption="Original Image", use_container_width=True)
    with cols[1]:
        if overlay_img_np is not None:
            st.image(overlay_img_np, caption=f"Detected Defects (Overlay) - Inference: {inference_time:.4f}s", use_container_width=True)
            
            # Download button for overlaid image
            img_to_save = Image.fromarray(overlay_img_np.astype(np.uint8))
            buf = io.BytesIO()
            img_to_save.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label=f"Download Overlaid Image for {image_name}",
                data=byte_im,
                file_name=f"detected_defects_overlay_{image_name.replace('/', '_')}.png",
                mime="image/png",
                key=f"download_overlay_button_{image_name}"
            )
        else:
            st.warning("Defect Overlay not available.")

    st.markdown("---")

    st.subheader("Advanced Visualizations")
    tab_conf_heatmap, tab_class_heatmaps, tab_raw_masks = st.tabs(["Overall Confidence Heatmap", "Class-wise Heatmaps", "Raw Binary Masks"])

    with tab_conf_heatmap:
        if confidence_heatmap_np is not None:
            st.image(confidence_heatmap_np, caption="Overall Confidence Heatmap (Max Probability Across Classes)", use_container_width=True)
        else:
            st.warning("Confidence Heatmap not available.")
    
    with tab_class_heatmaps:
        st.write("### Class-wise Probability Heatmaps")
        num_visible_classes = len(st.session_state.visible_classes)
        if class_heatmaps and num_visible_classes > 0:
            cols_per_row = min(num_visible_classes, 4)
            heatmap_cols = st.columns(cols_per_row)
            for k, class_name in enumerate(CONFIG.CLASSES):
                if class_name in st.session_state.visible_classes and class_name in class_heatmaps:
                    # Using k for column index, assuming CONFIG.CLASSES order is consistent
                    with heatmap_cols[k % cols_per_row]: # Use modulo to cycle through columns
                        st.markdown(f"**{class_name} Heatmap**")
                        st.image(class_heatmaps[class_name], use_container_width=True, caption=f"Probabilities for {class_name}")
        else:
            st.info("No class-wise heatmaps to display (check selected defects in sidebar).")

    with tab_raw_masks:
        st.write("### Raw Binary Masks (after thresholding)")
        if raw_predictions_for_masks is not None and raw_predictions_for_masks.shape[0] == len(CONFIG.CLASSES):
            num_classes = len(CONFIG.CLASSES)
            cols_per_row = min(num_classes, 4) 
            mask_cols = st.columns(cols_per_row)
            
            for k, class_name in enumerate(CONFIG.CLASSES):
                with mask_cols[k % cols_per_row]: # Use modulo to cycle through columns
                    current_threshold_for_mask = st.session_state.class_thresholds.get(class_name, CONFIG.DEFAULT_THRESHOLD)
                    # Use the raw predictions passed as argument
                    binary_mask_for_display = (raw_predictions_for_masks[k] > current_threshold_for_mask).astype(np.uint8) * 255
                    mask_image_pil = Image.fromarray(binary_mask_for_display)
                    
                    st.image(mask_image_pil, caption=f"{class_name} Mask (Threshold: {current_threshold_for_mask:.2f})", use_container_width=True, channels="GRAY")
                    area = np.sum(binary_mask_for_display > 0) / 255 # Divide by 255 because mask is 0/255
                    st.markdown(f"<small>Area: {int(area)} pixels</small>", unsafe_allow_html=True)
        else:
            st.info("Raw predictions not available or malformed to display masks.")


    st.markdown("---")

    st.subheader("Defect Summary")
    if defect_info_list:
        summary_strings = get_defect_summary_strings(defect_info_list, st.session_state.visible_classes)
        
        found_any_defect_in_visible_classes = any(info['detected'] and info['class_name'] in st.session_state.visible_classes for info in defect_info_list)

        if found_any_defect_in_visible_classes:
            st.success("Defects Detected (showing selected types)! 👇")
            for info_str in summary_strings:
                st.markdown(f"- {info_str}", unsafe_allow_html=True) # Allow HTML for bold tags
        else:
            st.info("✅ No significant defects of selected types detected at the current thresholds.")
            
        # Display the full report data for this single image
        if st.session_state.single_image_report_data:
            st.markdown("---")
            st.subheader("📊 Single Image Analysis Report (All Defects)")
            single_report_df = pd.DataFrame(st.session_state.single_image_report_data)
            single_report_df = single_report_df[['DefectType', 'Detected', 'AreaPixels', 'AreaPercentage', 'MaxConfidence', 'ThresholdUsed']]
            st.dataframe(single_report_df, use_container_width=True)
            
            csv_string = single_report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Report (CSV)",
                data=csv_string,
                file_name=f"defect_report_single_image_{image_name.replace('/', '_')}.csv",
                mime="text/csv",
                key=f"download_report_single_button_{image_name}"
            )

    else:
        st.info("No defect information available for this image.")
    st.markdown("---")


# ------------------- STREAMLIT APP LAYOUT -------------------
st.title("Batch & Single Image Defect Detection")

st.markdown("""
    Upload images (JPG, PNG) or a ZIP file containing images for batch processing.
    Alternatively, use your device's camera for single image detection.
""")

# --- Input Method Selection ---
source = st.radio(
    "Select Input Method:",
    ["Upload Images/ZIP (Batch)", "Use Camera (Single Image)"],
    index=0,
    horizontal=True,
    key="input_method_selector"
)

# --- Sidebar for Analysis Settings ---
st.sidebar.header("⚙️ Analysis Settings")

st.sidebar.markdown("### 🎯 Defect Thresholds")
for cls in CONFIG.CLASSES:
    st.session_state.class_thresholds[cls] = st.sidebar.slider(
        f"Threshold for {cls}",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.class_thresholds.get(cls, CONFIG.DEFAULT_THRESHOLD),
        step=0.05,
        key=f"threshold_slider_{cls}",
        help=f"Adjust the confidence threshold for '{cls}'."
    )

st.sidebar.markdown("---")
st.sidebar.header("👁️ Overlay & Heatmap Filter")
selected_classes_raw = st.sidebar.multiselect(
    "Select defects to display:",
    options=CONFIG.CLASSES,
    default=st.session_state.visible_classes,
    key="defect_display_multiselect",
    help="Choose which defect types to visualize in overlays and heatmaps."
)

if not selected_classes_raw:
    st.session_state.visible_classes = CONFIG.CLASSES
    st.sidebar.info("No specific defect types selected. Displaying ALL defects.")
else:
    st.session_state.visible_classes = selected_classes_raw

st.sidebar.markdown("---")
st.sidebar.header("🎨 Visualization Options")
st.sidebar.markdown("### 🌈 Heatmap Color Range")
col_min, col_max = st.sidebar.columns(2)
with col_min:
    st.session_state.min_heatmap_prob = st.sidebar.slider(
        "Min Probability",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.min_heatmap_prob,
        step=0.01,
        key="heatmap_min_range",
        help="Adjust the minimum probability value mapped to the heatmap color scale."
    )
with col_max:
    st.session_state.max_heatmap_prob = st.sidebar.slider(
        "Max Probability",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.max_heatmap_prob,
        step=0.01,
        key="heatmap_max_range",
        help="Adjust the maximum probability value mapped to the heatmap color scale."
    )
if st.session_state.min_heatmap_prob >= st.session_state.max_heatmap_prob:
    st.sidebar.error("Min Probability must be less than Max Probability. Adjusting...")
    st.session_state.max_heatmap_prob = st.session_state.min_heatmap_prob + 0.01
    if st.session_state.max_heatmap_prob > 1.0: # Prevent exceeding 1.0
        st.session_state.min_heatmap_prob = 1.0 - 0.01
        st.session_state.max_heatmap_prob = 1.0

# Selectbox for heatmap colormap
st.session_state.heatmap_cmap_name = st.sidebar.selectbox(
    "Select Heatmap Colormap:",
    options=['Greens', 'Blues', 'Reds', 'Oranges', 'viridis', 'plasma', 'magma', 'cividis', 'inferno', 'jet', 'hot', 'gray'],
    index=['Greens', 'Blues', 'Reds', 'Oranges', 'viridis', 'plasma', 'magma', 'cividis', 'inferno', 'jet', 'hot', 'gray'].index(st.session_state.heatmap_cmap_name),
    key="heatmap_cmap_selector",
    help="Choose the color scheme for probability heatmaps. 'Greens', 'Blues', 'Reds' are good for 2-3 color effects."
)


st.sidebar.markdown("---")
st.sidebar.markdown("### Defect Color Legend")
for class_name, color_rgb in CONFIG.COLORS.items():
    st.sidebar.markdown(
        f"<span style='color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]});'>■</span> {class_name}",
        unsafe_allow_html=True
    )
st.sidebar.markdown("---")
st.sidebar.info(f"Model running on: **{CONFIG.DEVICE.type.upper()}** with **{st.session_state.selected_encoder_name}** encoder.")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit & PyTorch")


# --- Handle Batch Uploads ---
if source == "Upload Images/ZIP (Batch)":
    # Clear camera related session state if switching from camera mode
    if st.session_state.camera_image_bytes is not None:
        clear_single_image_data() # This will rerun the app.
        
    # Initialize df_report here, so it's always defined in this scope
    # This is crucial for preventing NameError on initial load or after clearing data
    df_report = pd.DataFrame() 

    # Clear All Button for batch processing
    if st.session_state.uploaded_batch:
        # Call clear_batch_data directly. The rerun will happen after this script finishes.
        st.button("🗑️ Clear All Uploaded Images and Results", on_click=clear_batch_data, help="Remove all currently uploaded images and clear the batch report.")

    # Use the dynamic key for the file uploader
    uploaded_files = st.file_uploader(
        "Upload metal surface images or a ZIP file containing images...",
        type=["jpg", "jpeg", "png", "zip"],
        accept_multiple_files=True,
        key=f"file_uploader_batch_{st.session_state.file_uploader_batch_key}"
    )

    if uploaded_files:
        # Generate a unique identifier for the current set of uploaded files' contents
        # This hash ensures that if the user selects the exact same files again, we don't re-process
        current_uploaded_file_contents_hash = hashlib.md5(
            "".join([f.name + str(f.size) + str(f.file_id) for f in uploaded_files]).encode('utf-8')
        ).hexdigest()

        # Only process if the set of uploaded files has changed from the last time
        if current_uploaded_file_contents_hash != st.session_state.last_uploaded_files_hash_batch:
            st.session_state.last_uploaded_files_hash_batch = current_uploaded_file_contents_hash
            st.session_state.uploaded_batch = [] # Clear existing batch to avoid duplicates and ensure fresh processing

            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/zip":
                    try:
                        with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as zf:
                            image_files_in_zip_count = 0
                            for member in zf.namelist():
                                # Skip macOS resource forks and directories
                                if not member.startswith('__MACOSX/') and not member.endswith('/'):
                                    file_ext = os.path.splitext(member)[1].lower()
                                    if file_ext in CONFIG.IMAGE_EXTENSIONS:
                                        with zf.open(member) as file:
                                            file_bytes = file.read()
                                            st.session_state.uploaded_batch.append({
                                                'id': hashlib.md5(file_bytes).hexdigest(), # Unique ID for each image in zip
                                                'name': f"{uploaded_file.name}/{member}",
                                                'bytes': file_bytes,
                                                'predictions': None,
                                                'inference_time': None
                                            })
                                            image_files_in_zip_count += 1
                                    else:
                                        logger.info(f"Skipping non-image file in ZIP: {member}")
                            if image_files_in_zip_count == 0:
                                st.warning(f"No valid image files found in '{uploaded_file.name}'. Supported formats: {', '.join(CONFIG.IMAGE_EXTENSIONS)}")
                            else:
                                st.info(f"Extracted {image_files_in_zip_count} image(s) from '{uploaded_file.name}'.")
                    except zipfile.BadZipFile:
                        st.error(f"Failed to read '{uploaded_file.name}'. It might be a corrupted or invalid ZIP file.")
                        logger.error(f"Bad ZIP file: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"An error occurred while processing '{uploaded_file.name}': {e}")
                        logger.exception(f"Error processing ZIP file {uploaded_file.name}:")
                else:
                    file_bytes = uploaded_file.getvalue()
                    st.session_state.uploaded_batch.append({
                        'id': hashlib.md5(file_bytes).hexdigest(),
                        'name': uploaded_file.name,
                        'bytes': file_bytes,
                        'predictions': None,
                        'inference_time': None
                    })
            logger.info(f"New batch of {len(st.session_state.uploaded_batch)} images detected. Forcing rerun to process...")
            st.rerun() # Force rerun to process new images
        else:
            if st.session_state.uploaded_batch:
                st.info("files uploaded. Displaying results.")
    else: # If the uploader widget is empty (e.g., after clear_batch_data)
        st.session_state.last_uploaded_files_hash_batch = None # Reset hash so next upload is new

    # --- Batch Management and Processing ---
    if st.session_state.uploaded_batch:
        st.markdown("---")
        st.markdown(f"## 📊 Batch Status: {len(st.session_state.uploaded_batch)} Images Loaded")

        images_to_process_count = sum(1 for img in st.session_state.uploaded_batch if img['predictions'] is None)
        
        if images_to_process_count > 0:
            if model is None:
                st.warning("Model is not loaded. Please load a model on the 'Model Management' page to proceed with detection.")
            else:
                st.info(f"Processing {images_to_process_count} image(s) in batch... This may take a moment.")
                progress_bar = st.progress(0, text="Processing images...")
                
                total_images = len(st.session_state.uploaded_batch)
                processed_count = 0
                
                # Use ThreadPoolExecutor for parallel processing of images
                with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                    futures = {
                        executor.submit(run_inference, img_data['bytes'], model, validation_transform, CONFIG.DEVICE): i
                        for i, img_data in enumerate(st.session_state.uploaded_batch) if img_data['predictions'] is None
                    }
                    
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        try:
                            predictions, inference_time_single = future.result()
                            st.session_state.uploaded_batch[idx]['predictions'] = predictions
                            st.session_state.uploaded_batch[idx]['inference_time'] = inference_time_single
                            logger.info(f"Predictions generated for {st.session_state.uploaded_batch[idx]['name']} in {inference_time_single:.4f}s")
                        except Exception as e:
                            st.error(f"Error processing {st.session_state.uploaded_batch[idx]['name']}: {e}")
                            logger.exception(f"Error during prediction for {st.session_state.uploaded_batch[idx]['name']}:")
                            st.session_state.uploaded_batch[idx]['predictions'] = None # Mark as failed
                            st.session_state.uploaded_batch[idx]['inference_time'] = 0.0 # Mark time as 0
                        
                        processed_count += 1
                        progress_bar.progress(processed_count / images_to_process_count)
                
                progress_bar.empty()
                st.success("Batch processing complete!")
                st.rerun() # Rerun to display all processed results

        else: # All images are already processed
            st.success("All images in batch processed!")

        # --- Download Full Prediction Report (CSV) ---
        all_defect_data_for_csv_report = []
        total_batch_inference_time = 0.0
        
        for img_data in st.session_state.uploaded_batch:
            if img_data['predictions'] is not None:
                total_batch_inference_time += img_data['inference_time']
                
                # Re-run overlay_predictions_on_image to get detailed_info for CSV report
                _, detailed_info = overlay_predictions_on_image(
                    img_data['bytes'],
                    img_data['predictions'],
                    CONFIG.COLORS,
                    CONFIG.IMG_WIDTH,
                    CONFIG.IMG_HEIGHT,
                    CONFIG.CLASSES, # Always pass all classes for the full report
                    st.session_state.class_thresholds # Use per-class thresholds for detection logic in report
                )
                for defect in detailed_info:
                    bbox_str = f"({defect['bounding_box'][0]}, {defect['bounding_box'][1]}, {defect['bounding_box'][2]}, {defect['bounding_box'][3]})" if defect['bounding_box'] else 'N/A'
                    all_defect_data_for_csv_report.append({
                        'Image Name': img_data['name'],
                        'Defect Type': defect['class_name'],
                        'Detected': 'Yes' if defect['detected'] else 'No',
                        'Area (Pixels)': defect['area_pixels'],
                        'Area %': f"{defect['area_percentage']:.2f}%", # Use calculated percentage
                        'Avg Confidence': f"{defect['avg_confidence']:.4f}",
                        'Bounding Box (x,y,w,h)': bbox_str,
                        'Threshold Used': f"{defect['threshold_used']:.2f}"
                    })
            else:
                # Add entries for failed images
                for class_name in CONFIG.CLASSES:
                    all_defect_data_for_csv_report.append({
                        'Image Name': img_data['name'],
                        'Defect Type': class_name,
                        'Detected': 'No (Processing Failed)', # Indicate failure
                        'Area (Pixels)': 0,
                        'Area %': '0.00%',
                        'Avg Confidence': '0.0000',
                        'Bounding Box (x,y,w,h)': 'N/A',
                        'Threshold Used': st.session_state.class_thresholds.get(class_name, CONFIG.DEFAULT_THRESHOLD)
                    })

        # Initialize df_report from the session state DataFrame, which is guaranteed to be a DataFrame
        df_report = st.session_state.batch_analysis_df = pd.DataFrame(all_defect_data_for_csv_report)

        if not df_report.empty: # Check if DataFrame is not empty before displaying/downloading
            st.markdown("<hr style='border: 2px solid #3498db; margin: 40px 0;'>", unsafe_allow_html=True) # Visual Separator
            st.subheader("📊 Full Prediction Report (CSV)")
            st.dataframe(df_report, use_container_width=True)
            
            csv_buffer = io.StringIO()
            df_report.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Full Prediction Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name="defect_batch_report.csv",
                mime="text/csv",
                key="download_report_button_batch"
            )
        else:
            st.info("Upload and process images to generate a prediction report.")

        # --- Display Individual Image Results (Batch Mode) ---
        st.markdown("<hr style='border: 2px solid #3498db; margin: 40px 0;'>", unsafe_allow_html=True) # Visual Separator
        st.markdown("## 🖼️ Review Individual Images in Batch")

        if not st.session_state.uploaded_batch:
            st.info("No images uploaded yet for individual review.")
        else:
            for i, current_image_data in enumerate(st.session_state.uploaded_batch):
                st.markdown(f"### Image {i+1}: {current_image_data['name']}")
                
                if current_image_data['predictions'] is not None:
                    # Re-generate all visuals for display for each image
                    original_img_np, overlay_img_np, confidence_heatmap_np, class_heatmaps, defect_info_list_display, inference_time_display, raw_predictions_display = \
                        process_single_image_for_display(
                            current_image_data['bytes'],
                            current_image_data['name'],
                            model,
                            validation_transform,
                            CONFIG.DEVICE,
                            st.session_state.class_thresholds,
                            CONFIG.COLORS
                        )
                    
                    if original_img_np is not None: # Check if processing was successful
                        display_image_results(
                            original_img_np, overlay_img_np, confidence_heatmap_np, class_heatmaps,
                            defect_info_list_display, current_image_data['name'], inference_time_display, raw_predictions_display
                        )
                    else:
                        st.error(f"Failed to display results for {current_image_data['name']}.")
                else:
                    st.warning(f"Image {current_image_data['name']} was not processed successfully.")
                st.markdown("---") # Strong separator between individual images
    else:
        st.info("Upload images or a ZIP file to start batch processing.")

    # --- Per-Class Summary & Charts ---
    st.markdown("<hr style='border: 2px solid #3498db; margin: 40px 0;'>", unsafe_allow_html=True) # Visual Separator
    st.markdown("## 📈 Batch Defect Summary by Class")
    
    # Filter for successfully detected defects for summary and charts
    # Ensure df_report is not empty before filtering
    if not df_report.empty:
        df_summary_filtered = df_report[df_report['Detected'] == 'Yes'].copy()

        if not df_summary_filtered.empty:
            st.markdown(f"**Total Images Analyzed:** {len(st.session_state.uploaded_batch)}")
            st.markdown(f"**Total Batch Inference Time:** `{total_batch_inference_time:.4f} seconds`")

            # Convert Area (Pixels) and Avg Confidence to numeric for aggregation
            df_summary_filtered['Area (Pixels)'] = pd.to_numeric(df_summary_filtered['Area (Pixels)'], errors='coerce')
            df_summary_filtered['Avg Confidence'] = pd.to_numeric(df_summary_filtered['Avg Confidence'], errors='coerce')

            for class_name in CONFIG.CLASSES:
                class_df = df_summary_filtered[df_summary_filtered['Defect Type'] == class_name]
                detected_count = class_df['Image Name'].nunique()
                avg_area = class_df['Area (Pixels)'].mean()
                avg_confidence = class_df['Avg Confidence'].mean()

                st.markdown(
                    f"- **{class_name}:** Detected in `{detected_count}/{len(st.session_state.uploaded_batch)}` images "
                    f"| Avg. Area (when detected): `{avg_area:.2f} px²`"
                    f"| Avg. Confidence (when detected): `{avg_confidence:.4f}`"
                )
            
            st.markdown("#### Interactive Visualizations")
            
            defect_occurrence_counts = df_summary_filtered.groupby('Defect Type')['Image Name'].nunique().reset_index(name='NumImagesDetected')
            if not defect_occurrence_counts.empty:
                fig_occurrence = px.bar(
                    defect_occurrence_counts,
                    x='Defect Type',
                    y='NumImagesDetected',
                    title='Defect Occurrence Across Images',
                    labels={'Defect Type': 'Defect Type', 'NumImagesDetected': 'Number of Images Detected'},
                    color='Defect Type',
                    color_discrete_map={cls: f"rgb{CONFIG.COLORS[cls]}" for cls in CONFIG.CLASSES}
                )
                fig_occurrence.update_layout(xaxis_title="Defect Type", yaxis_title="Number of Images Detected")
                st.plotly_chart(fig_occurrence, use_container_width=True)
            else:
                st.info("No defects detected to plot occurrence.")

            total_area_by_defect = df_summary_filtered.groupby('Defect Type')['Area (Pixels)'].sum().reset_index(name='TotalAreaPixels')
            if not total_area_by_defect.empty and total_area_by_defect['TotalAreaPixels'].sum() > 0:
                fig_pie = px.pie(
                    total_area_by_defect,
                    values='TotalAreaPixels',
                    names='Defect Type',
                    title='Proportion of Total Detected Area by Defect Type',
                    color='Defect Type',
                    color_discrete_map={cls: f"rgb{CONFIG.COLORS[cls]}" for cls in CONFIG.CLASSES}
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No defects detected with significant area to plot proportions.")

        else:
            st.info("No defects were detected across the uploaded batch to generate charts.")
    else:
        st.info("No batch analysis data available to generate summary charts. Please upload and process images.")


# --- Handle Single Camera Image ---
elif source == "Use Camera (Single Image)":
    # Clear batch related session state if switching from batch mode
    if st.session_state.uploaded_batch:
        clear_batch_data() # This will rerun the app.

    st.markdown("#### Snap a picture of the surface for instant defect detection!")
    # Use the dynamic key for the camera input
    camera_image_file = st.camera_input("Click 'Take Photo' to capture a frame from your webcam:", help="Your browser will ask for camera permission. The live preview is shown above this button.", key=f"camera_input_single_{st.session_state.camera_input_single_key}")

    should_process_camera_image = False # Flag to control processing

    if camera_image_file is not None:
        new_image_bytes = camera_image_file.getvalue()
        
        # Check if the image bytes have actually changed from the last processed image
        # OR if predictions are not yet available for the current image bytes (e.g., first capture)
        if st.session_state.camera_image_bytes != new_image_bytes:
            st.session_state.camera_image_bytes = new_image_bytes
            st.session_state.single_image_predictions = None # Clear old predictions to force re-processing
            st.session_state.single_image_inference_time = None
            st.session_state.single_image_report_data = None
            logger.info("New camera image captured. Setting flag to process.")
            should_process_camera_image = True
        elif st.session_state.single_image_predictions is None and st.session_state.camera_image_bytes is not None:
            # This case handles a rerun where the image is the same but predictions somehow got cleared
            # or if the app just started and the image was already there from a previous session.
            logger.info("Camera image present but predictions missing. Setting flag to re-process.")
            should_process_camera_image = True
        else:
            logger.info("Same camera image, predictions already exist. Skipping re-processing.")
            # should_process_camera_image remains False

    # This block now handles the processing based on the flag
    if should_process_camera_image:
        st.subheader("Processing Camera Image")
        
        if model is None:
            st.warning("Model is not loaded. Please load a model on the 'Model Management' page to proceed with detection.")
        else:
            with st.spinner("Analyzing camera image for defects..."):
                try:
                    predictions, inference_time_single = run_inference(st.session_state.camera_image_bytes, model, validation_transform, CONFIG.DEVICE)
                    st.session_state.single_image_predictions = predictions
                    st.session_state.single_image_inference_time = inference_time_single
                    logger.info(f"Single camera image processed in {inference_time_single:.4f}s.")
                    
                    # Prepare data for single image report (all defects, not just selected for display)
                    _, defect_info_list_for_report = overlay_predictions_on_image(
                        st.session_state.camera_image_bytes, predictions, CONFIG.COLORS, CONFIG.IMG_WIDTH, CONFIG.IMG_HEIGHT,
                        CONFIG.CLASSES, st.session_state.class_thresholds
                    )
                    single_image_report_data_current = []
                    for defect in defect_info_list_for_report:
                        area_percentage = (defect['area_pixels'] / (CONFIG.IMG_WIDTH * CONFIG.IMG_HEIGHT) * 100) if (CONFIG.IMG_WIDTH * CONFIG.IMG_HEIGHT) > 0 else 0
                        single_image_report_data_current.append({
                            'DefectType': defect['class_name'],
                            'Detected': 'Yes' if defect['detected'] else 'No',
                            'AreaPixels': defect['area_pixels'],
                            'AreaPercentage': f"{area_percentage:.2f}%",
                            'MaxConfidence': f"{defect['avg_confidence']:.4f}",
                            'ThresholdUsed': f"{defect['threshold_used']:.2f}"
                        })
                    st.session_state.single_image_report_data = single_image_report_data_current

                    # NO st.rerun() here. The display logic will run immediately after this.
                except Exception as e:
                    st.error(f"Error processing camera image: {e}")
                    logger.exception("Error during single camera image processing:")
                    st.session_state.single_image_predictions = None # Mark as failed
                    st.session_state.single_image_inference_time = 0.0
                    st.session_state.single_image_report_data = None # Clear report data on failure

    # This block now displays if predictions are available, regardless of how they got there
    if st.session_state.single_image_predictions is not None:
        # Re-generate all visuals for display for the single camera image
        # This will use the predictions already in session state
        original_img_np, overlay_img_np, confidence_heatmap_np, class_heatmaps, defect_info_list_display, inference_time_display, raw_predictions_display = \
            process_single_image_for_display(
                st.session_state.camera_image_bytes,
                "Camera Image",
                model,
                validation_transform,
                CONFIG.DEVICE,
                st.session_state.class_thresholds,
                CONFIG.COLORS
            )
        
        if original_img_np is not None: # Check if processing was successful
            display_image_results(
                original_img_np, overlay_img_np, confidence_heatmap_np, class_heatmaps,
                defect_info_list_display, "Camera Image", inference_time_display, raw_predictions_display
            )
        else:
            st.error("Failed to display results for Camera Image.")

        st.button("📸 Take another photo", on_click=clear_single_image_data, key="take_another_photo_button")

    elif st.session_state.camera_image_bytes is None: # Only show this if no image is currently captured
        st.info("📸 Click 'Take Photo' above to capture an image from your webcam for defect detection.")

st.markdown("---")
st.markdown("""
    <div class='footer'>
        Developed using Streamlit & PyTorch | Inspired by Intel AI for Manufacturing initiatives.
        <br>
        © 2025 AI Defect Detection. All rights reserved.
    </div>
""", unsafe_allow_html=True)
