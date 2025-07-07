"""
Streamlit page for managing and inspecting the deep learning model.
Allows users to upload new model files, select encoder backbones,
and view the model's architectural summary.
"""

import streamlit as st
import torch
from torchinfo import summary # For generating model summary
import io
import os
import hashlib # For hashing file content
from typing import List, Dict, Any, Tuple

# Import modules from our structured project
from config import CONFIG, logger
from model import load_segmentation_model # Only load, not run inference here

# ------------------- CACHED UTILITY WRAPPERS -------------------
@st.cache_data
def cached_get_dummy_input(img_height: int, img_width: int, device: torch.device):
    """
    Generates a dummy input tensor for model summary.
    This is cached to avoid regenerating it on every rerun.
    """
    # Create a random tensor with batch size 1, 3 channels (RGB), and configured dimensions
    return torch.randn(1, 3, img_height, img_width).to(device)

# ------------------- MANUAL MODEL SUMMARY GENERATION (FALLBACK) -------------------
def get_manual_model_summary(model: torch.nn.Module) -> str:
    """
    Generates a detailed layer-by-layer summary of a PyTorch model by
    iterating through its named modules and parameters. This serves as a
    robust fallback when torchinfo fails due to tracing issues.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.

    Returns:
        str: A multi-line string containing the manual summary.
    """
    summary_lines = []
    summary_lines.append(f"Model Type: {model.__class__.__name__}")
    summary_lines.append(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    summary_lines.append(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    summary_lines.append("\n--- Layer-by-Layer Breakdown ---")

    for name, module in model.named_modules():
        # Skip the top-level module itself to avoid redundancy
        if name == '':
            continue
        
        summary_lines.append(f"\nModule: {name}")
        summary_lines.append(f"  Type: {module.__class__.__name__}")
        
        # List parameters directly owned by this module
        module_params = []
        for param_name, param in module.named_parameters(recurse=False): # recurse=False to only get direct params
            module_params.append(f"    - {param_name}: Shape={list(param.shape)}, Dtype={param.dtype}, RequiresGrad={param.requires_grad}")
        
        if module_params:
            summary_lines.append("  Parameters:")
            summary_lines.extend(module_params)
        else:
            summary_lines.append("  No direct parameters.")
            
    return "\n".join(summary_lines)


# ------------------- MODEL MANAGEMENT LOGIC -------------------
st.title("Model Management")
st.markdown("""
    This page allows you to configure the deep learning model used for defect detection.
    You can upload a new model file, select its encoder backbone, and view its architecture summary.
""")

st.markdown("---")
st.subheader("Upload Custom Model")

# Use a consistent key for the uploader to prevent multiple reruns on file selection
uploaded_model_file = st.file_uploader(
    "Upload a new PyTorch model file (.pth)",
    type=["pth"],
    key="model_uploader",
    help="Upload a trained PyTorch model file (e.g., a .pth file containing state_dict or the full model)."
)

# Encoder selection dropdown
encoder_options = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "efficientnet-b0", "efficientnet-b1"]
selected_encoder = st.selectbox(
    "Select Encoder Backbone (must match your model's training)",
    options=encoder_options,
    index=encoder_options.index(st.session_state.selected_encoder_name) if st.session_state.selected_encoder_name in encoder_options else 0,
    key="encoder_selector"
)

# Button to apply changes
if st.button("Apply Model Changes", key="apply_model_changes_button"):
    if uploaded_model_file is not None:
        # Save the uploaded file temporarily
        upload_dir = "models"
        os.makedirs(upload_dir, exist_ok=True) # Ensure directory exists
        
        # Create a unique filename for the uploaded model to avoid conflicts
        file_hash = hashlib.md5(uploaded_model_file.getvalue()).hexdigest()
        new_model_path = os.path.join(upload_dir, f"uploaded_model_{file_hash}.pth")
        
        with open(new_model_path, "wb") as f:
            f.write(uploaded_model_file.getvalue())
        
        # Update session state with the new model path and encoder
        st.session_state.current_model_path = new_model_path
        st.session_state.selected_encoder_name = selected_encoder
        
        # Clear cached model to force reload with new path/encoder
        load_segmentation_model.clear()
        
        st.success(f"Model '{uploaded_model_file.name}' loaded and encoder set to '{selected_encoder}'. App will reload with new model.")
        logger.info(f"New model '{uploaded_model_file.name}' uploaded to {new_model_path} and set as active.")
        st.rerun() # Rerun the app to apply changes and reload model
    else:
        # If no new file uploaded, but encoder changed, just update encoder
        if selected_encoder != st.session_state.selected_encoder_name:
            st.session_state.selected_encoder_name = selected_encoder
            load_segmentation_model.clear() # Clear cached model to force reload with new encoder
            st.success(f"Encoder updated to '{selected_encoder}'. App will reload with new encoder.")
            logger.info(f"Encoder updated to {selected_encoder}.")
            st.rerun()
        else:
            st.info("No new model uploaded and no encoder changes detected.")

st.markdown("---")
st.subheader("Current Active Model Details")
st.write(f"**Path:** `{st.session_state.current_model_path}`")
st.write(f"**Encoder:** `{st.session_state.selected_encoder_name}`")
st.write(f"**Device:** `{st.session_state.device.type.upper()}`")


st.markdown("---")
# Wrap the summary section in an expander
with st.expander("✨ View Model Architecture Summary"):
    st.write("This section provides a detailed overview of the model's layers and parameter counts.")
    try:
        # Load the model (it will be cached, so no redundant loading)
        model = load_segmentation_model(
            st.session_state.current_model_path,
            st.session_state.device,
            st.session_state.selected_encoder_name
        )
        
        # Get a dummy input tensor for the model summary
        dummy_input = cached_get_dummy_input(CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH, st.session_state.device)

        # Attempt to generate model summary using torchinfo
        model_summary_str_io = io.StringIO()
        summary(model, input_data=dummy_input, verbose=0, depth=3,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                mode="train",
                col_width=16, wrap_line=True,
                file=model_summary_str_io
                )
        
        st.text(model_summary_str_io.getvalue())
        logger.info("torchinfo model summary generated successfully.")

    except Exception as e:
        logger.error(f"torchinfo summary failed: {e}. Displaying robust layer breakdown as fallback.")
        # Fallback: display the manual model summary directly without a warning message
        manual_summary = get_manual_model_summary(model)
        st.text(manual_summary)
        st.info("Note: A detailed layer-by-layer summary with input/output shapes and FLOPs is not available for this model due to compatibility issues with the `torchinfo` library. The above is a robust breakdown of modules and their parameters.")


st.markdown("---")
st.markdown("### Model Metadata")
st.write("This section provides general information about the model's characteristics as defined in `config.py`.")
st.json(CONFIG.MODEL_INFO)


st.markdown("---")
st.markdown("""
    <div class='footer'>
        Developed using Streamlit & PyTorch | Inspired by Intel AI for Manufacturing initiatives.
        <br>
        © 2025 AI Defect Detection. All rights reserved.
    </div>
""", unsafe_allow_html=True)
