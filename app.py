"""
Main Streamlit application file for the AI-powered Surface Defect Detection App.
This file serves as the entry point and displays the welcome page.
"""

import streamlit as st
from config import initialize_session_state, setup_file_logger

# --- Page and App Initialization ---
# This should be the very first Streamlit command in your app
st.set_page_config(
    page_title="AI Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state and logging ONCE at the start of the app
initialize_session_state()
setup_file_logger()

# --- Global CSS for a professional and modern look ---
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
    :root {
        --primary-color: #007BFF;
        --secondary-color: #0056b3;
        --background-color: #F0F2F6;
        --sidebar-bg: #FFFFFF;
        --text-color: #31333F;
        --header-color: #0c2f5a;
        --border-color: #E6E6E6;
        --shadow-color: rgba(0, 0, 0, 0.05);
    }
    body, .stApp { font-family: 'Inter', sans-serif; background-color: var(--background-color); color: var(--text-color); }
    .main { padding: 2rem; }
    h1, h2, h3 { color: var(--header-color); }
    .stButton > button {
        background-image: linear-gradient(to right, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white; border-radius: 8px; padding: 10px 20px; font-weight: 600; border: none;
        box-shadow: 0 4px 10px var(--shadow-color); transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-image: linear-gradient(to right, var(--secondary-color) 0%, var(--primary-color) 100%);
        transform: translateY(-2px); box-shadow: 0 6px 15px var(--shadow-color);
    }
    [data-testid="stSidebar"] { background-color: var(--sidebar-bg); border-right: 1px solid var(--border-color); }
    .stPageLink { border-radius: 8px; padding: 10px; transition: background-color 0.2s ease; }
    .stPageLink:hover { background-color: #E0E0E0; }
    </style>
""", unsafe_allow_html=True)

# --- Welcome Content for Main Page ---
st.title("👋 Welcome to the AI Surface Defect Detection System")
st.markdown("""
    This application leverages advanced deep learning to accurately identify and classify
    surface defects on materials. You can analyze single images, process entire batches,
    and generate detailed PDF reports.
""")

st.info("Navigate using the pages in the sidebar to begin your analysis.", icon="👈")

st.markdown("---")

cols = st.columns(3)
with cols[0]:
    with st.container(border=True):
        st.subheader("📦 Batch & Single Analysis")
        st.write("Upload images, a ZIP file, or use your camera. View aggregate statistics and detailed results.")
        st.page_link("pages/1_Batch_Single_Detection.py", label="Go to Analysis", icon="➡️")

with cols[1]:
    with st.container(border=True):
        st.subheader("📄 PDF Reporting")
        st.write("Generate a professional PDF report from your batch analysis, complete with charts and annotated images.")
        st.page_link("pages/2_PDF_Report_Generation.py", label="Go to Reporting", icon="➡️")

with cols[2]:
    with st.container(border=True):
        st.subheader("⚙️ Model Management")
        st.write("Configure the active model, upload custom models, and inspect architecture details.")
        st.page_link("pages/3_Model_Management.py", label="Go to Management", icon="➡️")

# --- Footer ---
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; font-size: 0.9em; color: #666;'>
        Developed using Streamlit & PyTorch | © 2025 AI Defect Detection
    </div>
""", unsafe_allow_html=True)
