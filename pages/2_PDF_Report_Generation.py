"""
Streamlit page for generating PDF reports from batch analysis results.
This module provides a robust and comprehensive PDF report generation feature
using ReportLab, designed from scratch to address previous issues and
incorporate all necessary features and best practices.
"""

import streamlit as st
import pandas as pd
import io
import os
import datetime
import numpy as np
import time
from typing import List, Dict, Any, Tuple
from PIL import Image
import requests # For fetching logo (e.g., placeholder)

# ReportLab imports - carefully selected for compatibility and functionality
# TA_TOP (and TA_MIDDLE) are intentionally excluded from import to avoid ImportError,
# as they appear to be unavailable in the user's specific ReportLab environment.
# Vertical alignment will default to ReportLab's built-in behavior.
# VA_TOP, VA_MIDDLE, VA_BOTTOM are also removed due to ImportError.
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import ParagraphStyle
# --- IMPORTANT: TA_CENTER, TA_LEFT, TA_RIGHT ARE NO LONGER IMPORTED OR USED FOR ALIGNMENT ---
# The previous errors indicate these constants cause issues with alignment in the user's environment.
# from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT 
from reportlab.lib.colors import black, white, grey, HexColor # Explicitly use HexColor for custom colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# Plotly for charts - essential for data visualization in the report
import plotly.express as px
import plotly.io as pio

# Import modules from our structured project - assuming these are stable
from config import CONFIG, logger
from utils import overlay_predictions_on_image, create_confidence_heatmap # Import heatmap function


# Configure plotly to use kaleido for static image export.
# This ensures charts are rendered as static images within the PDF.
pio.defaults.default_format = "png"
pio.defaults.default_width = 800
pio.defaults.default_height = 500


# ------------------- SESSION STATE INITIALIZATION -------------------
# Initialize critical session state variables. This ensures that these variables
# are available and consistent across Streamlit reruns and different pages.
st.session_state.setdefault('current_model_path', CONFIG.MODEL_PATH)
st.session_state.setdefault('selected_encoder_name', CONFIG.DEFAULT_ENCODER)
st.session_state.setdefault('class_thresholds', {cls: CONFIG.DEFAULT_THRESHOLD for cls in CONFIG.CLASSES})
st.session_state.setdefault('uploaded_batch', []) # Stores details of all uploaded images and their predictions
st.session_state.setdefault('batch_analysis_df', pd.DataFrame()) # Stores aggregated analysis results for the batch


# ------------------- GLOBAL PDF STYLES (DEFINED ONCE AND CACHED) -------------------
@st.cache_resource
def get_pdf_styles():
    """
    Initializes and caches ReportLab ParagraphStyle objects.
    These styles define the appearance of text throughout the PDF report.
    Using basic, robust fonts and colors for maximum compatibility.
    Explicit alignment is removed from these styles.
    """
    custom_styles = {}
    
    # Title Page Styles - Enhanced for attractiveness and readability
    custom_styles['TitleStyle'] = ParagraphStyle(name='TitleStyle', fontSize=32, leading=38, 
                                                 fontName='Helvetica-Bold', textColor=HexColor('#1a2a3a')) # Darker, larger
    custom_styles['SubtitleStyle'] = ParagraphStyle(name='SubtitleStyle', fontSize=18, leading=22, 
                                                    fontName='Helvetica', textColor=HexColor('#3a4a5a')) # Larger
    custom_styles['SystemDescription'] = ParagraphStyle(name='SystemDescription', fontSize=12, leading=15, 
                                                        fontName='Helvetica', textColor=HexColor('#4a4a4a')) # New style for description
    
    # Section Heading Styles
    custom_styles['Heading1'] = ParagraphStyle(name='Heading1', fontSize=20, leading=24, 
                                               fontName='Helvetica-Bold', textColor=HexColor('#2c3e50'))
    custom_styles['Heading2'] = ParagraphStyle(name='Heading2', fontSize=16, leading=18, 
                                               fontName='Helvetica-Bold', textColor=HexColor('#34495e'))
    
    # Body Text Styles
    custom_styles['Normal'] = ParagraphStyle(name='Normal', fontSize=10, leading=12, 
                                             fontName='Helvetica', textColor=black)
    custom_styles['Small'] = ParagraphStyle(name='Small', fontSize=8, leading=10, 
                                            fontName='Helvetica', textColor=grey)
    # New style for defect summary lines - Adjusted for better visibility
    custom_styles['DefectSummaryStyle'] = ParagraphStyle(name='DefectSummaryStyle', fontSize=9, leading=11, 
                                                         fontName='Helvetica', textColor=black)
    
    # Footer Style for the PDF pages - Used by PageNumCanvas
    custom_styles['PageFooter'] = ParagraphStyle(name='PageFooter', fontSize=8, leading=10, 
                                            fontName='Helvetica', textColor=grey)
    return custom_styles

_pdf_styles = get_pdf_styles() # Retrieve the cached styles dictionary


# ------------------- PAGE NUMBERING AND FOOTER HELPER CLASS -------------------
class PageNumCanvas(canvas.Canvas):
    """
    A custom Canvas subclass that automatically adds page numbers and a consistent footer
    to each page of the PDF.
    """
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
        # The footer text is now managed here globally for all pages
        self.page_footer_text = "Inspired by Intel AI for Manufacturing initiatives. © 2025 AI Defect Detection. All rights reserved."
        self.page_footer_style = _pdf_styles['PageFooter'] # Use the defined style

    def showPage(self):
        """
        Overrides the default `showPage`. Saves the current state of the canvas
        before starting a new page. This is crucial for drawing on previous pages later.
        """
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """
        Overrides the default `save` method. Iterates through all saved page states,
        draws the page number and footer on each, and then calls the original `save` method.
        This ensures page numbers and footers are added to all pages after all content is finalized.
        """
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state) # Restore the canvas state for the current page
            self.draw_page_info(num_pages) # Draw the page number and footer for this page
            canvas.Canvas.showPage(self) # Finalize the current page
        canvas.Canvas.save(self) # Save the entire document

    def draw_page_info(self, page_count):
        """
        Draws the current page number and total page count, and the footer text
        at the bottom center of the page.
        """
        # Draw page number
        self.setFont('Helvetica', 9)
        self.drawCentredString(A4[0]/2.0, 0.5 * inch, f"Page {self._pageNumber} of {page_count}")

        # Draw footer text
        # Use drawCentredString for the footer text to ensure it's centered without complex Paragraph flowables here
        footer_y_pos = 0.3 * inch # Slightly above the page number
        self.setFont(self.page_footer_style.fontName, self.page_footer_style.fontSize)
        self.setFillColor(self.page_footer_style.textColor)
        self.drawCentredString(A4[0]/2.0, footer_y_pos, self.page_footer_text)


# ------------------- CACHED UTILITY WRAPPERS -------------------
# These wrappers use Streamlit's caching mechanisms (`st.cache_data`)
# to prevent redundant computations on app reruns, improving performance.
@st.cache_data
def cached_overlay_predictions_on_image_pdf(
    image_bytes: bytes,
    masks: np.ndarray,
    colors: dict,
    img_width: int,
    img_height: int,
    selected_defects_for_display: List[str],
    class_thresholds: Dict[str, float]
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Caches the image overlay generation process for the PDF report.
    This function processes an input image and its prediction masks to
    create an overlaid image (with detected defects highlighted) and
    extracts detailed information about each detected defect.
    """
    return overlay_predictions_on_image(
        image_bytes, masks, colors, img_width, img_height, selected_defects_for_display, class_thresholds
    )

@st.cache_data
def cached_create_confidence_heatmap(
    probabilities: np.ndarray,
    original_image_np: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = 'viridis'
) -> np.ndarray:
    """
    Caches the confidence heatmap generation process for the PDF report.
    """
    return create_confidence_heatmap(probabilities, original_image_np, alpha, cmap_name)


# ------------------- MAIN PDF GENERATION FUNCTION -------------------
def generate_defect_report_pdf(all_defect_data_df: pd.DataFrame, batch_images_data: List[Dict[str, Any]]) -> bytes:
    """
    Generates a comprehensive PDF report for defect detection results using ReportLab.
    This function orchestrates the creation of the entire PDF document, including:
    - A title page with report metadata.
    - An overall batch summary section with tables and charts.
    - Detailed individual image analysis sections with original images,
      overlays, heatmaps, and per-image defect tables.

    Args:
        all_defect_data_df (pd.DataFrame): A Pandas DataFrame containing aggregated
                                           defect data for the entire batch.
        batch_images_data (List[Dict[str, Any]]): A list of dictionaries, where each
                                                  dictionary contains data for a single
                                                  processed image (e.g., 'id', 'name',
                                                  'bytes', 'predictions', 'inference_time').

    Returns:
        bytes: The complete PDF file content as a bytes object.
    """
    start_time_pdf_gen = time.time()
    logger.info("Starting PDF generation process...")

    buffer = io.BytesIO() # Create an in-memory buffer to store the generated PDF
    # Configure the SimpleDocTemplate with A4 page size and standard margins
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=inch/2, leftMargin=inch/2,
                            topMargin=inch/2, bottomMargin=inch/2)
    
    styles = _pdf_styles # Access the predefined ReportLab ParagraphStyle objects
    elements = [] # This list will hold all the ReportLab flowables to be added to the document

    # --- 1. Title Page Construction ---
    logger.info("Constructing the title page.")
    
    # Add a large initial spacer to push content down slightly for better visual balance
    elements.append(Spacer(0, 1.5 * inch))

    # Optional: Company Logo (if fetched successfully)
    logo_url = "https://placehold.co/200x70/4CAF50/FFFFFF?text=AI+QC+Logo" # Slightly larger, themed placeholder
    try:
        logo_response = requests.get(logo_url, timeout=5) # Set a timeout for network requests
        if logo_response.status_code == 200:
            logo_img_data = io.BytesIO(logo_response.content)
            logo = RLImage(logo_img_data)
            logo.drawWidth = 2.0 * inch # Increase logo size
            logo.drawHeight = logo.drawWidth * (70/200) # Maintain aspect ratio
            elements.append(logo)
            elements.append(Spacer(0, 0.5 * inch)) # More space after logo
        else:
            logger.warning(f"Could not fetch logo from {logo_url}. Status: {logo_response.status_code}. Skipping logo.")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Network error fetching logo from {logo_url}: {e}. Skipping logo.")
    except Exception as e:
        logger.warning(f"Unexpected error loading logo: {e}. Skipping logo.")

    # Main Titles
    elements.append(Paragraph("AI Surface Defect Detection Report", styles['TitleStyle']))
    elements.append(Spacer(0, 0.2 * inch))
    elements.append(Paragraph("Comprehensive Analysis of Material Surface Quality", styles['SubtitleStyle']))
    elements.append(Spacer(0, 0.8 * inch)) # More space before description

    # System Description
    elements.append(Paragraph("This report provides a detailed analysis of surface quality using advanced AI-powered defect detection. It summarizes batch performance and offers granular insights for each inspected item, including defect locations, areas, and confidence scores.", styles['SystemDescription']))
    elements.append(Spacer(0, 0.8 * inch)) # Space after description

    # Report Metadata
    elements.append(Paragraph(f"<b>Generated by:</b> AI Surface QC System on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(0, 0.1 * inch))
    elements.append(Paragraph(f"<b>Total Images Analyzed:</b> {len(batch_images_data)}", styles['Normal']))
    
    total_batch_inference_time = sum(img['inference_time'] for img in batch_images_data if img['predictions'] is not None)
    avg_inference_time_per_image = total_batch_inference_time / len(batch_images_data) if len(batch_images_data) > 0 else 0
    elements.append(Paragraph(f"<b>Total Inference Time for Batch:</b> {total_batch_inference_time:.4f} seconds", styles['Normal']))
    elements.append(Spacer(0, 0.1 * inch))
    elements.append(Paragraph(f"<b>Average Inference Time per Image:</b> {avg_inference_time_per_image:.4f} seconds", styles['Normal']))
    elements.append(Spacer(0, 0.1 * inch))
    elements.append(Paragraph("<b>Visualization Mode:</b> Contour Overlay", styles['Normal']))

    elements.append(Spacer(0, 1.0 * inch)) # More space before page break

    # The footer is now handled by PageNumCanvas for all pages.
    elements.append(PageBreak()) # Force a page break to start new section on a fresh page
    logger.info("Title page construction complete and page break inserted.")


    # --- 2. Overall Batch Summary Section ---
    logger.info("Constructing the 'Overall Batch Summary' section.")
    elements.append(Paragraph("1. Overall Batch Summary", styles['Heading1']))
    elements.append(Spacer(0, 0.15 * inch))

    if not all_defect_data_df.empty:
        elements.append(Paragraph("<b>Defect Occurrence and Average Metrics:</b>", styles['Normal']))
        elements.append(Spacer(0, 0.1 * inch))
        elements.append(Paragraph("<i>(Note: 'N/A' indicates the defect was not detected in any images, so average metrics are not applicable.)</i>", styles['Small']))
        elements.append(Spacer(0, 0.1 * inch))

        # Prepare data for the summary table. All cell content is wrapped in Paragraphs.
        summary_table_data = [
            [Paragraph("<b>Defect Type</b>", styles['Normal']),
             Paragraph("<b>Detected in Images</b>", styles['Normal']),
             Paragraph("<b>Avg. Area (when detected)</b>", styles['Normal']),
             Paragraph("<b>Avg. Confidence (when detected)</b>", styles['Normal'])]
        ]
        for class_name in CONFIG.CLASSES:
            class_df = all_defect_data_df[all_defect_data_df['Defect Type'] == class_name]
            detected_count = class_df[class_df['Detected'] == 'Yes']['Image Name'].nunique()
            
            avg_area = class_df[class_df['Detected'] == 'Yes']['Area (Pixels)'].mean()
            avg_area_str = f"{avg_area:.2f} px²" if not pd.isna(avg_area) else "N/A"

            avg_confidence = class_df[class_df['Detected'] == 'Yes']['Avg Confidence'].astype(float).mean()
            avg_confidence_str = f"{avg_confidence:.4f}" if not pd.isna(avg_confidence) else "N/A"

            summary_table_data.append([
                Paragraph(str(class_name), styles['Normal']),
                Paragraph(str(f"{detected_count}/{len(batch_images_data)} images"), styles['Normal']),
                Paragraph(str(avg_area_str), styles['Normal']),
                Paragraph(str(avg_confidence_str), styles['Normal'])
            ])
        
        logger.info("Building and styling summary table.")
        summary_table = Table(summary_table_data, colWidths=[1.5*inch, 1.5*inch, 2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4CAF50')), # Header background color
            ('TEXTCOLOR', (0, 0), (-1, 0), white), # Header text color
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')), # Alternating row background
            ('BACKGROUND', (0, 2), (-1, -1), white), # Alternating row background
            ('GRID', (0,0), (-1,-1), 1, grey), # Grid lines
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,0), 3), # Corrected padding for header row
            # Removed ALIGN commands from TableStyle
        ]))
        elements.append(summary_table)
        elements.append(Spacer(0, 0.3 * inch))
        logger.info("Summary table added and styled.")

        # Model Information
        elements.append(Paragraph("<b>Model Information:</b>", styles['Normal']))
        elements.append(Spacer(0, 0.1 * inch))
        elements.append(Paragraph(f"- Model Path: `{st.session_state.current_model_path}`", styles['Normal']))
        elements.append(Paragraph(f"- Encoder Backbone: `{st.session_state.selected_encoder_name}`", styles['Normal']))
        elements.append(Paragraph(f"- Device Used: `{CONFIG.DEVICE.type.upper()}`", styles['Normal']))
        elements.append(Spacer(0, 0.3 * inch))

        # Detection Thresholds
        elements.append(Paragraph("<b>Detection Thresholds Used:</b>", styles['Normal']))
        elements.append(Spacer(0, 0.1 * inch))
        for class_name, threshold in st.session_state.class_thresholds.items():
            elements.append(Paragraph(f"- {class_name}: `{threshold:.2f}`", styles['Normal']))
        elements.append(Spacer(0, 0.3 * inch))

        # Charts Section: Defect Distribution
        logger.info("Adding defect distribution charts.")
        elements.append(Paragraph("<b>Defect Distribution Charts:</b>", styles['Normal']))
        elements.append(Spacer(0, 0.1 * inch))

        # Chart 1: Defect Occurrence Across Images (Bar Chart)
        defect_occurrence_counts = all_defect_data_df[all_defect_data_df['Detected'] == 'Yes'].groupby('Defect Type')['Image Name'].nunique().reset_index(name='NumImagesDetected')
        if not defect_occurrence_counts.empty:
            fig_occurrence = px.bar(
                defect_occurrence_counts,
                x='Defect Type',
                y='NumImagesDetected',
                title='Number of Images Detected by Defect Type',
                labels={'Defect Type': 'Defect Type', 'NumImagesDetected': 'Number of Images Detected'},
                color='Defect Type',
                color_discrete_map={cls: f"rgb{CONFIG.COLORS[cls]}" for cls in CONFIG.CLASSES}
            )
            fig_occurrence.update_layout(xaxis_title="Defect Type", yaxis_title="Number of Images Detected")
            
            start_time_chart1 = time.time()
            try:
                img_bytes_occurrence = pio.to_image(fig_occurrence, format="png")
                logger.info(f"Occurrence chart image conversion took {time.time() - start_time_chart1:.4f} seconds.")
                img_rl_occurrence = RLImage(io.BytesIO(img_bytes_occurrence))
                
                chart_width_px_occ = fig_occurrence.layout.width if fig_occurrence.layout.width is not None else pio.defaults.default_width
                chart_height_px_occ = fig_occurrence.layout.height if fig_occurrence.layout.height is not None else pio.defaults.default_height
                
                aspect_ratio_occurrence = chart_height_px_occ / chart_width_px_occ if chart_width_px_occ > 0 else (pio.defaults.default_height / pio.defaults.default_width)
                
                img_rl_occurrence.drawWidth = 6.5 * inch # Scale chart to fit page width
                img_rl_occurrence.drawHeight = img_rl_occurrence.drawWidth * aspect_ratio_occurrence
                elements.append(img_rl_occurrence)
                elements.append(Spacer(0, 0.2 * inch))
                logger.info("Occurrence chart added successfully.")
            except Exception as e:
                logger.error(f"Error generating or embedding occurrence chart: {e}")
                elements.append(Paragraph("<i>(Error: Could not generate Defect Occurrence Chart)</i>", styles['Small']))
                elements.append(Spacer(0, 0.2 * inch))
        else:
            elements.append(Paragraph("No defects detected to plot occurrence.", styles['Small']))

        # Chart 2: Proportion of Total Detected Area by Defect Type (Pie Chart)
        all_defect_data_df['Area (Pixels)'] = pd.to_numeric(all_defect_data_df['Area (Pixels)'], errors='coerce').fillna(0)
        total_area_by_defect = all_defect_data_df[all_defect_data_df['Detected'] == 'Yes'].groupby('Defect Type')['Area (Pixels)'].sum().reset_index(name='TotalAreaPixels')
        
        if not total_area_by_defect.empty and total_area_by_defect['TotalAreaPixels'].sum() > 0:
            fig_pie = px.pie(
                total_area_by_defect,
                values='TotalAreaPixels',
                names='Defect Type',
                title='Proportion of Total Detected Area by Defect Type',
                color='Defect Type',
                color_discrete_map={cls: f"rgb{CONFIG.COLORS[cls]}" for cls in CONFIG.CLASSES}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
            
            start_time_chart2 = time.time()
            try:
                img_bytes_pie = pio.to_image(fig_pie, format="png")
                logger.info(f"Pie chart image conversion took {time.time() - start_time_chart2:.4f} seconds.")
                img_rl_pie = RLImage(io.BytesIO(img_bytes_pie))
                
                chart_width_px_pie = fig_pie.layout.width if fig_pie.layout.width is not None else pio.defaults.default_width
                chart_height_px_pie = fig_pie.layout.height if fig_pie.layout.height is not None else pio.defaults.default_height
                
                aspect_ratio_pie = chart_height_px_pie / chart_width_px_pie if chart_width_px_pie > 0 else (pio.defaults.default_height / pio.defaults.default_width)
                
                img_rl_pie.drawWidth = 6.5 * inch
                img_rl_pie.drawHeight = img_rl_pie.drawWidth * aspect_ratio_pie
                elements.append(img_rl_pie)
                elements.append(Spacer(0, 0.3 * inch))
                logger.info("Pie chart added successfully.")
            except Exception as e:
                logger.error(f"Error generating or embedding pie chart: {e}")
                elements.append(Paragraph("<i>(Error: Could not generate Defect Area Proportion Chart)</i>", styles['Small']))
                elements.append(Spacer(0, 0.3 * inch))
        else:
            elements.append(Paragraph("No defects detected with significant area to plot proportions.", styles['Small']))
            elements.append(Spacer(0, 0.3 * inch))

        logger.info("Charts added.")

        # Full Defect Detection Data Table
        elements.append(Paragraph("<b>Full Defect Detection Data Table:</b>", styles['Normal']))
        elements.append(Spacer(0, 0.1 * inch))
        
        # Prepare data for the full data table. All cell content is wrapped in Paragraphs.
        full_table_data = [
            [Paragraph("<b>Image Name</b>", styles['Normal']),
             Paragraph("<b>Defect Type</b>", styles['Normal']),
             Paragraph("<b>Detected</b>", styles['Normal']),
             Paragraph("<b>Area (Pixels)</b>", styles['Normal']),
             Paragraph("<b>Area %</b>", styles['Normal']),
             Paragraph("<b>Avg Confidence</b>", styles['Normal']),
             Paragraph("<b>Bounding Box (x,y,w,h)</b>", styles['Normal']),
             Paragraph("<b>Threshold Used</b>", styles['Normal'])]
        ]
        for index, row in all_defect_data_df.iterrows():
            detected_status = row['Detected']
            
            area_pixels_str = f"{row['Area (Pixels)']:.0f}" if detected_status == 'Yes' and not pd.isna(row['Area (Pixels)']) else '0'
            area_percent_str = f"{row['Area %']:.2f}%" if detected_status == 'Yes' and not pd.isna(row['Area %']) else '0.00%'
            avg_confidence_str = f"{row['Avg Confidence']:.4f}" if detected_status == 'Yes' and not pd.isna(row['Avg Confidence']) else '0.0000'
            bbox_str = row['Bounding Box (x,y,w,h)'] if detected_status == 'Yes' else "N/A"
            threshold_used_str = f"{row['Threshold Used']:.2f}" if not pd.isna(row['Threshold Used']) else 'N/A' 

            full_table_data.append([
                Paragraph(str(row['Image Name']), styles['Normal']),
                Paragraph(str(row['Defect Type']), styles['Normal']),
                Paragraph(str(detected_status), styles['Normal']),
                Paragraph(str(area_pixels_str), styles['Normal']),
                Paragraph(str(area_percent_str), styles['Normal']),
                Paragraph(str(avg_confidence_str), styles['Normal']),
                Paragraph(str(bbox_str), styles['Normal']),
                Paragraph(str(threshold_used_str), styles['Normal'])
            ])

        table = Table(full_table_data, colWidths=[1.3*inch, 1.1*inch, 0.7*inch, 1*inch, 0.7*inch, 1*inch, 1.5*inch, 1*inch])
        
        logger.info("Applying style to full defect detection data table.")
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4CAF50')), # Header background
            ('TEXTCOLOR', (0, 0), (-1, 0), white), # Header text color
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Bolding for header row
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')), # Alternating row background
            ('BACKGROUND', (0, 2), (-1, -1), white), # Alternating row background
            ('GRID', (0,0), (-1,-1), 1, grey), # Grid lines
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            # Removed ALIGN commands from TableStyle
        ]))
        elements.append(table)
        elements.append(Spacer(0, 0.3 * inch))
        logger.info("Full defect detection data table added and styled.")

    else:
        elements.append(Paragraph("No defect data available for this batch.", styles['Normal']))
    
    elements.append(PageBreak()) # Force a page break before individual image details
    logger.info("Page break after batch summary.")


    # --- 3. Individual Image Details Section ---
    elements.append(Paragraph("2. Individual Image Details", styles['Heading1']))
    elements.append(Spacer(0, 0.15 * inch))
    logger.info("Starting individual image details section.")

    for img_data in batch_images_data:
        logger.info(f"Processing image {img_data['name']} for detailed view.")
        
        # --- DEBUG LOG: Check if predictions are available ---
        if img_data['predictions'] is None:
            logger.debug(f"[DEBUG IMAGE] No predictions available for {img_data['name']}. Skipping overlay generation.")
            elements.append(Paragraph(f"<b>Image: {img_data['name']}</b> - No predictions available (processing might have failed).", styles['Normal']))
            elements.append(Spacer(0, 0.2 * inch))
            elements.append(PageBreak())
            continue # Skip to next image if no predictions

        elements.append(Paragraph(f"<b>Image: {img_data['name']}</b> (Inference Time: {img_data['inference_time']:.4f}s)", styles['Heading2']))
        elements.append(Spacer(0, 0.1 * inch))

        current_class_thresholds = st.session_state.get('class_thresholds', {cls: CONFIG.DEFAULT_THRESHOLD for cls in CONFIG.CLASSES})

        # --- Original Image ---
        img_rl_original = None
        max_img_width = A4[0] - inch # Max width for a single image on page
        try:
            original_pil_img = Image.open(io.BytesIO(img_data['bytes']))
            original_pil_img = original_pil_img.resize((CONFIG.IMG_WIDTH, CONFIG.IMG_HEIGHT), Image.LANCZOS)
            original_img_buffer = io.BytesIO()
            original_pil_img.save(original_img_buffer, format='PNG')
            original_img_buffer.seek(0)
            img_rl_original = RLImage(original_img_buffer)
            img_aspect_ratio_orig = original_pil_img.height / original_pil_img.width
            img_rl_original.drawWidth = min(max_img_width, original_pil_img.width)
            img_rl_original.drawHeight = img_rl_original.drawWidth * img_aspect_ratio_orig
            
            elements.append(Paragraph("<b>Original Image</b>", styles['Normal']))
            elements.append(Spacer(0, 0.1 * inch))
            elements.append(img_rl_original)
            elements.append(Spacer(0, 0.2 * inch))
            logger.debug(f"[DEBUG IMAGE] Original image {img_data['name']} processed and added to PDF.")
        except Exception as e:
            logger.error(f"[ERROR IMAGE] Error processing original image {img_data['name']} for PDF: {e}", exc_info=True)
            elements.append(Paragraph("<i>[Original Image N/A]</i>", styles['Small']))
            elements.append(Spacer(0, 0.2 * inch))

        # --- Detected Defects Overlay Image ---
        start_time_overlay_gen = time.time()
        overlay_img_np = None
        detailed_info_for_image = []
        try:
            overlay_img_np, detailed_info_for_image = cached_overlay_predictions_on_image_pdf(
                img_data['bytes'],
                img_data['predictions'],
                CONFIG.COLORS,
                CONFIG.IMG_WIDTH,
                CONFIG.IMG_HEIGHT,
                CONFIG.CLASSES, # Pass all configured classes for processing
                current_class_thresholds
            )
            logger.info(f"Overlay generated for {img_data['name']} in {time.time() - start_time_overlay_gen:.4f} seconds.")
            logger.debug(f"[DEBUG IMAGE] overlay_img_np shape: {overlay_img_np.shape if overlay_img_np is not None else 'None'}, dtype: {overlay_img_np.dtype if overlay_img_np is not None else 'None'}")
            logger.debug(f"[DEBUG IMAGE] detailed_info_for_image: {detailed_info_for_image}")

            if overlay_img_np is not None:
                overlay_pil_img = Image.fromarray(overlay_img_np.astype(np.uint8))
                overlay_img_buffer = io.BytesIO()
                overlay_pil_img.save(overlay_img_buffer, format='PNG')
                overlay_img_buffer.seek(0)
                img_rl_overlay = RLImage(overlay_img_buffer)
                img_aspect_ratio_overlay = overlay_pil_img.height / overlay_pil_img.width
                img_rl_overlay.drawWidth = min(max_img_width, overlay_pil_img.width)
                img_rl_overlay.drawHeight = img_rl_overlay.drawWidth * img_aspect_ratio_overlay

                elements.append(Paragraph("<b>Detected Defects Overlay</b>", styles['Normal']))
                elements.append(Spacer(0, 0.1 * inch))
                elements.append(img_rl_overlay)
                elements.append(Spacer(0, 0.2 * inch))
                logger.debug(f"[DEBUG IMAGE] Overlay image {img_data['name']} processed and added to PDF.")
            else:
                elements.append(Paragraph("<i>[Detected Defects Overlay N/A]</i>", styles['Small']))
                elements.append(Spacer(0, 0.2 * inch))
                logger.debug(f"[DEBUG IMAGE] overlay_img_np was None for {img_data['name']}. Using N/A placeholder.")

        except Exception as e:
            logger.error(f"[ERROR IMAGE] Error generating or embedding overlay for {img_data['name']}: {e}", exc_info=True)
            elements.append(Paragraph("<i>[Detected Defects Overlay N/A]</i>", styles['Small']))
            elements.append(Spacer(0, 0.2 * inch))


        # --- Confidence Heatmap Image ---
        heatmap_img_np = None
        try:
            # Re-open original image for heatmap as well, ensuring consistent dimensions
            original_image_for_heatmap_pil = Image.open(io.BytesIO(img_data['bytes']))
            original_image_for_heatmap_np = np.array(original_image_for_heatmap_pil.resize((CONFIG.IMG_WIDTH, CONFIG.IMG_HEIGHT)).convert("RGB"))

            heatmap_img_np = cached_create_confidence_heatmap(
                img_data['predictions'],
                original_image_for_heatmap_np,
                alpha=0.5, # Default alpha for heatmap
                cmap_name='viridis' # Default colormap
            )
            logger.debug(f"[DEBUG IMAGE] Heatmap generated for {img_data['name']}.")

            if heatmap_img_np is not None:
                heatmap_pil_img = Image.fromarray(heatmap_img_np.astype(np.uint8))
                heatmap_img_buffer = io.BytesIO()
                heatmap_pil_img.save(heatmap_img_buffer, format='PNG')
                heatmap_img_buffer.seek(0)
                img_rl_heatmap = RLImage(heatmap_img_buffer)
                img_aspect_ratio_heatmap = heatmap_pil_img.height / heatmap_pil_img.width
                img_rl_heatmap.drawWidth = min(max_img_width, heatmap_pil_img.width)
                img_rl_heatmap.drawHeight = img_rl_heatmap.drawWidth * img_aspect_ratio_heatmap

                elements.append(Paragraph("<b>Confidence Heatmap</b>", styles['Normal']))
                elements.append(Spacer(0, 0.1 * inch))
                elements.append(img_rl_heatmap)
                elements.append(Spacer(0, 0.2 * inch))
                logger.debug(f"[DEBUG IMAGE] Heatmap image {img_data['name']} processed and added to PDF.")
            else:
                elements.append(Paragraph("<i>[Confidence Heatmap N/A]</i>", styles['Small']))
                elements.append(Spacer(0, 0.2 * inch))
                logger.debug(f"[DEBUG IMAGE] heatmap_img_np was None for {img_data['name']}. Using N/A placeholder.")

        except Exception as e:
            logger.error(f"[ERROR IMAGE] Error generating or embedding heatmap for {img_data['name']}: {e}", exc_info=True)
            elements.append(Paragraph("<i>[Confidence Heatmap N/A]</i>", styles['Small']))
            elements.append(Spacer(0, 0.2 * inch))


        elements.append(Paragraph("<b>Defect Details for this Image:</b>", styles['Normal']))
        elements.append(Spacer(0, 0.1 * inch))
        
        # Per-image defect details table. All cell content is wrapped in Paragraphs.
        per_image_table_data = [
            [Paragraph("<b>DefectType</b>", styles['Normal']),
             Paragraph("<b>Detected</b>", styles['Normal']),
             Paragraph("<b>AreaPixels</b>", styles['Normal']),
             Paragraph("<b>Area Percentage</b>", styles['Normal']),
             Paragraph("<b>MaxConfidence</b>", styles['Normal']),
             Paragraph("<b>Threshold Used</b>", styles['Normal'])]
        ]
        for defect in detailed_info_for_image:
            area_percentage = (defect['area_pixels'] / (CONFIG.IMG_WIDTH * CONFIG.IMG_HEIGHT) * 100) if (CONFIG.IMG_WIDTH * CONFIG.IMG_HEIGHT) > 0 else 0
            
            detected_status = 'True' if defect['detected'] else 'False'
            area_pixels_str = f"{defect['area_pixels']:.0f}" if defect['detected'] else '0'
            area_percent_str = f"{area_percentage:.2f}%" if defect['detected'] else '0.00%'
            max_confidence_str = f"{defect['avg_confidence']:.2f}" if defect['detected'] else '0.00'
            threshold_used_str = f"{defect['threshold_used']:.2f}"

            per_image_table_data.append([
                Paragraph(str(defect['class_name']), styles['Normal']),
                Paragraph(str(detected_status), styles['Normal']),
                Paragraph(str(area_pixels_str), styles['Normal']),
                Paragraph(str(area_percent_str), styles['Normal']),
                Paragraph(str(max_confidence_str), styles['Normal']),
                Paragraph(str(threshold_used_str), styles['Normal'])
            ])
        
        logger.info("Building and styling per-image table.")
        per_image_table = Table(per_image_table_data, colWidths=[1.1*inch, 0.7*inch, 0.8*inch, 1*inch, 1*inch, 1*inch])
        per_image_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#ADD8E6')), # Header background
            ('TEXTCOLOR', (0, 0), (-1, 0), black), # Header text color
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F5F5F5')), # Alternating row background
            ('GRID', (0,0), (-1,-1), 0.5, grey), # Grid lines
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            # Removed ALIGN commands from TableStyle
        ]))
        elements.append(per_image_table)
        elements.append(Spacer(0, 0.2 * inch))
        logger.info("Per-image table added and styled.")

        # Summary of detected defects in text format
        detected_defects_text = []
        for defect in detailed_info_for_image:
            if defect['detected']:
                bbox_str = f"({defect['bounding_box'][0]}, {defect['bounding_box'][1]}, {defect['bounding_box'][2]}, {defect['bounding_box'][3]})" if defect['bounding_box'] else "N/A"
                detected_defects_text.append(
                    f"- <b>{defect['class_name']}</b>: Area {defect['area_pixels']:.0f} px², Conf {defect['avg_confidence']:.4f}, BB {bbox_str}, Threshold {defect['threshold_used']:.2f}"
                )
        if detected_defects_text:
            elements.append(Paragraph("<b>Summary of Detected Defects:</b>", styles['Normal']))
            for text in detected_defects_text:
                # Apply the new DefectSummaryStyle for better readability
                elements.append(Paragraph(text, styles['DefectSummaryStyle']))
        else:
            elements.append(Paragraph("No significant defects detected in this image at the current thresholds.", styles['Normal']))
        
        elements.append(Spacer(0, 0.3 * inch))
        elements.append(PageBreak()) # Force a page break after each image's details
        logger.info(f"Finished processing detailed view for image {img_data['name']}.")


    logger.info("All elements prepared. Attempting to build PDF document.")
    try:
        # Build the PDF document using the collected flowables and custom page numbering
        doc.build(elements, canvasmaker=PageNumCanvas)
        logger.info(f"PDF document built successfully in {time.time() - start_time_pdf_gen:.4f} seconds.")
        buffer.seek(0) # Rewind the buffer to the beginning to read its content
        return buffer.getvalue() # Return the generated PDF content as bytes
    except Exception as e:
        logger.exception(f"Critical error during PDF document build: {e}")
        # Re-raise a more informative RuntimeError for Streamlit to display
        raise RuntimeError(f"Failed to build PDF document: {e}. Check logs for details.")


# ------------------- STREAMLIT APPLICATION LAYOUT -------------------
# This section defines the Streamlit UI for the PDF Report Generation page.
st.title("PDF Report Generation")
st.markdown("""
    Generate a comprehensive PDF report from your batch analysis results.
    This report will include an overall summary and detailed information for each image,
    including overlaid defect visualizations and performance metrics.
""")

# Check if there's any processed batch data available in session state.
# PDF generation requires data from the 'Batch/Single Detection' page.
if 'uploaded_batch' not in st.session_state or not st.session_state.uploaded_batch:
    st.info("Please go to 'Batch/Single Detection' page, upload and process images first to generate a report.")
else:
    st.markdown("---")
    st.subheader("Batch Data Overview")
    
    # Count how many images in the batch have actually been processed (have predictions)
    processed_images_count = sum(1 for img in st.session_state.uploaded_batch if img['predictions'] is not None)
    total_images_count = len(st.session_state.uploaded_batch)

    if processed_images_count == 0:
        st.warning("No images have been successfully processed in the current batch. Please process images on the 'Batch/Single Detection' page.")
    else:
        st.info(f"Currently loaded batch has **{processed_images_count}** processed images out of **{total_images_count}**.")

        # Aggregate all defect data into a DataFrame for the report.
        # This ensures all necessary data is in a consistent format for PDF generation.
        all_defect_data_for_report = []
        for img_data in st.session_state.uploaded_batch:
            current_class_thresholds = st.session_state.get('class_thresholds', {cls: CONFIG.DEFAULT_THRESHOLD for cls in CONFIG.CLASSES})

            if img_data['predictions'] is not None:
                # Use the direct overlay function to get detailed defect info for aggregation
                _, detailed_info = overlay_predictions_on_image(
                    img_data['bytes'],
                    img_data['predictions'],
                    CONFIG.COLORS,
                    CONFIG.IMG_WIDTH,
                    CONFIG.IMG_HEIGHT,
                    CONFIG.CLASSES, # Get info for all defined classes
                    current_class_thresholds # Use current thresholds from session state
                )
                for defect in detailed_info:
                    bbox_str = f"({defect['bounding_box'][0]}, {defect['bounding_box'][1]}, {defect['bounding_box'][2]}, {defect['bounding_box'][3]})" if defect['bounding_box'] else 'N/A'
                    area_percentage = (defect['area_pixels'] / (CONFIG.IMG_WIDTH * CONFIG.IMG_HEIGHT) * 100) if (CONFIG.IMG_WIDTH * CONFIG.IMG_HEIGHT) > 0 else 0

                    all_defect_data_for_report.append({
                        'Image Name': img_data['name'],
                        'Defect Type': defect['class_name'],
                        'Detected': 'Yes' if defect['detected'] else 'No',
                        'Area (Pixels)': defect['area_pixels'], 
                        'Area %': area_percentage,
                        'Avg Confidence': defect['avg_confidence'], 
                        'Bounding Box (x,y,w,h)': bbox_str,
                        'Threshold Used': defect['threshold_used'] 
                    })
            else:
                # If an image was uploaded but not processed, add placeholder rows for it
                for class_name in CONFIG.CLASSES:
                    all_defect_data_for_report.append({
                        'Image Name': img_data['name'],
                        'Defect Type': class_name,
                        'Detected': 'No',
                        'Area (Pixels)': 0,
                        'Area %': 0.0,
                        'Avg Confidence': 0.0,
                        'Bounding Box (x,y,w,h)': 'N/A',
                        'Threshold Used': current_class_thresholds.get(class_name, CONFIG.DEFAULT_THRESHOLD)
                    })

        if all_defect_data_for_report:
            df_for_pdf = pd.DataFrame(all_defect_data_for_report)
            
            # Ensure numeric columns are truly numeric, coercing errors to NaN and then filling with 0
            numeric_cols = ['Area (Pixels)', 'Area %', 'Avg Confidence', 'Threshold Used']
            for col in numeric_cols:
                df_for_pdf[col] = pd.to_numeric(df_for_pdf[col], errors='coerce').fillna(0)

            st.markdown("---")
            st.subheader("Generate PDF Report")
            # Button to trigger PDF generation and download
            if st.button("Download PDF Report", key="download_pdf_button"):
                with st.spinner("Generating PDF report... This may take a moment."):
                    try:
                        pdf_bytes = generate_defect_report_pdf(df_for_pdf, st.session_state.uploaded_batch)
                        st.download_button(
                            label="Click to Download PDF",
                            data=pdf_bytes,
                            file_name="defect_report.pdf",
                            mime="application/pdf",
                            key="final_pdf_download" # Unique key for the download button
                        )
                        st.success("PDF report generated successfully!")
                        logger.info("PDF report generated and download button displayed.")
                    except Exception as e:
                        st.error(f"Failed to generate PDF report: {e}")
                        logger.exception("Error generating PDF report:") # Log full traceback
            st.info("The PDF report will include detailed tables and overlaid images for each processed item in the batch.")
        else:
            st.info("No data available to generate a PDF report. Please ensure images are uploaded and processed on the 'Batch/Single Detection' page.")


st.markdown("---")
st.markdown("""
    <div class='footer'>
        Developed using Streamlit & PyTorch | Inspired by Intel AI for Manufacturing initiatives.
        <br>
        © 2025 AI Defect Detection. All rights reserved.\
    </div>
""", unsafe_allow_html=True)
