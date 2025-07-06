
# 🔍 AI-Based Surface Defect Detection System

A real-time, web-based application for detecting surface defects in industrial metal components using deep learning-based semantic segmentation (U-Net with ResNet34). Built with **Streamlit** and **PyTorch**, the app provides batch and webcam-based defect analysis, heatmap visualization, per-class thresholding, and downloadable reports.

---

## 🚀 Features

- 🧾 **Batch & Single Image Processing**
- 📸 **Webcam Capture Support**
- 🎨 **Real-time Visualization** (Contours + Probability Heatmaps)
- 📉 **Class-wise Thresholding**
- 📊 **Defect Distribution Graphs (Plotly)**
- 📥 **ZIP/CSV Report Downloads**
- ⚡ **GPU/CPU compatible PyTorch backend**

---

## 📌 Project Highlights

- ✅ **Model**: U-Net (ResNet34 encoder) trained on annotated industrial surface defect dataset.
- 🧠 **Classes**: Defect 1, Defect 2, Defect 3, Defect 4.
- 🧪 **Evaluation Metrics**:
  | Metric | Defect 1 | Defect 2 | Defect 3 | Defect 4 |
  |--------|----------|----------|----------|----------|
  | Precision | 0.7334 | 0.7342 | 0.7323 | 0.7326 |
  | Recall    | 0.7597 | 0.7569 | 0.7591 | 0.7581 |
  | F1-Score  | 0.7463 | 0.7454 | 0.7454 | 0.7451 |
  | IoU       | 0.5953 | 0.5941 | 0.5942 | 0.5938 |

- 📦 **Deployment**: Available via [Render](https://render.com/) / Streamlit Cloud (based on your deployment).
- 🖥️ **Interface**: Upload ZIP/images or use live webcam for instant detection.
- 📊 **Outputs**: Contour overlays, raw probability heatmaps, CSV summaries, ZIP report downloads.
- ⚙️ **Fully configurable**: Threshold sliders, class selection, visualization mode toggle.

---

## 🏗️ Tech Stack

- **Python 3.10+**
- **PyTorch**
- **Streamlit**
- **Albumentations** (image preprocessing)
- **Segmentation Models PyTorch**
- **OpenCV & Matplotlib**
- **Plotly** (charts)
- **Pillow, Pandas, NumPy**

---

## 📂 Project Structure

```
D-Visual-Inspection-for-Quality-Control-in-Manufacturing/
├── .stramlit/
│   └── config.toml
├── data/
│   ├── dataset.md
│   └── train.csv
├── main.ipynb
├── app.py
├── config.py
├── model.py
├── transforms.py
├── utils.py
├── requirements.txt
├── models/
│   └── uploaded_model_unet_qc_trained.pth
├── pages/
│   ├── 1_Batch_Single_Detection.py
│   ├── 3_PDF_Report_Generation.py
│   └── 4_Model_Management.py
├── README.md
└── .gitignore
```

---

## 💻 How to Run Locally

```bash
git clone https://github.com/yourusername/defect-detection-app.git
cd defect-detection-app

# Setup environment
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🔧 Setup Notes

- Make sure `.pth` is in the root directory.
- Supports both CPU and GPU (automatically detected).
- Default model input size: **128x800** (resize automatically applied).

---

## 📈 Future Improvements

- Integrate custom dataset uploader and re-training pipeline
- Add YOLOv8 real-time inference mode
- Improve mIoU via class-weighted loss or augmentation
- Add multilingual support for UI

---

## 📜 License

MIT License

---
