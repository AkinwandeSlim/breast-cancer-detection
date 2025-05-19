

---

# 🩺 Breast Cancer Detection System

The **Breast Cancer Detection System** is a deep learning web application that assists in **early breast cancer diagnosis** through medical image analysis. Built using **PyTorch** and **Streamlit**, it empowers healthcare professionals with interactive tissue visualizations and image classification tools. This project highlights my capabilities in **AI for healthcare**, **computer vision**, and **intuitive UI/UX design**.

---

## 📋 Table of Contents

* [🧠 Overview](#-overview)
* [🎥 Demo](#-demo)
* [✨ Features](#-features)
* [⚙️ Technical Details](#-technical-details)
* [🛠️ Skills Demonstrated](#-skills-demonstrated)
* [🚀 Setup Instructions](#-setup-instructions)
* [🖱️ Usage Guide](#-usage-guide)
* [⚠️ Limitations](#️-limitations)
* [📓 Notebooks](#-notebooks)
* [📬 Contact](#-contact)

---

## 🧠 Overview

This system processes breast tissue images to detect cancerous regions, supporting radiologists and researchers with early warning insights. Powered by a **ResNet18** deep learning model, it enables real-time tissue slice analysis and patch-level classification via a **Streamlit interface**.

🔍 **Core Highlights**:

* Deploys **CNN models** for medical imaging.
* Processes complex datasets using **Pandas** and **NumPy**.
* Offers interactive visualization via **Streamlit**.
* Provides probability-based cancer risk assessment.

---

## 🎥 Demo

📺 [Watch Demo on YouTube](https://www.youtube.com/watch?v=pXYgyJ3ne7A)
*See real-time tissue analysis, patch prediction, and UI in action.*

---

## ✨ Features

### 🔬 Tissue Visualization

* **Tissue Slice** – Composite image of a patient's tissue patches.
* **Cancer Mask** – Overlay showing predicted cancerous regions.
* **Heatmap** – Visual risk prediction map (Yellow-Red scale).
* **Expandable Logs** – View broken or unprocessed patches.

![Tissue Visualization](Screenshot%202025-05-19%20140634.jpg)

---

### 🖼️ Patch Prediction

* Upload individual image patches (JPG/PNG).
* Receive classification: **Cancerous** or **Not Cancerous**.
* Displays confidence scores and styled results (red/green).

![Patch Prediction](Screenshot%202025-05-19%20140955.jpg)

---

## ⚙️ Technical Details

### 🧩 Model Architecture

* **Base**: ResNet18 (TorchVision)
* **Custom Head**: Linear (512 → 256 → 2) with:

  * ReLU
  * Batch Normalization
  * Dropout (p=0.5)
* **Initialization**: Xavier Uniform
* **Device Support**: CPU and GPU-compatible

### 🧪 Data Pipeline

* **Input Structure**: `Breast_cancer_patient/` → `0` (non-cancer) & `1` (cancer) folders
* **Patch Coordinates**: Extracted with **Pandas**
* **Transforms**: Resize (50×50), normalize, and flip
* **Loader**: Custom `BreastCancerDataset` integrated with PyTorch’s DataLoader

### 📊 Visualization Tools

* Reconstructs tissue slices using `x,y` coordinates
* Cancer masks using `matplotlib` + transparency overlays
* Heatmaps with `YlOrRd` colormap
* Live interactivity with **Streamlit**

### 🧭 UI Navigation

* **Sidebar Pages**:

  * Tissue Visualization
  * Patch Prediction
* **User Feedback**: Spinners, alerts, and validation messages

---

## 🛠️ Skills Demonstrated

* 🧠 **Deep Learning** – Custom CNNs for image classification
* 🐍 **Python** – Clean, modular, scalable code
* 🔥 **PyTorch** – Model building, training, evaluation
* 🌐 **Streamlit** – Interactive web interface
* 📊 **Data Analysis** – Preprocessing with Pandas, NumPy
* 🎨 **Visualization** – Matplotlib, Seaborn
* 📁 **Software Engineering** – Reproducibility, error handling, clean UI

---

## 🚀 Setup Instructions

### ✅ Prerequisites

* Python 3.8+
* Git
* Required Libraries:

```bash
pip install torch torchvision streamlit pandas numpy matplotlib seaborn scikit-learn pillow scikit-image tensorflow tqdm
```

### 📦 Installation

```bash
git clone https://github.com/AkinwandeSlim/breast-cancer-detection.git
cd breast-cancer-detection
```

1. **Add Data & Model Files**:

   * Place your patient image folders into `Breast_cancer_patient/`
   * Place trained model (`model.pth` or `model_cuda.pth`) into `breast_data/`

2. **Launch App**:

```bash
streamlit run breast_cancer_app.py
```

Visit `http://localhost:8501` in your browser.

---

## 🖱️ Usage Guide

### 🔬 Tissue Visualization

* Select a patient folder
* Click **"Generate Visualization"**
* View:

  * Tissue Slice
  * Cancer Mask Overlay
  * Risk Heatmap
* See logs for unprocessed patches

### 🖼️ Patch Prediction

* Upload image patch
* See predicted label + confidence
* Visual style:

  * ✅ Green = Not Cancerous
  * ❌ Red = Cancerous

---

## ⚠️ Limitations

| Area                   | Current Limitation                                                           |
| ---------------------- | ---------------------------------------------------------------------------- |
| File Paths             | Hardcoded paths (e.g., `/content/drive/...`) must be changed per environment |
| Prediction Assumptions | Ground truth assumed from filename (ending in `0` or `1`)                    |
| Feature Scope          | Advanced models like Vision Transformers are not yet integrated              |
| Robustness             | May not handle missing/invalid inputs in some edge cases                     |
| Privacy                | Dataset and model not included (available on request)                        |

---

## 📓 Notebooks

| Notebook                                          | Description                      |
| ------------------------------------------------- | -------------------------------- |
| `BREAST CANCER DETECTION AND VISUALISATION.ipynb` | Data exploration, model training |
| `BREAST TISSUE VISUAL APP.ipynb`                  | UI/UX prototyping in Streamlit   |

---

## 📬 Contact

💡 Interested in AI for medical diagnostics?
📨 Reach out for collaboration or full access to the dataset and model:

* 📧 **Email**: [alexdata2022@gmail.com](mailto:alexdata2022@gmail.com)
* 🔗 **LinkedIn**: [akinwandealex](https://www.linkedin.com/in/akinwandealex)
* 💻 **GitHub**: [AkinwandeSlim](https://github.com/AkinwandeSlim)

---

## ❤️ Built with Passion

This project is a testament to using **AI for good**. By combining technical depth with real-world usability, it brings us one step closer to better, earlier cancer diagnosis.

---

