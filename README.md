# Breast Cancer Detection System

The **Breast Cancer Detection System** is a deep learning application designed to aid early breast cancer detection through image analysis. Using **PyTorch** and **Streamlit**, it visualizes breast tissue and classifies image patches as cancerous or non-cancerous, offering insights for medical professionals. This project showcases my expertise in deep learning, data processing, and user-friendly interface design, targeting healthcare providers, researchers, and data science teams.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technical Details](#technical-details)
- [Skills Demonstrated](#skills-demonstrated)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Limitations](#limitations)
- [Notebooks](#notebooks)
- [Contact](#contact)

## Overview

This system analyzes breast tissue images to identify cancerous regions, supporting early diagnosis. Built on a **ResNet18** model (PyTorch), it processes patient data, generates tissue visualizations, and predicts cancer likelihood. A **Streamlit** interface enables intuitive interaction, making it accessible for medical professionals to explore results and classify new images.

Key highlights:
- Deploying deep learning for medical imaging.
- Analyzing complex datasets with **Pandas** and **NumPy**.
- Building interactive interfaces with **Streamlit**.
- Creating visualizations with **Matplotlib** and **Seaborn**.

## Features

### 1. Tissue Visualization
- **Interactive Display**: Select a patient folder to view:
  - **Tissue Slice**: Composite image of tissue patches.
  - **Cancer Mask**: Red overlay highlighting cancerous areas.
  - **Heatmap**: Probability map of cancer risk (yellow-orange-red colormap).
- **Error Handling**: Lists unprocessed patches in an expandable section.
- **User Experience**: Features loading spinners and clear feedback.

![Tissue Visualization](Screenshot%202025-05-19%20140634.jpg)
*Visualization page showing tissue slice, cancer mask, and probability heatmap.*

### 2. Patch Prediction
- **Classification**: Upload a tissue patch (JPG, JPEG, PNG) to predict **Cancerous** or **Not Cancerous**.
- **Confidence Scores**: Shows prediction confidence.
- **Design**: Clean layout with image previews and styled results (green for non-cancerous, red for cancerous).

![Patch Prediction](Screenshot%202025-05-19%20140955.jpg)
*Patch prediction page displaying an uploaded image and classification result.*

## Technical Details

### Model
- **Architecture**: Modified **ResNet18** (torchvision) for binary classification.
- **Layers**: Custom fully connected layers (512, 256, 2) with ReLU, batch normalization, and dropout (0.5).
- **Initialization**: Xavier uniform for stable training.
- **Device**: Supports CPU/GPU with dynamic model loading.

### Data Processing
- **Dataset**: Patient data in `Breast_cancer_patient/` with `0` (non-cancerous) and `1` (cancerous) subfolders.
- **Preprocessing**: **Pandas** parses filenames for coordinates (x, y) into DataFrames.
- **Transforms**: **torchvision** resizes images to 50x50, normalizes, and applies random flips.
- **Dataset**: Custom `BreastCancerDataset` for efficient **DataLoader** integration.

### Visualization
- **Reconstruction**: Grids patches using coordinates, with red cancer masks.
- **Heatmap**: Maps predictions to a `YlOrRd` colormap via **Matplotlib**.
- **Streamlit**: Interactive rendering with error details.

### Interface
- **Pages**: “Tissue Visualization” and “Patch Prediction” via sidebar navigation.
- **UX**: Spinners, styled text, and responsive column layouts.
- **Errors**: Warns on missing data or invalid uploads.

## Skills Demonstrated
- **Deep Learning**: Building CNNs for medical imaging.
- **PyTorch**: Custom models and data pipelines.
- **Streamlit**: Interactive web apps for data science.
- **Python**: Modular code with **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**.
- **Data Preprocessing**: Managing complex image datasets.
- **Visualization**: Balancing technical and accessible visuals.
- **Engineering**: Clear documentation and user-focused design.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git ([git-scm.com](https://git-scm.com))
- Dependencies:
  ```bash
  pip install torch torchvision streamlit pandas numpy matplotlib seaborn scikit-learn pillow scikit-image tensorflow tqdm

Installation

Clone the repository:git clone https://github.com/AkinwandeSlim/breast-cancer-detection.git
cd breast-cancer-detection


Add dataset and model:
Place patient data in Breast_cancer_patient/ (subfolders 0, 1).
Place model (.pth or _cuda.pth) in breast_data/.
Note: Dataset/model not included due to privacy/size. Contact me for access.


Run:streamlit run breast_cancer_app.py


Opens at http://localhost:8501.



Usage
Tissue Visualization

Select a patient folder.
Click “Generate Visualization” for tissue slice, cancer mask, and heatmap.
Check “Broken Patches” for processing errors.

Patch Prediction

Upload a tissue patch.
View prediction (“Cancerous”/“Not Cancerous”) and confidence.
See image preview with styled results.

Limitations

Paths: Hardcoded (/content/drive/MyDrive/...), needing environment adjustments.
Prediction: Assumes filename ends with 0 or 1 for ground truth.
Features: Disabled components (e.g., Vision Transformer) await integration.
Dependencies: Specific library versions required.
Errors: Robust but may fail on edge cases (e.g., missing models).

These are opportunities for future improvements, and I’m ready to address them.
Notebooks
The repository includes:

BREAST CANCER DETECTION AND VISUALISATION.ipynb: Data analysis and model development.
BREAST TISSUE VISUAL APP.ipynb: Streamlit app prototyping.

These provide insights into the project’s development process.
Contact
I’m passionate about deep learning in healthcare. Reach out to discuss this project or collaboration:


- Email: alexdata2022@gmail.com
- LinkedIn: Your LinkedIn Profile
- GitHub: https://github.com/AkinwandeSlim


Built with ❤️ for advancing medical diagnostics through AI.```
