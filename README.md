# Fungal Infection Classifier in Type-2 Diabetes

A multi-modal Machine Learning system that classifies **fungal infection type**, estimates **severity**, and integrates **clinical biomarkers** with **image-based deep learning** for improved diagnosis in diabetes-associated fungal infections.

---

## 🚀 Project Overview

Type-2 Diabetes Mellitus (T2DM) increases the risk of fungal infections such as:

- **Aspergillus**
- **Dermatophytes**
- **Mucormycosis**

This project performs:

### ✔ Image Classification  
Fine-tuned **ResNet** model classifying fungal infection types from lesion images.

### ✔ Biomarker-Based Clinical Prediction  
LightGBM model trained on biomarkers like:  
FPG, PPG, OGTT, HbA1c, CRP, IL-6, IL-17, TNF-α, NLR, Neutrophils, Lymphocytes, β-hydroxybutyrate, Urine albumin, Age, BMI, Diabetes Duration, etc.

### ✔ Severity Prediction  
Synthetic-label generator + model to estimate fungal infection severity.

### ✔ Fusion Model  
Combines:
- Deep image embeddings  
- Clinical biomarkers  

into a single classifier for improved accuracy.

## 🧠 Model Architectures

### 🔹 Image Classifier (ResNet)
- ResNet-18 / ResNet-50
- Pretrained on ImageNet → fine-tuned
- Outputs: Aspergillus / Dermatophyte / Mucormycosis

### 🔹 Biomarker Model (LightGBM)
- 16–20 clinical features
- Tabular risk classification
- Feature importance supported

### 🔹 Fusion Model
- Image Embedding → Dense(256) → ReLU
- Biomarkers → Dense(32) → ReLU
- Concatenate → Dense(128) → Dense(3) → Softmax


---

## 🛠 Installation

### 1. Create environment
- python3 -m venv ml_env
- source ml_env/bin/activate
  
### 2. Install dependencies
- pip install -r torch torchvision lightgbm numpy pandas scikit-learn matplotlib opencv-python Pillow seaborn tqdm scipy joblib



---

## 🧪 Biomarker Inputs
Includes:

- FPG, PPG, OGTT
- HbA1c
- Neutrophil count, Lymphocyte count
- NLR
- CRP
- IL-6, IL-17, TNF-α
- β-hydroxybutyrate
- Urine albumin
- Age, BMI
- Diabetes Duration

---

## 📊 Synthetic Data
Scripts generate:

- Synthetic fungal images
- Synthetic biomarker datasets
- Synthetic severity labels
- Merged datasets

---

## 🏁 Future Improvements
- SHAP explainability
- Web UI (Streamlit)
- Deployment API
- More augmentation and fusion variants

---
### To run:
- "streamlit run app.py" in the terminal and hit!



