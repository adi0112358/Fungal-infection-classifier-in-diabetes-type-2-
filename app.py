import io
import numpy as np
from PIL import Image

import streamlit as st        # <-- THIS MUST COME BEFORE ST.WRITE()
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
import os                     # <-- import os anywhere after imports

# ===== DEBUG: VERIFY PATHS & MODEL FILE =====
st.write("CWD:", os.getcwd())
st.write("Model exists:", os.path.exists("/Users/phoenix/Desktop/ML peoject/resnet_infection_classifier.pt"))
st.write("Model size (MB):", os.path.getsize("/Users/phoenix/Desktop/ML peoject/resnet_infection_classifier.pt") / 1e6)



# ============ DEVICE ============
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ============ IMAGE MODEL ============
class ResNetInfectionClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # The saved model is a raw ResNet50, so we build the SAME structure:
        self.convnet = models.resnet50(weights=None)

        # Replace FC layer to match the saved number of classes
        num_feats = self.convnet.fc.in_features
        self.convnet.fc = nn.Linear(num_feats, num_classes)

    def forward(self, x):
        return self.convnet(x)




@st.cache_resource
def load_image_model():
    saved = torch.load("resnet_infection_classifier.pt", map_location=device)
    class_names = saved["class_names"]

    model = ResNetInfectionClassifier(num_classes=len(class_names))

    state_dict = saved["model_state_dict"]

    # FIX STATE DICT KEY NAMES
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict["convnet." + k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model, class_names




@st.cache_resource
def load_tabular_models():
    risk_model = joblib.load("risk_classifier.pkl")
    severity_model = joblib.load("severity_classifier.pkl")
    return risk_model, severity_model


# ============ IMAGE PREPROCESSING ============
img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def predict_infection_type(image: Image.Image, model, class_names):
    img = image.convert("RGB")
    x = img_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
    return class_names[idx], probs[idx]


# ============ TABULAR PREDICTION ============
def predict_risk_severity(features_np, risk_model, severity_model):
    """
    features_np: shape (1, 16)
    order: [FPG, PPG, OGTT, HbA1c,
            Neutrophil_count, Lymphocyte_count, NLR,
            CRP, IL6, IL17, TNF_alpha,
            beta_hydroxybutyrate, Urine_albumin,
            Age, BMI, Diabetes_duration]
    """
    risk_pred = risk_model.predict(features_np)[0]
    severity_pred = severity_model.predict(features_np)[0]
    return risk_pred, severity_pred


# ============ STREAMLIT UI ============
st.title("Diabetes Fungal Infection Decision Support (Prototype)")
st.write("Upload a fungal infection image and enter clinical data to get infection type, risk, and severity (synthetic prototype).")

# Load models
try:
    image_model, class_names = load_image_model()
except Exception as e:
    st.error(f"Error loading image model: {e}")
    st.stop()

try:
    risk_model, severity_model = load_tabular_models()
except Exception as e:
    st.error(f"Error loading tabular models: {e}")
    st.stop()

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Infection Image")
    uploaded_file = st.file_uploader(
        "Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"]
    )

    image_obj = None
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image_obj = Image.open(io.BytesIO(bytes_data))
        st.image(image_obj, caption="Uploaded Image", use_container_width=True)


with col2:
    st.subheader("2. Enter Clinical Biomarkers")

    FPG = st.number_input("FPG", value=110.0)
    PPG = st.number_input("PPG", value=160.0)
    OGTT = st.number_input("OGTT", value=150.0)
    HbA1c = st.number_input("HbA1c", value=7.0)

    Neutrophil_count = st.number_input("Neutrophil_count", value=4.5)
    Lymphocyte_count = st.number_input("Lymphocyte_count", value=2.0)
    NLR = st.number_input("NLR", value=2.2)

    CRP = st.number_input("CRP", value=6.0)
    IL6 = st.number_input("IL6", value=12.0)
    IL17 = st.number_input("IL17", value=18.0)
    TNF_alpha = st.number_input("TNF_alpha", value=20.0)

    beta_hydroxybutyrate = st.number_input("beta_hydroxybutyrate", value=0.6)
    Urine_albumin = st.number_input("Urine_albumin", value=30.0)
    Age = st.number_input("Age", value=50)
    BMI = st.number_input("BMI", value=27.0)
    Diabetes_duration = st.number_input("Diabetes_duration (years)", value=5.0)

    features = np.array([[
        FPG, PPG, OGTT, HbA1c,
        Neutrophil_count, Lymphocyte_count, NLR,
        CRP, IL6, IL17, TNF_alpha,
        beta_hydroxybutyrate, Urine_albumin,
        Age, BMI, Diabetes_duration
    ]])

st.markdown("---")

if st.button("Run Diagnosis"):
    if image_obj is None:
        st.error("Please upload an infection image first.")
    else:
        with st.spinner("Running models..."):
            # Image branch
            inf_type, inf_conf = predict_infection_type(image_obj, image_model, class_names)

            # Tabular branch
            risk_pred, severity_pred = predict_risk_severity(features, risk_model, severity_model)

        st.subheader("Diagnosis Result (Prototype)")
        st.write(f"**Infection Type (Image model)**: {inf_type}  (confidence ~{inf_conf*100:.1f}%)")
        st.write(f"**Risk Level (Tabular model)**: {risk_pred}")
        st.write(f"**Severity (Tabular model)**: {severity_pred}")

        st.info("Note: Image model is trained on synthetic data only (prototype, not for clinical use).")
