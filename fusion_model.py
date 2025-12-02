import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

print("Fusion model started...")

# ===========================
#  REBUILD THE IMAGE MODEL
# ===========================
class InfectionClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=5):
        super(InfectionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),   # model.0
            nn.ReLU(),                  # model.1
            nn.Dropout(0.25),           # model.2
            nn.Linear(512, 256),        # model.3
            nn.ReLU(),                  # model.4
            nn.Dropout(0.25),           # model.5
            nn.Linear(256, num_classes) # model.6
        )

    def forward(self, x):
        return self.model(x)

# ===========================
#  LOAD THE IMAGE MODEL
# ===========================
print("Loading infection_classifier.pt...")

saved = torch.load("infection_classifier.pt", map_location="cpu")
state_dict = saved["model_state_dict"]

image_model = InfectionClassifier()
image_model.load_state_dict(state_dict)
image_model.eval()

class_names = saved["class_names"]
print("Loaded infection_classifier.pt with classes:", class_names)

# ===========================
#  LOAD OTHER MODELS
# ===========================
risk_model = joblib.load("risk_classifier.pkl")
severity_model = joblib.load("severity_classifier.pkl")
print("Loaded risk & severity models.")

# ===========================
#  LOAD DATA
# ===========================
embeddings = np.load("image_embeddings.npy")
clinical_raw = pd.read_csv("clinical_raw.csv")
print("Loaded embeddings & clinical data.")
print("COLUMNS:", clinical_raw.columns.tolist())


# ===========================
#  RUN A SAMPLE FUSION PREDICTION
# ===========================

sample_idx = 0
img_embed = torch.tensor(embeddings[sample_idx], dtype=torch.float32).unsqueeze(0)

# Image prediction
with torch.no_grad():
    img_logits = image_model(img_embed)
    img_pred_idx = img_logits.argmax(dim=1).item()
    img_pred_label = class_names[img_pred_idx]

# Risk & Severity
sample_row = clinical_raw.iloc[sample_idx]
# Prepare clinical features (drop non-numerical columns)
features = sample_row.drop([
    "Patient_ID",
    "Infection_Type",
    "Severity"
]).values.reshape(1, -1)


risk_pred = risk_model.predict(features)[0]
severity_pred = severity_model.predict(features)[0]


print("\n========= FUSION OUTPUT =========")
print("Infection Type (Image) :", img_pred_label)
print("Risk Level (Tabular)   :", risk_pred)
print("Severity (Tabular)     :", severity_pred)
print("================================\n")
