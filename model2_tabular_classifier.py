import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD CLINICAL DATA
# ============================================================

clinical_df = pd.read_csv("clinical_raw.csv")
severity_df = pd.read_csv("severity_labels.csv")
infection_df = pd.read_csv("infection_labels.csv")

# Merge severity + infection + clinical
merged = clinical_df.merge(severity_df, on="Patient_ID").merge(infection_df, on="Patient_ID")


# ============================================================
# 2. CREATE RISK LABEL (High/Low infection risk)
#    This is synthetic but based on clinical markers.
# ============================================================

def compute_risk(row):
    score = 0
    if row["HbA1c"] > 8: score += 1
    if row["CRP"] > 8: score += 1
    if row["NLR"] > 3: score += 1
    if row["IL6"] > 20: score += 1
    if row["beta_hydroxybutyrate"] > 0.9: score += 1
    
    return "High" if score >= 2 else "Low"

merged["Risk"] = merged.apply(compute_risk, axis=1)


# ============================================================
# 3. FEATURES & LABELS
# ============================================================

features = [
    "FPG","PPG","OGTT","HbA1c",
    "Neutrophil_Count","Lymphocyte_Count","NLR",
    "CRP","IL6","IL17","TNF_alpha",
    "beta_hydroxybutyrate","Urine_albumin",
    "BMI","Age","Duration_of_Diabetes"
]

X = merged[features]

y_risk = merged["Risk"]                     # Risk model labels
y_severity = merged["Severity"]             # Severity model labels


# ============================================================
# 4. TRAIN–VAL SPLIT
# ============================================================

X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
    X, y_risk, test_size=0.2, random_state=42
)

X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
    X, y_severity, test_size=0.2, random_state=42
)


# ============================================================
# 5. SCALING
# ============================================================

scaler = StandardScaler()
X_train_r = scaler.fit_transform(X_train_r)
X_val_r = scaler.transform(X_val_r)

X_train_s = scaler.fit_transform(X_train_s)
X_val_s = scaler.transform(X_val_s)


# ============================================================
# 6. LIGHTGBM MODELS
# ============================================================

risk_model = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=-1,
    random_state=42
)

severity_model = lgb.LGBMClassifier(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=-1,
    random_state=42,
    objective="multiclass",
    class_weight={
        "mild": 1.0,
        "moderate": 1.5,
        "severe": 20.0
    }
)



# ============================================================
# 7. TRAIN MODELS
# ============================================================

print("\nTraining Risk Model...")
risk_model.fit(X_train_r, y_train_r)

print("\nTraining Severity Model...")
severity_model.fit(X_train_s, y_train_s)


# ============================================================
# 8. EVALUATE MODELS
# ============================================================

print("\n============= RISK MODEL EVALUATION =============\n")
risk_preds = risk_model.predict(X_val_r)
print("Accuracy:", accuracy_score(y_val_r, risk_preds))
print(classification_report(y_val_r, risk_preds))

print("\n============= SEVERITY MODEL EVALUATION =============\n")
severity_preds = severity_model.predict(X_val_s)
print("Accuracy:", accuracy_score(y_val_s, severity_preds))
print(classification_report(y_val_s, severity_preds))


# ============================================================
# 9. SAVE MODELS
# ============================================================

joblib.dump(risk_model, "risk_classifier.pkl")
joblib.dump(severity_model, "severity_classifier.pkl")

print("\nModels saved as:")
print(" - risk_classifier.pkl")
print(" - severity_classifier.pkl")


# ============================================================
# 10. OPTIONAL: CONFUSION MATRIX
# ============================================================

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


plot_conf_matrix(y_val_r, risk_preds, "Risk Model Confusion Matrix")
plot_conf_matrix(y_val_s, severity_preds, "Severity Model Confusion Matrix")


# ============================================================
# 11. INFERENCE EXAMPLE
# ============================================================

def predict_patient(data_dict):
    df = pd.DataFrame([data_dict])
    df_scaled = scaler.transform(df)
    
    risk_out = risk_model.predict(df_scaled)[0]
    severity_out = severity_model.predict(df_scaled)[0]
    
    return risk_out, severity_out


print("\nExample prediction:")
example = X.iloc[0].to_dict()
print(predict_patient(example))
