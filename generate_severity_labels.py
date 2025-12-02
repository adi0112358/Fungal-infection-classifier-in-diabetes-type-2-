import pandas as pd

# Load the clinical dataset
df = pd.read_csv("clinical_raw.csv")

def compute_severity(row):
    score = 0
    
    # CRP
    if row["CRP"] > 60: score += 4
    elif row["CRP"] > 20: score += 3
    elif row["CRP"] > 8: score += 2
    else: score += 1

    # IL-6
    if row["IL6"] > 80: score += 4
    elif row["IL6"] > 40: score += 3
    elif row["IL6"] > 20: score += 2
    else: score += 1

    # HbA1c
    if row["HbA1c"] > 9: score += 3
    elif row["HbA1c"] > 7.5: score += 2
    else: score += 1

    # NLR
    if row["NLR"] > 4: score += 3
    elif row["NLR"] > 2.5: score += 2

    # Ketones
    if row["beta_hydroxybutyrate"] > 1.2: score += 3
    elif row["beta_hydroxybutyrate"] > 0.8: score += 2

    # Assign final class
    if score >= 14:
        return "critical"
    elif score >= 10:
        return "severe"
    elif score >= 6:
        return "moderate"
    else:
        return "mild"

# Apply severity scoring
df["Severity"] = df.apply(compute_severity, axis=1)

# Save new severity file
df[["Patient_ID", "Severity"]].to_csv("severity_labels.csv", index=False)

print("New severity_labels.csv generated successfully!")
