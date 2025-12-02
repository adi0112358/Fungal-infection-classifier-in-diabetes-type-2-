import pandas as pd
import numpy as np

# Load infection labels
df = pd.read_csv("infection_labels.csv")

N = len(df)
dim = 512  # embedding dimension

# Define cluster centers for each infection type
cluster_centers = {
    "Candida": np.random.normal(0, 1, dim),
    "Mucormycosis": np.random.normal(3, 1, dim),
    "Dermatophyte": np.random.normal(-3, 1, dim),
    "Aspergillus": np.random.normal(1.5, 1, dim)
}

embeddings = np.zeros((N, dim))

# Generate clustered embeddings
for i, infection in enumerate(df["Infection_Type"]):
    center = cluster_centers[infection]
    # Add small noise around the center
    embeddings[i] = center + np.random.normal(0, 0.5, dim)

# Save new embeddings
np.save("image_embeddings.npy", embeddings)

print("Clustered embeddings generated successfully!")
print("Saved as image_embeddings.npy")
