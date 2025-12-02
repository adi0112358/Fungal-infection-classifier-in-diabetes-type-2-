import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. DEVICE SETUP — USE MPS ON MACBOOK M1/M2/M3
# ============================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2. CUSTOM DATASET (embeddings + labels)
# ============================================================
class InfectionDataset(Dataset):
    def __init__(self, embedding_path, label_path):
        self.embeddings = np.load(embedding_path)
        labels_df = pd.read_csv(label_path)

        # Encode labels (string → number)
        le = LabelEncoder()
        self.labels = le.fit_transform(labels_df["Infection_Type"])

        # Save class names for inference later
        self.class_names = list(le.classes_)

        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ============================================================
# 3. SIMPLE FULLY CONNECTED CLASSIFIER
# ============================================================
class InfectionClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=5):
        super(InfectionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# 4. LOAD DATA
# ============================================================
embedding_path = "image_embeddings.npy"
label_path = "infection_labels.csv"

dataset = InfectionDataset(embedding_path, label_path)

# Train-val split (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)


# ============================================================
# 5. SETUP MODEL, LOSS, OPTIMIZER
# ============================================================
model = InfectionClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15

# ============================================================
# 6. TRAINING LOOP
# ============================================================
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # VALIDATION
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {total_loss:.3f}  Val Accuracy: {val_acc:.2f}%")

# ============================================================
# 7. SAVE MODEL
# ============================================================
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": dataset.class_names
}, "infection_classifier.pt")

print("\nModel training complete!")
print("Saved as infection_classifier.pt")


# ============================================================
# 8. OPTIONAL — INFERENCE FUNCTION (for testing)
# ============================================================
def predict(embedding_vector):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(embedding_vector, dtype=torch.float32).to(device)
        output = model(x)
        _, pred = torch.max(output.data, 0)
        return dataset.class_names[pred.item()]

