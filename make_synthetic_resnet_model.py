import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using:", device)

# -------------------------------
# LOAD SYNTHETIC IMAGE DATA
# -------------------------------
root = "synthetic_image_data/train"
classes = sorted(os.listdir(root))
class_to_idx = {c: i for i, c in enumerate(classes)}

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class ImageDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for label in os.listdir(root):
            folder = os.path.join(root, label)
            for fname in os.listdir(folder):
                if fname.lower().endswith(("png", "jpg", "jpeg")):
                    self.samples.append((os.path.join(folder, fname), class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return tf(img), label

dataset = ImageDataset(root)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Synthetic image dataset loaded.")
print("Total images:", len(dataset))

# -------------------------------
# BUILD RESNET-50
# -------------------------------
model = models.resnet50(weights=None)
num_feats = model.fc.in_features
model.fc = nn.Linear(num_feats, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# -------------------------------
# TRAIN 2 EPOCHS
# -------------------------------
for epoch in range(2):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/2 Loss: {total_loss:.4f}")

# -------------------------------
# SAVE CORRECT CHECKPOINT
# -------------------------------
save_dict = {
    "model_state_dict": model.state_dict(),
    "class_names": classes,
    "architecture": "resnet50"
}

torch.save(save_dict, "resnet_infection_classifier.pt")
print("Saved: resnet_infection_classifier.pt")
