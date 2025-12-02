import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------
# DATASET PATH — your real + synthetic images are here
# ------------------------------------------------------
root = "synthetic_image_data/train"
val_root = "synthetic_image_data/val"

# ------------------------------------------------------
# TRANSFORMS
# ------------------------------------------------------
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------------------------------------------
# LOAD DATASETS
# ------------------------------------------------------
train_ds = datasets.ImageFolder(root=root, transform=train_tf)
val_ds = datasets.ImageFolder(root=val_root, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

class_names = train_ds.classes
print("Classes:", class_names)

# ------------------------------------------------------
# MODEL
# ------------------------------------------------------
model = models.resnet50(weights=None)
num_feats = model.fc.in_features
model.fc = nn.Linear(num_feats, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ------------------------------------------------------
# TRAINING
# ------------------------------------------------------
EPOCHS = 10  # good for real+synthetic
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # VALIDATION
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = correct / total * 100
    print(f"EPOCH {epoch+1}/{EPOCHS}   Loss={total_loss:.3f}   Val Acc={acc:.2f}%")

# ------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------
save_data = {
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "architecture": "resnet50"
}

torch.save(save_data, "resnet_infection_classifier.pt")
print("Saved: resnet_infection_classifier.pt 🚀")
