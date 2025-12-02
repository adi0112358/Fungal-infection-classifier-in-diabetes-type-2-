import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =============================
# PATHS (update if needed)
# =============================
train_dir = "synthetic_image_data/train"
val_dir = "synthetic_image_data/val"

# =============================
# TRANSFORMS
# =============================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = ImageFolder(train_dir, transform=train_tf)
val_data = ImageFolder(val_dir, transform=val_tf)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

class_names = train_data.classes
print("Classes:", class_names)

# =============================
# BUILD RESNET50 PRETRAINED
# =============================
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_feats = model.fc.in_features
model.fc = nn.Linear(num_feats, len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# =============================
# TRAINING LOOP
# =============================
EPOCHS = 7

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # VALIDATION
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total * 100
    print(f"EPOCH {epoch+1}/{EPOCHS}  Loss={total_loss:.3f}   Val Acc={val_acc:.2f}%")

# =============================
# SAVE MODEL
# =============================
save_dict = {
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "architecture": "resnet50"
}

torch.save(save_dict, "resnet_infection_classifier.pt")
print("Saved model → resnet_infection_classifier.pt")
