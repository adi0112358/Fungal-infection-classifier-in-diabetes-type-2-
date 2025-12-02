import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ======== DATA AUGMENTATION ============
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========= LOAD DATASETS ===========
train_data = datasets.ImageFolder("image_data/train", transform=train_transforms)
val_data   = datasets.ImageFolder("image_data/val", transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

class_names = train_data.classes
print("Classes:", class_names)

# ======= MODEL SETUP (ResNet50) =========
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Replace final FC layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model = model.to(device)

# ====== Training Setup =======
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ====== TRAINING LOOP =========
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ===== Validation ======
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total * 100

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}  Val Acc: {val_acc:.2f}%")

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names
}, "resnet_infection_classifier.pt")

print("\nModel saved as resnet_infection_classifier.pt")
