import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --------------------
# Config
# --------------------
BATCH_SIZE = 8
IMAGE_SIZE = 64
EPOCHS = 12
DATA_DIR = "data/RandomsVSGodkendte"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------
# Transforms
# --------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# --------------------
# Datasets
# --------------------
train_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/val",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0   # vigtigt på Windows
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", train_dataset.classes)

# --------------------
# Simple CNN
# --------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Randoms vs Godkendte
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = SimpleCNN().to(device)

# --------------------
# Loss & optimizer
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------
# Training loop
# --------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # --------------------
    # Validation
    # --------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {avg_loss:.3f} "
          f"Val Acc: {accuracy:.2f}%")
    
torch.save(model.state_dict(), "RandomsVsGodkendte_model.pth")
print("Model saved")
print("Training finished")
