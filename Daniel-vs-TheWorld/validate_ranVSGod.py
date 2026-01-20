import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --------------------
# Model-definition (uændret)
# --------------------
IMAGE_SIZE = 64

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
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --------------------
# Load model
# --------------------
device = torch.device("cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("RandomsVsGodkendte_model.pth", map_location=device))
model.eval()

# --------------------
# Transform (skal matche træning)
# --------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# --------------------
# Load multiple images
# --------------------
IMAGE_DIR = r"C:\Users\DSkov\OneDrive\Studie\Programmering\GitHub\computer-vision-2026\Daniel-vs-TheWorld\Billeder"  # mappe med billeder
images = []
filenames = []

for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(IMAGE_DIR, file)
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)
        filenames.append(file)

if not images:
    raise RuntimeError("Ingen billeder fundet")

batch = torch.stack(images).to(device)

# --------------------
# Inference (batch)
# --------------------
with torch.no_grad():
    outputs = model(batch)
    probs = torch.softmax(outputs, dim=1)
    confidences, predictions = torch.max(probs, 1)

classes = ["Godkendte", "Randoms"]

# --------------------
# Resultater
# --------------------
for file, pred, conf in zip(filenames, predictions, confidences):
    print(f"{file:25} → {classes[pred.item()]:10} ({conf.item()*100:.1f}%)")
