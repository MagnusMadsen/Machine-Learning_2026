import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --------------------
# Samme model-definition som træning
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
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))
model.eval()

# --------------------
# Transform (SKAL matche træning)
# --------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# --------------------
# Load image
# --------------------
image_path = "IMG_0334.jpeg"  # ← dit billede
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# --------------------
# Inference
# --------------------
with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

classes = ["cat", "dog"]

print(f"Prediction: {classes[predicted.item()]}")
print(f"Confidence: {confidence.item() * 100:.1f}%")
