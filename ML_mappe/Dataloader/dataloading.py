import  os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import cv2
import numpy as np
from PIL import Image

'''
Dette er den lidt lettere måde at lave datasæt på end den indbyggede metode i PyTorch.
Det er lettere fordi vi bare ændre i stierne hvis vi vil have andet data.
'''

# Hvilke mapper vores billededata ligger i
trainDir = "../../images/pss/train/" 
testDir = "../../images/pss/test/"

# Hvordan vi vil behandle billederne. Her bare en resize til at gøre alle billederne mindre, og random flips.
transform = transforms.Compose([
            transforms.Resize(size=(64,64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
            ])

trainData = datasets.ImageFolder(root=trainDir,
                                 transform = transform,
                                 target_transform=None)
print(f"Train data:\n{trainData}")

testData = datasets.ImageFolder(root=testDir,
                                transform = transform,
                                target_transform=None)
print(f"Test data:\n{testData}")
