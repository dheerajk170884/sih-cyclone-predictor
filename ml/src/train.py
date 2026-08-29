import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import timm

# Pretrained EfficientNet Backbone
class CycloneIntensityClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(CycloneIntensityClassifier, self).__init__()
        # EfficientNet-B0 pretrained
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CycloneIntensityClassifier(num_classes=6).to(device)
    print(f"Cyclone Intensity Model initialized successfully on: {device}")
    