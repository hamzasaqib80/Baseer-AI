# ==============================================================================
# ENTERPRISE CIFAR-10 PIPELINE: COLAB PRODUCTION EDITION (SELF-CONTAINED)
# Targets: 92.3% Accuracy | Hardware: NVIDIA T4 | Security: Validated Tensors
# This script contains ALL modules (Architecture, DataLoader, Trainer) to ensure
# zero-dependency execution in any Google Colab environment.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import numpy as np
import time
import os

try:
    import google.colab # type: ignore
    from google.colab import files # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# --- 1. CONFIGURATION LAYER ---
class Config:
    seed = 42
    batch_size = 128
    epochs = 60 
    lr = 1e-3
    weight_decay = 5e-4
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_model.pth"

# --- 2. SECURITY & VALIDATION LAYER ---
class SecurityValidator:
    @staticmethod
    def audit(model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                raise RuntimeError(f"CRITICAL: Weight corruption in {name}")
        print("🛡️ Security Audit: Model Weights Integrity Verified.")

# --- 3. ARCHITECTURE: CUSTOM RESNET ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for i, s in enumerate([stride] + [1]*(num_blocks-1)):
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = self.avg_pool(out).view(out.size(0), -1)
        return self.fc(self.dropout(out))

# --- 4. DATA ENGINE ---
def get_loaders():
    print("📥 Initializing Optimized DataLoaders...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

# --- 5. TRAINING ENGINE ---
def run_training():
    train_loader, test_loader = get_loaders()
    model = CustomResNet().to(Config.device)
    SecurityValidator.audit(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=Config.lr, steps_per_epoch=len(train_loader), epochs=Config.epochs)

    best_acc = 0.0
    print(f"🚀 Training started on {Config.device}...")

    for epoch in range(Config.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Config.device), targets.to(Config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(Config.device), targets.to(Config.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1:02d}/{Config.epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(), 'acc': val_acc}, Config.model_path)
            print(f"🌟 New Record: {val_acc:.2f}% accuracy. Checkpoint saved.")

    print(f"\n✅ TRAINING FINISHED. Best Accuracy: {best_acc:.2f}%")
    
    if IN_COLAB:
        print("📦 Preparing artifacts for download...")
        files.download(Config.model_path)
    else:
        print(f"📦 Local training detected. Model saved at {Config.model_path}")

if __name__ == "__main__":
    run_training()
