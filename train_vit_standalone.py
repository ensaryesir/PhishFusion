"""
ViT Siamese Fine-Tuning - Standalone Script
PhishFusion workspace için optimize edilmiş versiyon
"""

import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

# Import ViT from timm (already installed in pixi)
import timm


# ==================== KONFİGÜRASYON ====================

CONFIG = {
    # Data
    'TRAIN_DIR': 'datasets/OCR_aided_Siamese_model/3061_training_set/logos',
    
    # Model
    'VIT_MODEL': 'vit_base_patch16_224',
    'PRETRAINED': True,
    'NUM_CLASSES': 277,  # 277 brand
    
    # Training
    'EPOCHS': 30,
    'BATCH_SIZE': 16,
    'LR': 1e-4,
    'VAL_SPLIT': 0.2,
    
    # Output
    'OUTPUT_DIR': 'vit_checkpoints',
    'SAVE_FREQ': 5,
    
    # Device
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print("\n" + "="*70)
print("ViT Siamese Fine-Tuning - Standalone Version")
print("="*70)
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# ==================== DATASET ====================

class LogoDataset(Dataset):
    """Simple logo dataset"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        brand_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for brand_idx, brand_dir in enumerate(brand_dirs):
            logos = list(brand_dir.glob('*.png')) + list(brand_dir.glob('*.jpg'))
            for logo_path in logos:
                self.images.append(str(logo_path))
                self.labels.append(brand_idx)
        
        print(f"\n✓ Dataset: {len(self.images)} images, {len(brand_dirs)} brands")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except:
            return torch.zeros(3, 224, 224), self.labels[idx]


# ==================== MODEL ====================

class ViTClassifier(nn.Module):
    """ViT-based classifier"""
    
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.classifier = nn.Linear(self.vit.num_features, num_classes)
    
    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)


# ==================== TRAINING ====================

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': total_loss/(pbar.n+1), 'acc': 100.*correct/total})
    
    return total_loss/len(loader), 100.*correct/total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss/len(loader), 100.*correct/total


def main():
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = LogoDataset(CONFIG['TRAIN_DIR'], transform)
    
    val_size = int(len(dataset) * CONFIG['VAL_SPLIT'])
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    print(f"Training: {len(train_set)}, Validation: {len(val_set)}")
    
    # Model
    model = ViTClassifier(CONFIG['VIT_MODEL'], CONFIG['NUM_CLASSES'], CONFIG['PRETRAINED'])
    model = model.to(CONFIG['DEVICE'])
    
    print(f"\n✓ Model: {CONFIG['VIT_MODEL']}")
    print(f"✓ Device: {CONFIG['DEVICE']}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LR'])
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_acc = 0
    log = []
    
    for epoch in range(1, CONFIG['EPOCHS']+1):
        print(f"\nEpoch {epoch}/{CONFIG['EPOCHS']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['DEVICE'], epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['DEVICE'])
        
        log.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 
                   'val_loss': val_loss, 'val_acc': val_acc})
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{CONFIG['OUTPUT_DIR']}/best_model.pth")
            print(f"✓ New best: {val_acc:.2f}%")
        
        # Save checkpoint
        if epoch % CONFIG['SAVE_FREQ'] == 0:
            torch.save(model.state_dict(), f"{CONFIG['OUTPUT_DIR']}/epoch_{epoch}.pth")
    
    # Save final
    torch.save(model.state_dict(), f"{CONFIG['OUTPUT_DIR']}/last_model.pth")
    
    with open(f"{CONFIG['OUTPUT_DIR']}/log.json", 'w') as f:
        json.dump(log, f, indent=2)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(f"\nModels saved:")
    print(f"  {CONFIG['OUTPUT_DIR']}/best_model.pth")
    print(f"  {CONFIG['OUTPUT_DIR']}/last_model.pth")
    print(f"\nNext: Copy best_model.pth to PhishIntention/checkpoints/")


if __name__ == '__main__':
    main()
