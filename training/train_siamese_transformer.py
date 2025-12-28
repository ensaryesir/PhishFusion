"""
Improved ViT Siamese Fine-Tuning Script
Incorporates best practices:
- Progressive unfreezing (freeze backbone initially)
- Stronger data augmentation
- Learning rate scheduling
- Better error handling
- Early stopping
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
import timm
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import json
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


# ===================== CONFIGURATION =====================
CONFIG = {
    'TRAIN_DIR': '../datasets/OCR_aided_Siamese_model/3061_training_set/logos',  # Relative to project root
    'VIT_MODEL': 'swin_tiny_patch4_window7_224',  # 28M params - fair comparison with ResNet50 (25M) ✅
    'PRETRAINED': True,
    'NUM_CLASSES': 277,
    
    # TRAINING SCHEDULE
    'EPOCHS': 50,
    'BATCH_SIZE': 16,              # Per-GPU batch size
    'ACCUMULATION_STEPS': 2,       # ← BASELINE: Effective batch = 32 (proven optimal)
    'LR': 1e-4,                    # Optimal for Swin-Tiny
    'VAL_SPLIT': 0.2,
    'USE_STRATIFIED_SPLIT': False,
    
    # OUTPUT
    'OUTPUT_DIR': 'checkpoints_swin_tiny',  # Relative to training folder
    'SAVE_FREQ': 5,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # UNFREEZING STRATEGY
    'FREEZE_EPOCHS': 8,            # Balanced freeze period
    'PARTIAL_UNFREEZE': True,      # Only unfreeze last 2 blocks + norm
    'EARLY_STOP_PATIENCE': 10,
    
    # REGULARIZATION (BASELINE - PROVEN!)
    'WEIGHT_DECAY': 0.01,          # ← BASELINE: Proven optimal regularization
    'LABEL_SMOOTHING': 0.1,        # Standard
    
    # SYSTEM
    'NUM_WORKERS': 0,              # Windows: must be 0
    'PERSISTENT_WORKERS': False,
    'USE_WEIGHTED_SAMPLER': True,  # Class imbalance handling
    'GRADIENT_CLIP_NORM': 1.0,     # Gradient stability
    'USE_MIXED_PRECISION': True,   # ~30% faster (CUDA only)
    
    # AUGMENTATION (NO MIXUP - Proven best for small dataset)
    'USE_MIXUP': False,            # ← DISABLED: Proven to hurt accuracy on small dataset
    'MIXUP_ALPHA': 0.2,
    'USE_CUTMIX': False,
    'CUTMIX_ALPHA': 1.0,
    'AUGMENT_PROB': 0.5,
}


# ===================== MODEL =====================
class ImprovedViTSiamese(nn.Module):
    """Improved ViT with progressive unfreezing"""
    
    def __init__(self, model_name, num_classes, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Freeze backbone initially
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("🔒 Backbone frozen for initial training")
        
        # Classifier with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.vit.num_features, num_classes)
        )
    
    def unfreeze_backbone(self, partial=False, num_blocks=2):
        """Unfreeze backbone for fine-tuning (supports both ViT and Swin)"""
        
        # Detect architecture type
        is_swin = hasattr(self.vit, 'layers')  # Swin uses 'layers'
        is_vit = hasattr(self.vit, 'blocks')   # ViT uses 'blocks'
        
        if partial:
            if is_swin:
                # Swin: Unfreeze last N layers
                total_layers = len(self.vit.layers)
                for i in range(total_layers - num_blocks, total_layers):
                    for param in self.vit.layers[i].parameters():
                        param.requires_grad = True
                
                # Unfreeze norm layer
                if hasattr(self.vit, 'norm'):
                    for param in self.vit.norm.parameters():
                        param.requires_grad = True
                
                print(f"🔓 Partially unfrozen: last {num_blocks} Swin layers + norm")
                
            elif is_vit:
                # ViT: Unfreeze last N transformer blocks
                total_blocks = len(self.vit.blocks)
                for i in range(total_blocks - num_blocks, total_blocks):
                    for param in self.vit.blocks[i].parameters():
                        param.requires_grad = True
                
                # Unfreeze norm layer
                if hasattr(self.vit, 'norm'):
                    for param in self.vit.norm.parameters():
                        param.requires_grad = True
                
                print(f"🔓 Partially unfrozen: last {num_blocks} ViT blocks + norm")
            else:
                print("⚠️ Unknown architecture, unfreezing all parameters")
                for param in self.vit.parameters():
                    param.requires_grad = True
        else:
            # Unfreeze all parameters
            for param in self.vit.parameters():
                param.requires_grad = True
            
            arch_name = "Swin" if is_swin else "ViT" if is_vit else "Unknown"
            print(f"🔓 {arch_name} backbone fully unfrozen")
    
    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)


# ===================== DATASET =====================
class ImprovedLogoDataset(Dataset):
    """Dataset with stronger augmentation"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Load images and labels
        brand_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for class_idx, brand_dir in enumerate(brand_dirs):
            self.class_names.append(brand_dir.name)
            img_files = list(brand_dir.glob('*.png')) + list(brand_dir.glob('*.jpg'))
            
            for img_file in img_files:
                self.images.append(str(img_file))
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, self.labels[idx]
        
        except Exception as e:
            print(f"⚠️ Error loading {self.images[idx]}: {e}")
            # Return a blank image instead
            return torch.zeros(3, 224, 224), self.labels[idx]


# ===================== DATA AUGMENTATION =====================
def get_transforms(is_training=True):
    """Get transforms with stronger augmentation for training"""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),  # Original value
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Original
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Original
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Original
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation: Mix two images and their labels
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation: Cut and paste patches between images
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Generate random box
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x_cutmix = x.clone()
    x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    
    return x_cutmix, y_a, y_b, lam


# ===================== EARLY STOPPING =====================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
        
        return self.early_stop


# ===================== TRAINING =====================
def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=1, 
                scaler=None, gradient_clip_norm=None, use_mixup=False, use_cutmix=False,
                mixup_alpha=0.2, cutmix_alpha=1.0, augment_prob=0.5):
    """Train for one epoch with gradient accumulation, mixed precision, and advanced augmentation"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    use_amp = scaler is not None
    optimizer.zero_grad()  # Zero gradients at start
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Apply Mixup or CutMix randomly
        use_advanced_aug = (use_mixup or use_cutmix) and (np.random.rand() < augment_prob)
        
        if use_advanced_aug:
            # Randomly choose between Mixup and CutMix
            if use_mixup and use_cutmix:
                apply_mixup = np.random.rand() < 0.5
            elif use_mixup:
                apply_mixup = True
            else:
                apply_mixup = False
            
            if apply_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            else:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha)
        
        # Mixed precision forward pass
        if use_amp:
            with autocast():
                outputs = model(images)
                if use_advanced_aug:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam) / accumulation_steps
                else:
                    loss = criterion(outputs, labels) / accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            if use_advanced_aug:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam) / accumulation_steps
            else:
                loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
        
        # Update weights after accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                # Gradient clipping
                if gradient_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Gradient clipping
                if gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        
        # Calculate accuracy (for Mixup/CutMix, use primary label)
        if use_advanced_aug:
            correct += (lam * predicted.eq(labels_a).sum().item() + 
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.3f}',
            'acc': f'{100.*correct/total:.1f}'
        })
    
    # Final gradient update if last batch didn't complete accumulation
    if (len(dataloader) % accumulation_steps) != 0:
        if use_amp:
            if gradient_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        optimizer.zero_grad()
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


# ===================== MAIN =====================
def main():
    print("="*70)
    print("Improved ViT Siamese Fine-Tuning")
    print("="*70)
    
    # Print configuration
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {CONFIG['TRAIN_DIR']}")
    
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    full_dataset = ImprovedLogoDataset(CONFIG['TRAIN_DIR'], transform=None)
    
    # Stratified split to ensure equal class distribution
    if CONFIG['USE_STRATIFIED_SPLIT']:
        print("\n📊 Using stratified split for balanced validation set")
        labels = [full_dataset.labels[i] for i in range(len(full_dataset))]
        
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=CONFIG['VAL_SPLIT'],
            random_state=42
        )
        train_idx, val_idx = next(sss.split(range(len(full_dataset)), labels))
        train_indices = train_idx.tolist()
        val_indices = val_idx.tolist()
    else:
        # Regular random split
        val_size = int(len(full_dataset) * CONFIG['VAL_SPLIT'])
        train_size = len(full_dataset) - val_size
        train_indices, val_indices = random_split(
            range(len(full_dataset)), 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        train_indices = train_indices.indices
        val_indices = val_indices.indices
    
    # Create separate datasets with different transforms
    class TransformedSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            try:
                original_idx = self.indices[idx]
                img = Image.open(self.dataset.images[original_idx]).convert('RGB')
                label = self.dataset.labels[original_idx]
                
                if self.transform:
                    img = self.transform(img)
                
                return img, label
            except Exception as e:
                print(f"⚠️ Error loading {self.dataset.images[original_idx]}: {e}")
                # Return blank tensor with correct label
                return torch.zeros(3, 224, 224), self.dataset.labels[original_idx]
    
    train_dataset = TransformedSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformedSubset(full_dataset, val_indices, val_transform)
    
    print(f"✓ Dataset: {len(full_dataset)} images, {CONFIG['NUM_CLASSES']} brands")
    print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Weighted sampler for class imbalance
    train_sampler = None
    shuffle = True
    
    if CONFIG['USE_WEIGHTED_SAMPLER']:
        print("\n⚖️  Using weighted sampler for balanced training")
        train_labels = [full_dataset.labels[i] for i in train_indices]
        class_counts = Counter(train_labels)
        weights = [1.0 / class_counts[label] for label in train_labels]
        train_sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True
        )
        shuffle = False  # Sampler handles shuffling
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True if CONFIG['DEVICE'] == 'cuda' else False,
        persistent_workers=CONFIG['PERSISTENT_WORKERS'] if CONFIG['NUM_WORKERS'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True if CONFIG['DEVICE'] == 'cuda' else False,
        persistent_workers=CONFIG['PERSISTENT_WORKERS'] if CONFIG['NUM_WORKERS'] > 0 else False
    )
    
    # Create model
    print(f"\n🤖 Creating model: {CONFIG['VIT_MODEL']}")
    model = ImprovedViTSiamese(
        CONFIG['VIT_MODEL'],
        CONFIG['NUM_CLASSES'],
        pretrained=CONFIG['PRETRAINED'],
        freeze_backbone=True  # Start with frozen backbone
    ).to(CONFIG['DEVICE'])
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['LABEL_SMOOTHING'])
    print(f"\n🎯 Label smoothing: {CONFIG['LABEL_SMOOTHING']}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['LR'],
        weight_decay=CONFIG['WEIGHT_DECAY']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['EPOCHS']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['EARLY_STOP_PATIENCE'])
    
    # Mixed precision scaler (CUDA only)
    scaler = None
    if CONFIG['USE_MIXED_PRECISION'] and CONFIG['DEVICE'] == 'cuda':
        scaler = GradScaler()
        print(f"⚡ Mixed precision training enabled")
    elif CONFIG['USE_MIXED_PRECISION'] and CONFIG['DEVICE'] == 'cpu':
        print(f"⚠️  Mixed precision disabled (CPU mode)")
    
    # Training history
    history = []
    best_val_acc = 0.0
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    # Training loop
    for epoch in range(1, CONFIG['EPOCHS'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['EPOCHS']}")
        
        # Unfreeze backbone after freeze epochs
        if epoch == CONFIG['FREEZE_EPOCHS'] + 1:
            model.unfreeze_backbone(
                partial=CONFIG['PARTIAL_UNFREEZE'],
                num_blocks=2  # Only unfreeze last 2 transformer blocks
            )
            # Recreate optimizer with unfrozen parameters
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CONFIG['LR'],
                weight_decay=CONFIG['WEIGHT_DECAY']
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CONFIG['EPOCHS'] - CONFIG['FREEZE_EPOCHS']
            )
        
        # Train with gradient accumulation, mixed precision, and advanced augmentation
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            CONFIG['DEVICE'], CONFIG['ACCUMULATION_STEPS'],
            scaler=scaler,
            gradient_clip_norm=CONFIG['GRADIENT_CLIP_NORM'],
            use_mixup=CONFIG['USE_MIXUP'],
            use_cutmix=CONFIG['USE_CUTMIX'],
            mixup_alpha=CONFIG['MIXUP_ALPHA'],
            cutmix_alpha=CONFIG['CUTMIX_ALPHA'],
            augment_prob=CONFIG['AUGMENT_PROB']
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['DEVICE'])
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"LR:    {current_lr:.6f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{CONFIG['OUTPUT_DIR']}/best_model.pth")
            print(f"✓ New best: {val_acc:.2f}%")
        
        # Save checkpoint
        if epoch % CONFIG['SAVE_FREQ'] == 0:
            torch.save(model.state_dict(), f"{CONFIG['OUTPUT_DIR']}/epoch_{epoch}.pth")
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
            print(f"No improvement for {CONFIG['EARLY_STOP_PATIENCE']} epochs")
            break
    
    # Save final model
    torch.save(model.state_dict(), f"{CONFIG['OUTPUT_DIR']}/last_model.pth")
    
    # Save history
    with open(f"{CONFIG['OUTPUT_DIR']}/log.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(f"\nModels saved:")
    print(f"  {CONFIG['OUTPUT_DIR']}/best_model.pth")
    print(f"  {CONFIG['OUTPUT_DIR']}/last_model.pth")


if __name__ == '__main__':
    main()
