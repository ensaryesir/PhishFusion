"""
TTA (Test-Time Augmentation) Evaluation
Evaluates model with 5-crop augmentation to boost accuracy
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from collections import OrderedDict
import os
import json
from datetime import datetime

# ===================== CONFIG =====================
CONFIG = {
    # Model to test
    'MODEL_PATH': 'checkpoints_swin_tiny/best_model.pth',
    'VIT_MODEL': 'swin_tiny_patch4_window7_224',
    'BEST_EPOCH': 39,  # From training log
    
    # Dataset
    'TEST_DIR': '../datasets/OCR_aided_Siamese_model/3061_training_set/logos',
    'NUM_CLASSES': 277,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'VAL_SPLIT': 0.2,
    'BATCH_SIZE': 1,
    
    # Output
    'OUTPUT_LOG': 'checkpoints_swin_tiny/tta_results.json',
}

# ===================== MODEL (Same as training) =====================
class ImprovedViTSiamese(nn.Module):
    """Same architecture as training script"""
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=277, pretrained=False):
        super().__init__()
        
        # Load ViT/Swin backbone
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            img_size=224
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.vit.num_features, num_classes)
        )
    
    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)


# ===================== DATASET =====================
class LogoDataset(Dataset):
    """Simple dataset for validation - MUST match training script exactly"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        brand_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for idx, brand_dir in enumerate(brand_dirs):
            # CRITICAL: Match training script exactly - only .png and .jpg (no .jpeg)
            image_files = list(brand_dir.glob('*.png')) + list(brand_dir.glob('*.jpg'))
            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback for corrupted images
            image = Image.new('RGB', (224, 224), (255, 255, 255))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ===================== MULTI-SCALE TTA EVALUATION =====================
def evaluate_with_tta(model, val_dataset, device):
    """
    Evaluate model with Multi-Scale TTA
    - Original size (224x224)
    - Zoomed view (256→224 center crop)
    - Horizontal flip
    Total: 3 augmentations (better for logos than cropping)
    """
    print(f"\n{'='*70}")
    print("TTA (Multi-Scale) EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {CONFIG['MODEL_PATH']}")
    print(f"Val Samples: {len(val_dataset)}")
    print(f"Device: {device}")
    print(f"\nStrategy: Original + Zoom + Horizontal Flip (3 views)")
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # Multi-Scale Transforms (better for logos than cropping)
    transforms_list = [
        # 1. Original (224x224) - Standard view
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]),
        # 2. Slight Zoom (256 → 224 Center Crop) - Zoomed view
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
        # 3. Horizontal Flip - Mirror view (some logos are symmetric)
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),  # Force flip
            transforms.ToTensor(),
            normalize
        ]),
    ]
    
    model.eval()
    
    # Track per-augmentation accuracy
    correct_per_aug = [0, 0, 0]  # [original, zoomed, flipped]
    correct_tta = 0
    total = 0
    
    print(f"\nProcessing {len(val_dataset)} validation images...")
    
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc="Multi-Scale TTA"):
            # Get image path and label from Subset
            actual_idx = val_dataset.indices[i]
            img_path = val_dataset.dataset.images[actual_idx]
            label = val_dataset.dataset.labels[actual_idx]
            
            # Load image
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                continue
            
            # Apply 3 different transformations
            batch_inputs = []
            for transform in transforms_list:
                batch_inputs.append(transform(img))
            
            # Stack into batch: [3, 3, 224, 224]
            inputs = torch.stack(batch_inputs).to(device)
            
            # Get predictions for all 3 views
            # Get predictions for all 3 views
            outputs = model(inputs)  # [3, 277]
            
            # Per-augmentation predictions
            for aug_idx in range(3):
                pred = outputs[aug_idx].argmax().item()
                if pred == label:
                    correct_per_aug[aug_idx] += 1
            
            # Use softmax probabilities for better averaging
            probs = F.softmax(outputs, dim=1)
            
            # Average probabilities across all views (TTA ensemble)
            avg_prob = probs.mean(dim=0)  # [277]
            pred_tta = avg_prob.argmax().item()
            
            if pred_tta == label:
                correct_tta += 1
            total += 1
    
    # Calculate accuracies
    acc_original = 100.0 * correct_per_aug[0] / total
    acc_zoomed = 100.0 * correct_per_aug[1] / total
    acc_flipped = 100.0 * correct_per_aug[2] / total
    acc_tta = 100.0 * correct_tta / total
    
    results = {
        'combined_accuracy': round(acc_tta, 2),
        'original_accuracy': round(acc_original, 2),
        'zoomed_accuracy': round(acc_zoomed, 2),
        'flipped_accuracy': round(acc_flipped, 2),
        'num_samples': total
    }
    
    return results


# ===================== SAVE RESULTS =====================
def save_results_to_json(tta_results, output_path):
    """Save TTA results to JSON log file"""
    
    # Baseline results from training (epoch 39)
    baseline_results = {
        "val_accuracy": 85.06,
        "train_accuracy": 99.88,
        "val_loss": 1.6955,
        "train_loss": 0.9567,
        "num_samples": tta_results['num_samples']
    }
    
    # Calculate improvement
    improvement = round(tta_results['combined_accuracy'] - baseline_results['val_accuracy'], 2)
    
    # Comprehensive results
    results = {
        "model": CONFIG['VIT_MODEL'],
        "checkpoint": CONFIG['MODEL_PATH'],
        "best_epoch": CONFIG['BEST_EPOCH'],
        "timestamp": datetime.now().isoformat(),
        "device": CONFIG['DEVICE'],
        
        "baseline_results": baseline_results,
        
        "tta_results": {
            "combined_accuracy": tta_results['combined_accuracy'],
            "improvement": improvement,
            "num_augmentations": 3,
            "per_augmentation": {
                "original": tta_results['original_accuracy'],
                "zoomed": tta_results['zoomed_accuracy'],
                "horizontal_flip": tta_results['flipped_accuracy']
            }
        },
        
        "summary": {
            "baseline_vs_tta": f"{baseline_results['val_accuracy']}% → {tta_results['combined_accuracy']}%",
            "absolute_improvement": f"+{improvement}%",
            "relative_improvement": f"+{round(100 * improvement / baseline_results['val_accuracy'], 2)}%"
        }
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    return results


# ===================== MAIN =====================
def main():
    print(f"\n{'='*70}")
    print("TTA EVALUATION WITH JSON LOGGING")
    print(f"{'='*70}")
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {CONFIG['TEST_DIR']}")
    full_dataset = LogoDataset(CONFIG['TEST_DIR'])
    print(f"✓ Total images: {len(full_dataset)}")
    
    # Split into train/val (same as training - seed 42)
    val_size = int(len(full_dataset) * CONFIG['VAL_SPLIT'])
    train_size = len(full_dataset) - val_size
    
    _, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Validation images: {len(val_dataset)}")
    
    # Load model
    print(f"\n🤖 Loading model: {CONFIG['VIT_MODEL']}")
    model = ImprovedViTSiamese(
        model_name=CONFIG['VIT_MODEL'],
        num_classes=CONFIG['NUM_CLASSES'],
        pretrained=False
    )
    
    # Load checkpoint
    print(f"📦 Loading checkpoint: {CONFIG['MODEL_PATH']}")
    checkpoint = torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'])
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if exists
    clean_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_dict[name] = v
    
    model.load_state_dict(clean_dict, strict=False)
    model.to(CONFIG['DEVICE'])
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Evaluate with TTA
    tta_results = evaluate_with_tta(model, val_dataset, CONFIG['DEVICE'])
    
    # Save results to JSON
    final_results = save_results_to_json(tta_results, CONFIG['OUTPUT_LOG'])
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"📊 Baseline Accuracy:    85.06%")
    print(f"🚀 TTA Accuracy:         {tta_results['combined_accuracy']:.2f}%")
    print(f"📈 Improvement:          {final_results['tta_results']['improvement']:+.2f}%")
    print(f"\n📋 Per-Augmentation Breakdown:")
    print(f"   - Original (224×224):     {tta_results['original_accuracy']:.2f}%")
    print(f"   - Zoomed (256→224):       {tta_results['zoomed_accuracy']:.2f}%")
    print(f"   - Horizontal Flip:        {tta_results['flipped_accuracy']:.2f}%")
    print(f"   - Combined (TTA):         {tta_results['combined_accuracy']:.2f}%")
    print(f"\n📁 Results saved to: {CONFIG['OUTPUT_LOG']}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
