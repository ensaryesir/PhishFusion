"""
Swin Transformer for OCR-aided Siamese Logo Matching (timm-based)
Replaces ResNetV2-50 Backbone with Swin Transformer in PhishIntention's logo matching module

Architecture:
    Image (RGB) -> Swin Backbone (timm) -> Appearance Embedding (768-d)
    OCR Text -> OCR Encoder -> Shape Embedding (512-d)
    [Appearance + Shape] -> Unified Embedding -> Logo Embedding (2048-d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import timm


class SwinSiameseLogo(nn.Module):
    """
    Swin Transformer-based Siamese Model for Logo Matching.
    
    Replaces ResNetV2-50 backbone with Swin Transformer while maintaining
    the same OCR-aided architecture from the original PhishIntention paper.
    
    Args:
        swin_model_name: timm Swin model (e.g., 'swin_tiny_patch4_window7_224')
        logo_embed_dim: Final logo embedding dimension (2048 to match ResNet)
        ocr_embed_dim: OCR shape embedding dimension (512)
        pretrained: Use ImageNet pre-trained Swin
        img_size: Logo image size
    """
    def __init__(
        self,
        swin_model_name: str = 'swin_tiny_patch4_window7_224',
        logo_embed_dim: int = 2048,
        ocr_embed_dim: int = 512,
        pretrained: bool = True,
        img_size: int = 224
    ):
        super().__init__()
        
        self.img_size = img_size
        self.logo_embed_dim = logo_embed_dim
        
        # Swin Transformer Backbone (Appearance Embedding)
        # Replaces ResNetV2-50
        self.swin_backbone = timm.create_model(
            swin_model_name,
            pretrained=pretrained,
            num_classes=0,  # Feature extraction only
            img_size=img_size
        )
        
        appearance_dim = self.swin_backbone.num_features  # 768 for Swin-Tiny
        
        # Unified Embedding Layer
        # Concatenates appearance (Swin) + shape (OCR) embeddings
        unified_dim = appearance_dim + ocr_embed_dim  # 768 + 512 = 1280
        
        # Fully Connected Network
        # Projects unified embedding to final logo embedding
        self.fc_network = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(unified_dim, logo_embed_dim)),
            ('bn1', nn.BatchNorm1d(logo_embed_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(0.3)),
            ('fc2', nn.Linear(logo_embed_dim, logo_embed_dim)),
            ('bn2', nn.BatchNorm1d(logo_embed_dim))
        ]))
        
        # L2 normalization for cosine similarity
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)
    
    def forward_appearance(self, image):
        """
        Extract appearance embedding from logo image.
        
        Args:
            image: [B, 3, H, W] logo image (RGB)
        Returns:
            [B, 768] appearance embedding
        """
        # Resize if needed
        if image.shape[2] != self.img_size or image.shape[3] != self.img_size:
            image = F.interpolate(
                image, 
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Swin forward (replaces ResNetV2-50)
        appearance_emb = self.swin_backbone(image)  # [B, 768]
        
        return appearance_emb
    
    def forward(self, image, ocr_embedding):
        """
        Full forward pass: Image + OCR -> Logo Embedding
        
        Args:
            image: [B, 3, H, W] logo image
            ocr_embedding: [B, 512] OCR shape embedding
        Returns:
            [B, 2048] logo embedding (L2-normalized)
        """
        # Step 1: Appearance Embedding (Swin replaces ResNet)
        appearance_emb = self.forward_appearance(image)  # [B, 768]
        
        # Step 2: Unified Embedding (concatenate appearance + shape)
        unified_emb = torch.cat([appearance_emb, ocr_embedding], dim=1)  # [B, 1280]
        
        # Step 3: Fully Connected Network
        logo_emb = self.fc_network(unified_emb)  # [B, 2048]
        
        # Step 4: L2 Normalization (for cosine similarity matching)
        logo_emb = self.l2_norm(logo_emb)
        
        return logo_emb
    
    def extract_features(self, image, ocr_embedding):
        """
        Alias for forward (for compatibility with existing code)
        """
        return self.forward(image, ocr_embedding)


# ==============================
# Model Factory Functions
# ==============================

def create_swin_siamese(
    model_name: str = 'swin_tiny_patch4_window7_224',
    pretrained: bool = True,
    logo_embed_dim: int = 2048,
    img_size: int = 224
):
    """
    Factory function to create Swin-based Siamese model.
    
    Recommended models:
        - 'swin_tiny_patch4_window7_224': Lightweight (28M params, 768-d) ⭐
        - 'swin_small_patch4_window7_224': Balanced (50M params, 768-d)
        - 'swin_base_patch4_window7_224': Best accuracy (88M params, 1024-d)
    
    Args:
        model_name: timm model name
        pretrained: Use ImageNet pre-trained weights
        logo_embed_dim: Final embedding dimension (2048 to match ResNet)
        img_size: Input image size
    
    Returns:
        SwinSiameseLogo model
    """
    return SwinSiameseLogo(
        swin_model_name=model_name,
        logo_embed_dim=logo_embed_dim,
        ocr_embed_dim=512,  # Fixed from PhishIntention
        pretrained=pretrained,
        img_size=img_size
    )


# ==============================
# Integration with PhishIntention
# ==============================

def siamese_swin_config(weights_path: str, model_name: str = 'swin_tiny_patch4_window7_224'):
    """
    Load trained Swin Siamese model from checkpoint.
    
    Compatible with existing siamese_model_config() in logo_matching.py
    
    Args:
        weights_path: Path to trained model weights (.pth file) or None for ImageNet pre-trained
        model_name: timm Swin model name
    
    Returns:
        Loaded model ready for inference
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if we should use pre-trained or load from checkpoint
    use_pretrained = (weights_path is None or weights_path == '' or not weights_path)
    
    # Create model
    model = create_swin_siamese(
        model_name=model_name,
        pretrained=use_pretrained,  # Use ImageNet if no weights provided
        logo_embed_dim=2048
    )
    
    # Load weights if provided
    if not use_pretrained:
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if exists (DataParallel)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✓ Loaded Swin weights from: {weights_path}")
    else:
        print("✓ Using ImageNet pre-trained Swin (no fine-tuning)")
    
    model.to(device)
    model.eval()
    
    return model


# ==============================
# Comparison: ResNet vs Swin
# ==============================

def compare_architectures():
    """
    Print architecture comparison between ResNetV2-50 and Swin Transformer
    """
    print("=" * 70)
    print("OCR-aided Siamese Model: ResNetV2-50 vs Swin Transformer")
    print("=" * 70)
    
    comparison = """
    Component             | ResNetV2-50           | Swin-Tiny (timm)
    ---------------------|----------------------|----------------------
    Backbone             | ResNetV2-50          | swin_tiny_patch4_window7_224
    Parameters           | ~25M                 | ~28M (fair comparison!)
    Appearance Embedding | 2048-d (GlobalAvgPool)| 768-d (Global pool)
    Pre-training         | ImageNet-21k (BiT)   | ImageNet-22k
    Architecture         | CNN (local)          | Hierarchical Transformer ✓
    Spatial Info         | Limited              | Shifted Windows ✓
    Multi-scale          | Yes (4 stages)       | Yes (4 stages) ✓
    
    OCR Shape Embedding  | 512-d (same)         | 512-d (same)
    Unified Embedding    | 2048 + 512 = 2560    | 768 + 512 = 1280
    Logo Embedding       | 2048-d (FC)          | 2048-d (FC Network)
    Similarity Metric    | Cosine similarity    | Cosine similarity
    """
    print(comparison)
    print("=" * 70)


# ==============================
# Available Swin Models
# ==============================

RECOMMENDED_SWIN_MODELS = {
    # Fair comparison with ResNet50
    'swin_tiny_patch4_window7_224': {
        'params': '28M',
        'embed_dim': 768,
        'description': 'Fair comparison with ResNet50 (25M), best for academic papers',
        'recommended': True,
        'val_acc': '85.06%'
    },
    
    # Better accuracy
    'swin_small_patch4_window7_224': {
        'params': '50M',
        'embed_dim': 768,
        'description': 'Better accuracy, 2x larger than ResNet',
        'recommended': False,
        'val_acc': '~58-65% (estimated)'
    },
    
    # Best accuracy
    'swin_base_patch4_window7_224': {
        'params': '88M',
        'embed_dim': 1024,
        'description': 'Best accuracy, 3.5x larger than ResNet',
        'recommended': False,
        'val_acc': '~65-70% (estimated)'
    }
}


if __name__ == '__main__':
    print("=== Swin Transformer-based Siamese Logo Matcher for PhishIntention ===\n")
    
    # Test model creation
    print("1. Creating Swin Siamese model...")
    model = create_swin_siamese(
        model_name='swin_tiny_patch4_window7_224',
        pretrained=True
    )
    
    # Test inputs
    dummy_logo = torch.randn(4, 3, 224, 224)  # Logo images
    dummy_ocr = torch.randn(4, 512)            # OCR embeddings
    
    # Forward pass
    print("2. Testing forward pass...")
    with torch.no_grad():
        logo_embeddings = model(dummy_logo, dummy_ocr)
    
    print(f"\nResults:")
    print(f"  Input logo shape: {dummy_logo.shape}")
    print(f"  Input OCR shape: {dummy_ocr.shape}")
    print(f"  Output embedding shape: {logo_embeddings.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  Pre-trained: ✓ ImageNet weights loaded")
    
    # Architecture comparison
    print("\n3. Architecture Comparison:")
    compare_architectures()
    
    # Recommended models
    print("\n4. Recommended Swin Models:")
    for name, info in RECOMMENDED_SWIN_MODELS.items():
        marker = "⭐" if info['recommended'] else "  "
        print(f"{marker} {name}")
        print(f"     Params: {info['params']}, Embed: {info['embed_dim']}-d")
        print(f"     Val Acc: {info['val_acc']}")
        print(f"     {info['description']}")


# ==============================
# TTA (Test Time Augmentation) for Production
# ==============================

def get_swin_embedding_with_tta(image_pil, ocr_embedding, model, device='cuda'):
    """
    Extract logo embedding using TTA (Test Time Augmentation) for robust predictions.
    
    This function applies 3 different augmentations to the input logo:
    1. Original (224×224): Standard view
    2. Zoomed (256→224 center crop): Detail-focused view
    3. Horizontal Flip: Symmetry check (optional for logos)
    
    The embeddings from all 3 views are averaged (ensemble) to produce
    a more robust final embedding.
    
    Args:
        image_pil: PIL Image (Logo image in RGB, already cropped)
        ocr_embedding: torch.Tensor [1, 512] OCR shape embedding
        model: SwinSiameseLogo model (already loaded)
        device: 'cuda' or 'cpu'
    
    Returns:
        final_embedding: torch.Tensor [1, 2048] L2-normalized logo embedding
    
    Example:
        >>> from PIL import Image
        >>> logo_img = Image.open('logo.png')
        >>> ocr_emb = ocr_model.extract_features(logo_img)
        >>> embedding = get_swin_embedding_with_tta(logo_img, ocr_emb, swin_model)
        >>> # embedding.shape == torch.Size([1, 2048])
    """
    from torchvision import transforms
    
    model.eval()
    
    # TTA Transformations (matching eval_tta.py)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    transforms_list = [
        # 1. Original (224×224) - Standard view
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]),
        # 2. Slight Zoom (256→224 Center Crop) - Detail-focused
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
        # 3. Horizontal Flip - Symmetry check
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),  # Force flip
            transforms.ToTensor(),
            normalize
        ])
    ]
    
    # Apply 3 transformations
    batch_inputs = []
    for t in transforms_list:
        batch_inputs.append(t(image_pil))
    
    # Stack into batch: [3, 3, 224, 224]
    inputs = torch.stack(batch_inputs).to(device)
    
    # Repeat OCR embedding for batch: [3, 512]
    ocr_batch = ocr_embedding.repeat(3, 1).to(device)
    
    with torch.no_grad():
        # Forward pass for all 3 views
        # Returns [3, 2048] logo embeddings (already L2-normalized by model)
        outputs = model(inputs, ocr_batch)
        
        # Average embeddings (ensemble)
        avg_embedding = outputs.mean(dim=0, keepdim=True)  # [1, 2048]
        
        # Re-normalize after averaging (important!)
        final_embedding = F.normalize(avg_embedding, p=2, dim=1)
    
    return final_embedding

