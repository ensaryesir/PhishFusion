"""
Vision Transformer for OCR-aided Siamese Logo Matching (timm-based)
Replaces ResNetV2-50 Backbone with ViT in PhishIntention's logo matching module

Architecture:
    Image (RGB) -> ViT Backbone (timm) -> Appearance Embedding (768-d)
    OCR Text -> OCR Encoder -> Shape Embedding (512-d)
    [Appearance + Shape] -> Unified Embedding -> Logo Embedding (2048-d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import timm


class ViTSiameseLogo(nn.Module):
    """
    ViT-based Siamese Model for Logo Matching.
    
    Replaces ResNetV2-50 backbone with Vision Transformer while maintaining
    the same OCR-aided architecture from the original PhishIntention paper.
    
    Args:
        vit_model_name: timm ViT model (e.g., 'vit_base_patch16_224')
        logo_embed_dim: Final logo embedding dimension (2048 to match ResNet)
        ocr_embed_dim: OCR shape embedding dimension (512)
        pretrained: Use ImageNet pre-trained ViT
        img_size: Logo image size
    """
    def __init__(
        self,
        vit_model_name: str = 'vit_base_patch16_224',
        logo_embed_dim: int = 2048,
        ocr_embed_dim: int = 512,
        pretrained: bool = True,
        img_size: int = 224
    ):
        super().__init__()
        
        self.img_size = img_size
        self.logo_embed_dim = logo_embed_dim
        
        # ViT Backbone (Appearance Embedding)
        # Replaces ResNetV2-50
        self.vit_backbone = timm.create_model(
            vit_model_name,
            pretrained=pretrained,
            num_classes=0,  # Feature extraction only
            img_size=img_size
        )
        
        appearance_dim = self.vit_backbone.num_features  # 768 for ViT-Base
        
        # Unified Embedding Layer
        # Concatenates appearance (ViT) + shape (OCR) embeddings
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
        
        # ViT forward (replaces ResNetV2-50)
        appearance_emb = self.vit_backbone(image)  # [B, 768]
        
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
        # Step 1: Appearance Embedding (ViT replaces ResNet)
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

def create_vit_siamese(
    model_name: str = 'vit_base_patch16_224',
    pretrained: bool = True,
    logo_embed_dim: int = 2048,
    img_size: int = 224
):
    """
    Factory function to create ViT-based Siamese model.
    
    Recommended models:
        - 'vit_base_patch16_224': Standard (86M params, 768-d features)
        - 'vit_small_patch16_224': Lightweight (22M params, 384-d features)
        - 'vit_large_patch16_224': Large (307M params, 1024-d features)
        - 'deit_base_distilled_patch16_224': Distilled ViT (better for small datasets)
    
    Args:
        model_name: timm model name
        pretrained: Use ImageNet pre-trained weights
        logo_embed_dim: Final embedding dimension (2048 to match ResNet)
        img_size: Input image size
    
    Returns:
        ViTSiameseLogo model
    """
    return ViTSiameseLogo(
        vit_model_name=model_name,
        logo_embed_dim=logo_embed_dim,
        ocr_embed_dim=512,  # Fixed from PhishIntention
        pretrained=pretrained,
        img_size=img_size
    )


# ==============================
# Integration with PhishIntention
# ==============================

def siamese_vit_config(weights_path: str, model_name: str = 'vit_base_patch16_224'):
    """
    Load trained ViT Siamese model from checkpoint.
    
    Compatible with existing siamese_model_config() in logo_matching.py
    
    Args:
        weights_path: Path to trained model weights (.pth file) or None for ImageNet pre-trained
        model_name: timm ViT model name
    
    Returns:
        Loaded model ready for inference
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if we should use pre-trained or load from checkpoint
    use_pretrained = (weights_path is None or weights_path == '' or not weights_path)
    
    # Create model
    model = create_vit_siamese(
        model_name=model_name,
        pretrained=use_pretrained,  # Use ImageNet if no weights provided
        logo_embed_dim=2048
    )
    
    # Load weights if provided
    if not use_pretrained:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Remove 'module.' prefix if exists (DataParallel)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        print(f"✓ Loaded ViT weights from: {weights_path}")
    else:
        print("✓ Using ImageNet pre-trained ViT (no fine-tuning)")
    
    model.to(device)
    model.eval()
    
    return model


# ==============================
# Comparison: ResNet vs ViT
# ==============================

def compare_architectures():
    """
    Print architecture comparison between ResNetV2-50 and ViT
    """
    print("=" * 70)
    print("OCR-aided Siamese Model: ResNetV2-50 vs Vision Transformer")
    print("=" * 70)
    
    comparison = """
    Component             | ResNetV2-50           | ViT-Base (timm)
    ---------------------|----------------------|----------------------
    Backbone             | ResNetV2-50          | vit_base_patch16_224
    Parameters           | ~25M                 | ~86M
    Appearance Embedding | 2048-d (GlobalAvgPool)| 768-d (CLS token)
    Pre-training         | ImageNet-21k (BiT)   | ImageNet-21k
    Receptive Field      | Hierarchical (local) | Global (attention)
    Long-range Deps      | Limited              | Excellent ✓
    
    OCR Shape Embedding  | 512-d (same)         | 512-d (same)
    Unified Embedding    | 2048 + 512 = 2560    | 768 + 512 = 1280
    Logo Embedding       | 2048-d (FC)          | 2048-d (FC Network)
    Similarity Metric    | Cosine similarity    | Cosine similarity
    """
    print(comparison)
    print("=" * 70)


# ==============================
# Available ViT Models
# ==============================

RECOMMENDED_VIT_MODELS = {
    # Balanced performance
    'vit_base_patch16_224': {
        'params': '86M',
        'embed_dim': 768,
        'description': 'Standard ViT-Base, best for most cases',
        'recommended': True
    },
    
    # Lightweight
    'vit_small_patch16_224': {
        'params': '22M',
        'embed_dim': 384,
        'description': 'Lightweight, faster inference',
        'recommended': False
    },
    
    # Large capacity
    'vit_large_patch16_224': {
        'params': '307M',
        'embed_dim': 1024,
        'description': 'Best accuracy, slower',
        'recommended': False
    },
    
    # Data-efficient (distilled)
    'deit_base_distilled_patch16_224': {
        'params': '87M',
        'embed_dim': 768,
        'description': 'DeiT with knowledge distillation',
        'recommended': True
    }
}


if __name__ == '__main__':
    print("=== ViT-based Siamese Logo Matcher for PhishIntention ===\n")
    
    # Test model creation
    print("1. Creating ViT Siamese model...")
    model = create_vit_siamese(
        model_name='vit_base_patch16_224',
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
    print("\n4. Recommended ViT Models:")
    for name, info in RECOMMENDED_VIT_MODELS.items():
        marker = "⭐" if info['recommended'] else "  "
        print(f"{marker} {name}")
        print(f"     Params: {info['params']}, Embed: {info['embed_dim']}-d")
        print(f"     {info['description']}")
