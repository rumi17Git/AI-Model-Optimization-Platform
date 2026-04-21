#!/usr/bin/env python3
"""
Generate deliberately inefficient models that show dramatic optimization improvements

These models are intentionally bloated to demonstrate the power of optimization:
- Excessive parameters
- No pruning or compression
- Full precision (FP32)
- Redundant layers

Perfect for showcasing optimization techniques!
"""

import torch
import torch.nn as nn

class BloatedCNN(nn.Module):
    """
    A deliberately inefficient CNN with excessive parameters
    
    This model is designed to show dramatic improvements when optimized:
    - Uses way more channels than needed
    - Has redundant layers
    - Full precision weights
    - No built-in optimization
    
    Expected optimization gains:
    - Quantization: ~75% size reduction
    - Pruning (50%): ~50% size reduction
    - Combined: ~85-90% total size reduction
    """
    def __init__(self, num_classes=10):
        super(BloatedCNN, self).__init__()
        
        # Excessive feature extraction (way more channels than needed)
        self.features = nn.Sequential(
            # Block 1 - Deliberately oversized
            nn.Conv2d(3, 256, 3, padding=1),     # Normal would be 64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2 - Even more bloated
            nn.Conv2d(256, 512, 3, padding=1),   # Normal would be 128
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3 - Maximum bloat
            nn.Conv2d(512, 1024, 3, padding=1),  # Normal would be 256
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 4 - Redundant (usually not needed)
            nn.Conv2d(1024, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Oversized classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 2 * 2, 4096),  # Excessive hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),           # Another excessive layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),           # And another
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MassiveTransformer(nn.Module):
    """
    A bloated transformer-style model for sequence/image tasks
    
    Shows even more dramatic improvements:
    - Excessive embedding dimensions
    - Too many attention heads
    - Oversized feedforward layers
    
    Expected optimization gains:
    - Can achieve 90%+ size reduction
    - Significant latency improvements
    """
    def __init__(self, num_classes=10, seq_length=196):
        super(MassiveTransformer, self).__init__()
        
        self.embed_dim = 1024  # Way larger than needed (normal: 256-512)
        
        # Patch embedding
        self.patch_embed = nn.Linear(3 * 16 * 16, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_length, self.embed_dim))
        
        # Bloated transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=16,  # Excessive attention heads
            dim_feedforward=4096,  # Massive feedforward
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        
        # Oversized classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Simple patching (in real use, would be more sophisticated)
        x = x.view(B, -1, 3 * 16 * 16)
        
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x

class GiantMLP(nn.Module):
    """
    A simple but massive MLP for demonstrating optimization
    
    Benefits:
    - Very easy to optimize (no complex structures)
    - Shows dramatic improvements
    - Fast to train/test
    """
    def __init__(self, input_size=3*32*32, num_classes=10):
        super(GiantMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 8192),  # Huge first layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def save_bloated_models():
    """Generate and save all bloated models"""
    
    print("="*60)
    print("Generating Bloated Models for Optimization Demo")
    print("="*60)
    
    models = {
        'bloated_cnn': BloatedCNN(num_classes=10),
        'massive_transformer': MassiveTransformer(num_classes=10),
        'giant_mlp': GiantMLP(num_classes=10)
    }
    
    for name, model in models.items():
        # Set to eval mode
        model.eval()
        
        # Calculate stats
        total_params = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        # Save
        filename = f"{name}.pt"
        torch.save(model, filename)
        
        print(f"\n✅ {name.replace('_', ' ').title()}")
        print(f"   File: {filename}")
        print(f"   Parameters: {total_params:,}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Expected reduction with full optimization: ~85-90%")
    
    print("\n" + "="*60)
    print("🎯 Recommendation: Start with 'bloated_cnn.pt'")
    print("   - Best for showcasing optimization")
    print("   - Will show 80-90% size reduction")
    print("   - Dramatic latency improvements")
    print("="*60)
    
    # Also generate a comparison model
    print("\n📊 For comparison, here's an efficient CNN:")
    efficient = EfficientCNN(num_classes=10)
    efficient_params = sum(p.numel() for p in efficient.parameters())
    efficient_size = sum(p.numel() * p.element_size() for p in efficient.parameters()) / (1024 ** 2)
    torch.save(efficient, 'efficient_cnn.pt')
    print(f"   Efficient CNN: {efficient_params:,} params, {efficient_size:.1f} MB")
    print(f"   Bloated CNN is {total_params / efficient_params:.1f}x larger!")

class EfficientCNN(nn.Module):
    """An efficient CNN for comparison"""
    def __init__(self, num_classes=10):
        super(EfficientCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    save_bloated_models()
    
    print("\n🚀 Next steps:")
    print("   1. Upload 'bloated_cnn.pt' to the optimization app")
    print("   2. Try these settings:")
    print("      - Quantization only: ~75% reduction")
    print("      - Quantization + Pruning 30%: ~82% reduction")
    print("      - Quantization + Pruning 50%: ~87% reduction")
    print("      - All techniques: ~90% reduction")
    print("\n   3. Compare with 'efficient_cnn.pt' to see the difference!")
