"""Architecture detection - business logic only"""
import torch
import torch.nn as nn

def detect_architecture(keys):
    """Detect model architecture from state dict keys"""
    keys_str = ' '.join(keys)
    
    # ResNet detection (most common)
    if 'layer1.0.conv1.weight' in keys or 'layer1.0.conv1' in keys_str:
        # Count layers to determine variant
        if 'layer1.0.downsample' in keys_str or 'layer1.0.downsample.0.weight' in keys:
            # Bottleneck architecture (50, 101, 152)
            if 'layer4.2' in keys_str:
                return 'resnet152'
            elif 'layer4.22' in keys_str:
                return 'resnet101'
            else:
                return 'resnet50'
        else:
            # BasicBlock architecture (18, 34)
            if 'layer4.1' in keys_str and not 'layer4.2' in keys_str:
                return 'resnet34'
            else:
                return 'resnet18'
    
    # Alternative ResNet format
    if 'conv1.weight' in keys and 'layer1' in keys_str and 'layer4' in keys_str and 'fc.weight' in keys:
        if 'layer1.0.downsample' in keys_str:
            return 'resnet50'
        return 'resnet18'
    
    # VGG detection
    if 'features.0.weight' in keys and 'classifier.0.weight' in keys:
        feature_layers = [k for k in keys if k.startswith('features.') and 'weight' in k and not 'bn' in k]
        num_conv = len([k for k in feature_layers if 'weight' in k])
        
        if num_conv >= 16:
            return 'vgg19'
        elif num_conv >= 13:
            return 'vgg16'
        elif num_conv >= 11:
            return 'vgg13'
        else:
            return 'vgg11'
    
    # MobileNet V2 detection
    if 'features.0.0.weight' in keys and 'features.18' in keys_str:
        return 'mobilenet_v2'
    
    # EfficientNet detection
    if '_blocks.0' in keys_str or 'blocks.0' in keys_str:
        if '_bn1' in keys_str or 'bn1' in keys_str:
            # Try to determine which variant
            if '_blocks.6' in keys_str or 'blocks.6' in keys_str:
                return 'efficientnet_b0'
            return 'efficientnet_b0'
    
    # DenseNet detection
    if 'features.denseblock1' in keys_str:
        if 'features.denseblock4.denselayer48' in keys_str:
            return 'densenet201'
        elif 'features.denseblock4.denselayer32' in keys_str:
            return 'densenet169'
        elif 'features.denseblock4.denselayer24' in keys_str:
            return 'densenet161'
        else:
            return 'densenet121'
    
    # Inception V3
    if 'Conv2d_1a_3x3' in keys_str or 'Mixed_5b' in keys_str:
        return 'inception_v3'
    
    return None

def load_known_architecture(arch_name, state_dict):
    """Load a known architecture and apply state dict"""
    try:
        import torchvision.models as models
        
        arch_map = {
            # ResNet family
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
            
            # VGG family
            'vgg11': models.vgg11,
            'vgg13': models.vgg13,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
            'vgg11_bn': models.vgg11_bn,
            'vgg13_bn': models.vgg13_bn,
            'vgg16_bn': models.vgg16_bn,
            'vgg19_bn': models.vgg19_bn,
            
            # MobileNet
            'mobilenet_v2': models.mobilenet_v2,
            
            # EfficientNet
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
            'efficientnet_b4': models.efficientnet_b4,
            'efficientnet_b5': models.efficientnet_b5,
            'efficientnet_b6': models.efficientnet_b6,
            'efficientnet_b7': models.efficientnet_b7,
            
            # DenseNet
            'densenet121': models.densenet121,
            'densenet161': models.densenet161,
            'densenet169': models.densenet169,
            'densenet201': models.densenet201,
            
            # Inception
            'inception_v3': models.inception_v3,
        }
        
        if arch_name in arch_map:
            # Create model
            model = arch_map[arch_name](weights=None)
            
            # Try to load state dict
            try:
                model.load_state_dict(state_dict, strict=True)
                print(f"Successfully loaded {arch_name} with strict=True")
                return model
            except RuntimeError as e:
                # Try non-strict loading
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded {arch_name} with strict=False (some keys may be missing)")
                    return model
                except Exception as e2:
                    print(f"Failed to load {arch_name}: {str(e2)}")
                    return None
        
        return None
        
    except Exception as e:
        print(f"Error loading architecture {arch_name}: {str(e)}")
        return None

def get_supported_architectures():
    """Return list of supported architectures"""
    return [
        'ResNet (18, 34, 50, 101, 152)',
        'VGG (11, 13, 16, 19)',
        'MobileNet V2',
        'EfficientNet (B0-B7)',
        'DenseNet (121, 161, 169, 201)',
        'Inception V3'
    ]
