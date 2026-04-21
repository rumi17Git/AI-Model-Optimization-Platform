"""Model loading logic - business logic only"""
import torch
import torch.nn as nn
import pickle
import importlib.util
import io

def load_model_flexible(file_path):
    """
    Flexible model loader that handles various PyTorch save formats
    
    Returns: (model, format_type)
    """
    
    # Try standard load first (works for most cases)
    try:
        # Use weights_only=False to allow custom classes
        loaded = torch.load(file_path, map_location='cpu', weights_only=False)
        return _process_loaded_data(loaded)
    except TypeError:
        # PyTorch version doesn't support weights_only parameter
        try:
            loaded = torch.load(file_path, map_location='cpu')
            return _process_loaded_data(loaded)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    except Exception as e:
        # If that fails, try custom unpickler
        try:
            with open(file_path, 'rb') as f:
                loaded = _custom_unpickle(f)
            return _process_loaded_data(loaded)
        except Exception as e2:
            raise Exception(
                f"Could not load model.\n"
                f"Error: {str(e)}\n\n"
                f"Try re-saving with: torch.save(model, 'file.pt')"
            )

def _custom_unpickle(file_handle):
    """Custom unpickler with persistent_load support"""
    
    class CustomUnpickler(pickle.Unpickler):
        def __init__(self, file_handle):
            super().__init__(file_handle)
        
        def persistent_load(self, pid):
            """Handle persistent load instructions"""
            # This handles storage objects that PyTorch uses
            if isinstance(pid, tuple) and len(pid) > 0:
                typename = pid[0]
                if typename == 'storage':
                    # pid is ('storage', storage_type, key, location, size)
                    storage_type, key, location, size = pid[1:]
                    # Return a placeholder - the actual tensor will be reconstructed
                    return None
            return None
        
        def find_class(self, module, name):
            """Handle class lookups"""
            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError):
                # Try to import from torch modules
                if module.startswith('torch'):
                    try:
                        mod = importlib.import_module(module)
                        return getattr(mod, name)
                    except:
                        pass
                # Return a dummy class if not found
                return type(name, (), {'__module__': module})
    
    return CustomUnpickler(file_handle).load()

def _process_loaded_data(loaded):
    """Process loaded data into model"""
    
    # Case 1: Full model
    if isinstance(loaded, nn.Module):
        loaded.eval()
        return loaded, 'full_model'
    
    # Case 2: Dictionary
    elif isinstance(loaded, dict):
        # Check for full model in dict
        if 'model' in loaded and isinstance(loaded.get('model'), nn.Module):
            model = loaded['model']
            model.eval()
            return model, 'checkpoint_model'
        
        # Check for state_dict
        if 'state_dict' in loaded:
            state_dict = loaded['state_dict']
        else:
            # Assume the dict itself is a state_dict
            state_dict = loaded
        
        # Try to load from state dict
        model = _load_from_state_dict(state_dict)
        if model:
            return model, 'state_dict'
        else:
            raise ValueError("Could not reconstruct model from state_dict")
    
    else:
        raise ValueError(f"Unknown format: {type(loaded)}")

def _load_from_state_dict(state_dict):
    """Try to load model from state dict - only known architectures"""
    from modules.architecture_detector import detect_architecture, load_known_architecture
    
    # Get keys - handle both dict and OrderedDict
    try:
        keys = list(state_dict.keys())
    except:
        keys = []
    
    if not keys:
        raise ValueError("State dict is empty or invalid")
    
    # Detect architecture
    arch = detect_architecture(keys)
    
    if arch:
        print(f"Detected architecture: {arch}")
        model = load_known_architecture(arch, state_dict)
        if model:
            model.eval()
            return model
        else:
            raise ValueError(f"Detected {arch} but failed to load")
    
    # No known architecture
    raise ValueError(
        "Unknown architecture. Supported architectures:\n"
        "- ResNet (18, 34, 50, 101, 152)\n"
        "- VGG (11, 13, 16, 19)\n"
        "- MobileNet V2\n"
        "- EfficientNet B0-B7\n"
        "- DenseNet (121, 161, 169, 201)\n\n"
        "Please upload the full model with: torch.save(model, 'file.pt')"
    )

def create_sample_model():
    """Create a sample model for testing"""
    
    class SampleCNN(nn.Module):
        def __init__(self):
            super(SampleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 28 * 28, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 10)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    model = SampleCNN()
    model.eval()
    return model
