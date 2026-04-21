"""
Optimization engine - production-grade business logic
(Includes: quantization, pruning, and knowledge distillation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset
import copy
import logging
import time
from typing import Dict, Optional, Tuple

# Library-safe logger (no basicConfig here)
logger = logging.getLogger(__name__)

# Future-proof quantization import
try:
    import torch.ao.quantization as quant
except ImportError:
    import torch.quantization as quant


class QuantizationEngine:
    """Advanced quantization with multiple strategies"""

    @staticmethod
    def dynamic_quantization(model, backend="fbgemm", layers_to_quantize=None):
        """
        Apply dynamic quantization to model.
        NOTE: Conv2d is NOT supported for dynamic quantization.
        """
        try:
            torch.backends.quantized.engine = backend

            if layers_to_quantize is None:
                layers_to_quantize = {nn.Linear, nn.LSTM, nn.GRU}

            quantized_model = quant.quantize_dynamic(
                model,
                layers_to_quantize,
                dtype=torch.qint8,
            )

            quantized_model.eval()
            logger.info(f"Dynamic quantization applied (backend={backend})")
            return quantized_model

        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            raise RuntimeError(f"Quantization failed: {e}")

    @staticmethod
    def static_quantization(model, calibration_data, backend="fbgemm", fuse_patterns=None):
        """
        Apply static quantization (requires calibration data).
        Module fusion is optional and must be explicitly provided.
        """
        try:
            torch.backends.quantized.engine = backend

            model = copy.deepcopy(model)
            model.eval()
            model.qconfig = quant.get_default_qconfig(backend)

            if fuse_patterns:
                quant.fuse_modules(model, fuse_patterns, inplace=True)
                logger.info("Module fusion applied")

            quant.prepare(model, inplace=True)

            with torch.no_grad():
                model(calibration_data)

            quant.convert(model, inplace=True)

            logger.info("Static quantization applied")
            return model

        except Exception as e:
            logger.error(f"Static quantization failed: {e}")
            raise RuntimeError(f"Static quantization failed: {e}")

    @staticmethod
    def quantization_aware_training(model, backend="fbgemm"):
        """
        Prepare model for quantization-aware training (QAT).
        """
        try:
            torch.backends.quantized.engine = backend

            model.train()
            model.qconfig = quant.get_default_qat_qconfig(backend)
            quant.prepare_qat(model, inplace=True)

            logger.info("Model prepared for QAT")
            return model

        except Exception as e:
            logger.error(f"QAT preparation failed: {e}")
            raise RuntimeError(f"QAT failed: {e}")


class PruningEngine:
    """Advanced pruning with multiple strategies"""

    @staticmethod
    def l1_unstructured_pruning(model, amount=0.5, layer_types=None):
        if not 0.0 <= amount <= 1.0:
            raise ValueError(f"Pruning amount must be between 0 and 1, got {amount}")

        pruned_model = copy.deepcopy(model)
        layer_types = layer_types or (nn.Linear, nn.Conv2d)

        total_params_before = 0
        total_params_after = 0
        pruned_layers = 0

        for module in pruned_model.modules():
            if isinstance(module, layer_types):
                params_before = module.weight.numel()
                total_params_before += params_before

                prune.l1_unstructured(module, "weight", amount=amount)
                prune.remove(module, "weight")

                params_after = torch.count_nonzero(module.weight).item()
                total_params_after += params_after
                pruned_layers += 1

        actual_sparsity = 1.0 - (total_params_after / total_params_before)
        logger.info(f"L1 pruning applied to {pruned_layers} layers")
        logger.info(f"Achieved sparsity: {actual_sparsity * 100:.2f}%")

        pruned_model.eval()
        return pruned_model

    @staticmethod
    def structured_pruning(model, amount=0.3, pruning_dim=0):
        pruned_model = copy.deepcopy(model)

        for module in pruned_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=amount,
                    n=2,
                    dim=pruning_dim,
                )
                prune.remove(module, "weight")

        logger.info(f"Structured pruning applied ({amount * 100:.1f}%)")
        pruned_model.eval()
        return pruned_model

    @staticmethod
    def global_pruning(model, amount=0.5):
        pruned_model = copy.deepcopy(model)

        parameters_to_prune = [
            (m, "weight")
            for m in pruned_model.modules()
            if isinstance(m, (nn.Linear, nn.Conv2d))
        ]

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        logger.info(f"Global pruning applied ({amount * 100:.1f}%)")
        pruned_model.eval()
        return pruned_model

    @staticmethod
    def iterative_pruning(model, target_sparsity=0.9, iterations=5):
        pruned_model = copy.deepcopy(model)
        amount_per_iter = 1 - (1 - target_sparsity) ** (1 / iterations)

        for i in range(iterations):
            logger.info(f"Pruning iteration {i + 1}/{iterations}")

            for module in pruned_model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, "weight", amount=amount_per_iter)
                    prune.remove(module, "weight")

        logger.info(f"Iterative pruning complete (target {target_sparsity * 100:.1f}%)")
        pruned_model.eval()
        return pruned_model


class KnowledgeDistillationEngine:
    """Advanced knowledge distillation with multiple strategies"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str = 'cpu'
    ):
        """
        Initialize KD engine
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            device: 'cpu' or 'cuda'
        """
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.device = device
        
        # Set teacher to eval mode (never trains)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        logger.info(f"KD Engine initialized on {device}")
    
    @staticmethod
    def distillation_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        true_labels: torch.Tensor,
        temperature: float = 3.0,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss
        
        Args:
            student_logits: Student model output (before softmax)
            teacher_logits: Teacher model output (before softmax)
            true_labels: Ground truth labels
            temperature: Softening temperature for distillation
            alpha: Weight for distillation loss (1-alpha for hard loss)
        
        Returns:
            Combined loss
        """
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distillation_loss = F.kl_div(
            soft_student,
            soft_targets,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Hard loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, true_labels)
        
        # Combined loss
        total_loss = alpha * distillation_loss + (1 - alpha) * hard_loss
        
        return total_loss
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        temperature: float = 3.0,
        alpha: float = 0.7
    ) -> Tuple[float, float]:
        """
        Train student for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer for student
            temperature: Distillation temperature
            alpha: Distillation weight
        
        Returns:
            (average_loss, accuracy)
        """
        self.student.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = self.teacher(data)
            
            # Get student predictions
            student_logits = self.student(data)
            
            # Compute loss
            loss = self.distillation_loss(
                student_logits,
                teacher_logits,
                target,
                temperature,
                alpha
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Validate student model
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            (average_loss, accuracy)
        """
        self.student.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.student(data)
                loss = F.cross_entropy(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def distill(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        learning_rate: float = 0.001,
        temperature: float = 3.0,
        alpha: float = 0.7,
        patience: int = 3,
        verbose: bool = True
    ) -> Dict:
        """
        Full distillation training loop
        
        Args:
            train_loader: Training data
            val_loader: Validation data (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate
            temperature: Distillation temperature
            alpha: Distillation loss weight
            patience: Early stopping patience
            verbose: Print progress
        
        Returns:
            Training history dict
        """
        optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience // 2, factor=0.5
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'best_epoch': 0
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        if verbose:
            logger.info(f"Starting distillation: {epochs} epochs, T={temperature}, α={alpha}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, temperature, alpha
            )
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    history['best_val_acc'] = val_acc
                    history['best_epoch'] = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                        f"Time: {time.time()-start_time:.2f}s"
                    )
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                        f"Time: {time.time()-start_time:.2f}s"
                    )
        
        logger.info("Distillation complete")
        return history
    
    @staticmethod
    def create_synthetic_data(
        input_shape: Tuple,
        num_samples: int = 1000,
        num_classes: int = 10
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create synthetic data for demonstration
        
        Args:
            input_shape: Shape of input (e.g., (3, 224, 224))
            num_samples: Number of samples
            num_classes: Number of classes
        
        Returns:
            (train_loader, val_loader)
        """
        # Generate random data
        train_data = torch.randn(num_samples, *input_shape)
        train_labels = torch.randint(0, num_classes, (num_samples,))
        
        val_data = torch.randn(num_samples // 5, *input_shape)
        val_labels = torch.randint(0, num_classes, (num_samples // 5,))
        
        # Create datasets
        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"Created synthetic data: {num_samples} train, {num_samples//5} val samples")
        
        return train_loader, val_loader


# Backward-compatible API

def apply_quantization(model, backend="fbgemm", method="dynamic", **kwargs):
    """
    Apply quantization to model
    
    Args:
        model: PyTorch model
        backend: 'fbgemm' or 'qnnpack'
        method: 'dynamic', 'static', or 'qat'
        **kwargs: Additional arguments for specific methods
    
    Returns:
        Quantized model
    """
    if method == "dynamic":
        return QuantizationEngine.dynamic_quantization(model, backend)
    elif method == "static":
        return QuantizationEngine.static_quantization(model, **kwargs, backend=backend)
    elif method == "qat":
        return QuantizationEngine.quantization_aware_training(model, backend)
    else:
        raise ValueError(f"Unknown quantization method: {method}")


def apply_pruning(model, amount=0.5, method="l1_unstructured"):
    """
    Apply pruning to model
    
    Args:
        model: PyTorch model
        amount: Pruning amount (0.0-1.0)
        method: 'l1_unstructured', 'structured', 'global', or 'iterative'
    
    Returns:
        Pruned model
    """
    if method == "l1_unstructured":
        return PruningEngine.l1_unstructured_pruning(model, amount)
    elif method == "structured":
        return PruningEngine.structured_pruning(model, amount)
    elif method == "global":
        return PruningEngine.global_pruning(model, amount)
    elif method == "iterative":
        return PruningEngine.iterative_pruning(model, amount)
    else:
        raise ValueError(f"Unknown pruning method: {method}")


def apply_knowledge_distillation(
    teacher_model: nn.Module,
    student_model: nn.Module,
    temperature: float = 3.0,
    train_loader: Optional[DataLoader] = None,
    epochs: int = 5,
    device: str = 'cpu',
    use_synthetic: bool = True
) -> nn.Module:
    """
    Apply knowledge distillation
    
    Args:
        teacher_model: Pre-trained teacher
        student_model: Student to train
        temperature: Distillation temperature
        train_loader: Training data (optional)
        epochs: Training epochs
        device: 'cpu' or 'cuda'
        use_synthetic: Use synthetic data if train_loader is None
    
    Returns:
        Trained student model
    """
    try:
        # Initialize KD engine
        kd_engine = KnowledgeDistillationEngine(teacher_model, student_model, device)
        
        # Get or create data
        if train_loader is None:
            if use_synthetic:
                logger.info("No training data provided, using synthetic data for demo")
                
                # Infer input shape from teacher
                try:
                    # Try to get input shape from first layer
                    first_layer = next(teacher_model.children())
                    if isinstance(first_layer, nn.Conv2d):
                        input_shape = (first_layer.in_channels, 32, 32)
                    elif isinstance(first_layer, nn.Linear):
                        input_shape = (first_layer.in_features,)
                    else:
                        input_shape = (3, 224, 224)  # Default
                except:
                    input_shape = (3, 224, 224)
                
                train_loader, val_loader = kd_engine.create_synthetic_data(
                    input_shape=input_shape,
                    num_samples=500
                )
            else:
                logger.warning("No training data - returning untrained student model")
                return student_model
        else:
            val_loader = None
        
        # Perform distillation
        history = kd_engine.distill(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            temperature=temperature,
            verbose=True
        )
        
        logger.info(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
        if val_loader:
            logger.info(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
        
        return kd_engine.student
        
    except Exception as e:
        logger.error(f"Knowledge distillation failed: {e}")
        logger.warning("Returning original student model")
        return student_model


def calculate_compression_ratio(original_size, optimized_size):
    """Calculate compression ratio"""
    if original_size <= 0:
        raise ValueError("original_size must be > 0")
    return (1.0 - optimized_size / original_size) * 100.0


def get_model_sparsity(model) -> Dict[str, float]:
    """
    Calculate sparsity statistics for a model
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with sparsity metrics
    """
    total_params = 0
    zero_params = 0

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            params = module.weight.numel()
            zeros = (module.weight == 0).sum().item()
            total_params += params
            zero_params += zeros

    sparsity = (zero_params / total_params * 100) if total_params > 0 else 0.0

    return {
        "total_parameters": total_params,
        "zero_parameters": zero_params,
        "active_parameters": total_params - zero_params,
        "sparsity_percentage": sparsity,
    }