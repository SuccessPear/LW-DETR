"""
Model conversion utilities for quantizing LW-DETR from FP32 to INT8.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import copy


def convert_module_to_quantized(module, track_distributions=False):
    """
    Convert standard PyTorch modules to quantized versions.
    """
    from .quant_modules import (
        QuantizedLinear, QuantizedAct, QuantizedLayerNorm,
        QuantizedReLU, QuantizedGELU
    )
    
    if isinstance(module, nn.Linear):
        # Convert Linear to QuantizedLinear
        q_module = QuantizedLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            track_distributions=track_distributions
        )
        # Copy weights and bias if they exist
        if module.weight is not None:
            q_module.weight.data = module.weight.data.clone()
        if module.bias is not None:
            q_module.bias.data = module.bias.data.clone()
        return q_module
    
    elif isinstance(module, nn.LayerNorm):
        # Convert LayerNorm to QuantizedLayerNorm
        q_module = QuantizedLayerNorm(
            module.normalized_shape if isinstance(module.normalized_shape, int) 
            else module.normalized_shape[0],
            eps=module.eps,
            track_distributions=track_distributions
        )
        q_module.weight.data = module.weight.data.clone()
        q_module.bias.data = module.bias.data.clone()
        return q_module
    
    elif isinstance(module, nn.ReLU):
        return QuantizedReLU(track_distributions=track_distributions)
    
    elif isinstance(module, nn.GELU):
        return QuantizedGELU(track_distributions=track_distributions)
    
    return None


def convert_transformer_to_quantized(model, track_distributions=True):
    """
    Convert a standard Transformer model to quantized version.
    Recursively converts all eligible layers while preserving model structure.
    
    Args:
        model: Original FP32 model
        track_distributions: Whether to track distributions for benchmarking
    
    Returns:
        Quantized model with INT8 support
    """
    model_copy = copy.deepcopy(model)
    
    for name, module in model_copy.named_modules():
        # Skip if it's the module itself
        if not isinstance(module, (nn.Linear, nn.LayerNorm, nn.ReLU, nn.GELU)):
            # Recursively process child modules
            for child_name, child_module in module.named_children():
                q_module = convert_module_to_quantized(child_module, track_distributions)
                if q_module is not None:
                    setattr(module, child_name, q_module)
    
    return model_copy


def replace_attention_with_quantized(model, use_deformable=True, track_distributions=True):
    """
    Replace attention modules with quantized versions.
    
    Args:
        model: Original model
        use_deformable: Whether to use quantized deformable attention
        track_distributions: Whether to track distributions
    
    Returns:
        Model with quantized attention
    """
    from .quant_transformer import (
        QuantizedMultiheadAttention, QuantizedDecoderLayer, QuantizedTransformerEncoderLayer
    )
    from .quant_deformable_attn import QuantizedMSDeformAttn
    
    for name, module in model.named_modules():
        if 'self_attn' in name or 'attention' in name.lower():
            # Try to replace with quantized version
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]
            
            try:
                parent = model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                
                # Create quantized replacement
                if hasattr(module, 'num_heads'):
                    embed_dim = module.embed_dim if hasattr(module, 'embed_dim') else 256
                    num_heads = module.num_heads
                    dropout = module.dropout if hasattr(module, 'dropout') else 0.0
                    
                    q_attn = QuantizedMultiheadAttention(
                        embed_dim, num_heads, dropout=dropout,
                        track_distributions=track_distributions
                    )
                    setattr(parent, module_name, q_attn)
            except:
                pass
    
    return model


def initialize_quantization_ranges(model, dataloader, num_batches=10, device='cuda'):
    """
    Initialize quantization ranges by running through sample data.
    This sets the min/max values for each quantization layer.
    
    Args:
        model: Quantized model
        dataloader: DataLoader with sample inputs
        num_batches: Number of batches to process
        device: Device to run on
    
    Returns:
        Model with initialized quantization ranges
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # Move to device
            if isinstance(images, list):
                images = [img.to(device) for img in images]
            else:
                images = images.to(device)
            
            if isinstance(targets, list):
                targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]
            else:
                targets = targets.to(device)
            
            # Forward pass to collect statistics
            try:
                _ = model(images, targets)
            except:
                # Adapted forward for inference
                try:
                    _ = model(images)
                except:
                    pass
    
    return model


def create_quantized_lwdetr_model(original_model, track_distributions=True, device='cuda'):
    """
    Create a quantized version of LW-DETR from a trained FP32 model.
    This is the main entry point for model quantization.
    
    Args:
        original_model: Trained FP32 LW-DETR model
        track_distributions: Whether to track activation distributions
        device: Device to use
    
    Returns:
        Quantized INT8 LW-DETR model
    """
    # Create a deep copy
    quant_model = copy.deepcopy(original_model)
    
    # Replace linear layers with quantized versions
    for name, module in quant_model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]
            
            if parent_name:
                parent = quant_model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                
                from .quant_modules import QuantizedLinear
                q_linear = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    track_distributions=track_distributions
                )
                q_linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    q_linear.bias.data = module.bias.data.clone()
                
                setattr(parent, module_name, q_linear)
    
    # Replace layer norms with quantized versions
    for name, module in quant_model.named_modules():
        if isinstance(module, nn.LayerNorm):
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]
            
            if parent_name:
                parent = quant_model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                
                from .quant_modules import QuantizedLayerNorm
                q_norm = QuantizedLayerNorm(
                    module.normalized_shape[0] if isinstance(module.normalized_shape, (tuple, list))
                    else module.normalized_shape,
                    eps=module.eps,
                    track_distributions=track_distributions
                )
                q_norm.weight.data = module.weight.data.clone()
                q_norm.bias.data = module.bias.data.clone()
                
                setattr(parent, module_name, q_norm)
    
    return quant_model.to(device)


def save_quantized_model(model, save_path, include_distributions=True):
    """
    Save quantized model with optional distribution statistics.
    
    Args:
        model: Quantized model
        save_path: Path to save model
        include_distributions: Whether to save distribution data
    """
    state_dict = {
        'model': model.state_dict(),
        'config': model.config if hasattr(model, 'config') else None,
    }
    
    if include_distributions:
        distributions = {}
        for name, module in model.named_modules():
            if hasattr(module, 'distributions'):
                distributions[name] = module.distributions
        state_dict['distributions'] = distributions
    
    torch.save(state_dict, save_path)
    print(f"Quantized model saved to {save_path}")


def load_quantized_model(checkpoint_path, model_class, device='cuda'):
    """
    Load a quantized model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model_class: Model class to instantiate
        device: Device to load on
    
    Returns:
        Loaded quantized model and distributions if available
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model and load state dict
    model = model_class()
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    
    # Load distributions if available
    distributions = checkpoint.get('distributions', None)
    
    return model, distributions
