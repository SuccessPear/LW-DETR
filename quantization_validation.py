"""
Validation and testing script for quantized LW-DETR models.
Verifies that quantization was applied correctly and compare performance.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import json


def validate_quantization_applied(model) -> Dict[str, int]:
    """
    Verify that quantization has been applied to the model.
    
    Args:
        model: Model to validate
    
    Returns:
        Dictionary with counts of quantized layers
    """
    counts = {
        'total_layers': 0,
        'quantized_linear': 0,
        'quantized_act': 0,
        'quantized_attention': 0,
        'quantized_deform_attn': 0,
        'quantized_matmul': 0,
        'quantized_norm': 0,
        'other_layers': 0,
    }
    
    for name, module in model.named_modules():
        counts['total_layers'] += 1
        
        module_type = module.__class__.__name__
        
        if 'QuantizedLinear' in module_type:
            counts['quantized_linear'] += 1
        elif 'QuantizedAct' in module_type:
            counts['quantized_act'] += 1
        elif 'QuantizedMultiheadAttention' in module_type:
            counts['quantized_attention'] += 1
        elif 'QuantizedMSDeformAttn' in module_type or 'QuantizedDeformableAttention' in module_type:
            counts['quantized_deform_attn'] += 1
        elif 'QuantizedMatMul' in module_type:
            counts['quantized_matmul'] += 1
        elif 'QuantizedLayerNorm' in module_type:
            counts['quantized_norm'] += 1
        else:
            counts['other_layers'] += 1
    
    return counts


def check_weight_ranges(model) -> Dict[str, Tuple[float, float]]:
    """
    Check weight ranges in quantized layers.
    For INT8, values should be approximately in [-128, 127] range.
    
    Args:
        model: Quantized model
    
    Returns:
        Dictionary mapping layer names to (min, max) weight values
    """
    weight_ranges = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module, 'weight_scale'):
            # This is a quantized linear layer
            weights = module.weight.data
            weight_ranges[name] = (weights.min().item(), weights.max().item())
    
    return weight_ranges


def test_forward_pass(fp32_model, quant_model, batch_size=2, 
                      img_size=800, device='cuda') -> Tuple[bool, Dict]:
    """
    Test that forward passes work correctly on both models.
    
    Args:
        fp32_model: FP32 reference model
        quant_model: INT8 quantized model
        batch_size: Batch size for test
        img_size: Image size
        device: Device to run on
    
    Returns:
        (success, details) tuple
    """
    fp32_model.eval()
    quant_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    details = {
        'fp32_shape': None,
        'quant_shape': None,
        'shape_match': False,
        'fp32_dtype': None,
        'quant_dtype': None,
        'output_mse': None,
        'output_mae': None,
    }
    
    try:
        with torch.no_grad():
            fp32_out = fp32_model(dummy_input)
            quant_out = quant_model(dummy_input)
        
        # Handle different output formats
        if isinstance(fp32_out, (tuple, list)):
            fp32_out = fp32_out[0]
        if isinstance(quant_out, (tuple, list)):
            quant_out = quant_out[0]
        
        details['fp32_shape'] = tuple(fp32_out.shape)
        details['quant_shape'] = tuple(quant_out.shape)
        details['shape_match'] = fp32_out.shape == quant_out.shape
        details['fp32_dtype'] = str(fp32_out.dtype)
        details['quant_dtype'] = str(quant_out.dtype)
        
        # Compute output differences
        mse = ((fp32_out - quant_out) ** 2).mean().item()
        mae = (fp32_out - quant_out).abs().mean().item()
        
        details['output_mse'] = mse
        details['output_mae'] = mae
        
        success = details['shape_match'] and not torch.isnan(quant_out).any()
        
        return success, details
    
    except Exception as e:
        print(f"Error during forward pass: {e}")
        return False, details


def check_distribution_tracking(model) -> Dict[str, bool]:
    """
    Verify that distribution tracking is enabled and working.
    
    Args:
        model: Quantized model
    
    Returns:
        Dictionary indicating which modules have tracking enabled
    """
    tracking_status = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'distributions'):
            has_data = bool(module.distributions)
            tracking_status[name] = has_data
    
    return tracking_status


def validate_quantization_scales(model, tolerance=1e-5) -> Dict[str, bool]:
    """
    Check that quantization scales are valid (not zero, not NaN).
    
    Args:
        model: Quantized model
        tolerance: Minimum acceptable scale value
    
    Returns:
        Dictionary indicating validity of scales per layer
    """
    scale_validity = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'fc_scaling_factor') or hasattr(module, 'act_scaling_factor'):
            scale = module.fc_scaling_factor if hasattr(module, 'fc_scaling_factor') else module.act_scaling_factor
            
            is_valid = (
                scale.abs().min() > tolerance and  # Not too close to zero
                not torch.isnan(scale).any() and   # No NaN values
                not torch.isinf(scale).any()       # No Inf values
            )
            scale_validity[name] = is_valid
    
    return scale_validity


def compare_params(fp32_model, quant_model) -> Dict:
    """
    Compare model sizes between FP32 and quantized versions.
    
    Args:
        fp32_model: FP32 model
        quant_model: Quantized model
    
    Returns:
        Comparison statistics
    """
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    def get_model_size_mb(model):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
        return param_size + buffer_size
    
    return {
        'fp32_params': count_params(fp32_model),
        'quant_params': count_params(quant_model),
        'fp32_size_mb': get_model_size_mb(fp32_model),
        'quant_size_mb': get_model_size_mb(quant_model),
        'compression_ratio': get_model_size_mb(fp32_model) / get_model_size_mb(quant_model),
    }


def run_full_validation(fp32_model, quant_model, device='cuda',
                       output_file='validation_report.json'):
    """
    Run comprehensive validation of quantized model.
    
    Args:
        fp32_model: FP32 reference model
        quant_model: Quantized model
        device: Device to use
        output_file: File to save validation report
    
    Returns:
        Validation report dictionary
    """
    print("\n" + "="*60)
    print("QUANTIZATION VALIDATION REPORT")
    print("="*60)
    
    report = {}
    
    # 1. Check quantization applied
    print("\n[1/5] Checking quantization was applied...")
    quant_counts = validate_quantization_applied(quant_model)
    report['quantization_layer_counts'] = quant_counts
    
    print(f"✓ Total layers: {quant_counts['total_layers']}")
    print(f"✓ Quantized Linear: {quant_counts['quantized_linear']}")
    print(f"✓ Quantized Attention: {quant_counts['quantized_attention']}")
    print(f"✓ Quantized Deformable Attn: {quant_counts['quantized_deform_attn']}")
    print(f"✓ Quantized Activation: {quant_counts['quantized_act']}")
    print(f"✓ Quantized LayerNorm: {quant_counts['quantized_norm']}")
    
    # 2. Check weight ranges
    print("\n[2/5] Checking weight ranges...")
    weight_ranges = check_weight_ranges(quant_model)
    report['weight_ranges_sample'] = {
        name: (min_val, max_val) 
        for name, (min_val, max_val) in list(weight_ranges.items())[:5]
    }
    
    print(f"✓ Layers with quantized weights: {len(weight_ranges)}")
    for i, (name, (min_val, max_val)) in enumerate(list(weight_ranges.items())[:3]):
        print(f"  {i+1}. {name}: [{min_val:.4f}, {max_val:.4f}]")
    
    # 3. Test forward pass
    print("\n[3/5] Testing forward pass...")
    success, forward_details = test_forward_pass(fp32_model, quant_model, device=device)
    report['forward_pass'] = forward_details
    
    if success:
        print("✓ Forward pass successful")
        print(f"  FP32 output shape: {forward_details['fp32_shape']}")
        print(f"  INT8 output shape: {forward_details['quant_shape']}")
        print(f"  Output MSE: {forward_details['output_mse']:.6f}")
        print(f"  Output MAE: {forward_details['output_mae']:.6f}")
    else:
        print("✗ Forward pass failed")
    
    # 4. Check distribution tracking
    print("\n[4/5] Checking distribution tracking...")
    tracking_status = check_distribution_tracking(quant_model)
    
    if tracking_status:
        print(f"✓ Modules with tracking enabled: {len(tracking_status)}")
        active_tracking = sum(1 for v in tracking_status.values() if v)
        print(f"✓ Modules currently tracking data: {active_tracking}")
        report['tracking_enabled'] = True
    else:
        print("⚠ No distribution tracking found")
        report['tracking_enabled'] = False
    
    # 5. Validate scales
    print("\n[5/5] Validating quantization scales...")
    scale_validity = validate_quantization_scales(quant_model)
    valid_scales = sum(1 for v in scale_validity.values() if v)
    total_scales = len(scale_validity)
    
    print(f"✓ Valid quantization scales: {valid_scales}/{total_scales}")
    report['scale_validity'] = {
        'valid_count': valid_scales,
        'total_count': total_scales,
    }
    
    # Parameter comparison
    print("\n[BONUS] Parameter and Size Comparison...")
    param_comparison = compare_params(fp32_model, quant_model)
    report['size_comparison'] = param_comparison
    
    print(f"✓ FP32 model size: {param_comparison['fp32_size_mb']:.2f} MB")
    print(f"✓ INT8 model size: {param_comparison['quant_size_mb']:.2f} MB")
    print(f"✓ Compression ratio: {param_comparison['compression_ratio']:.2f}x")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    summary = {
        'quantization_applied': quant_counts['quantized_linear'] > 0,
        'forward_pass_works': success,
        'tracking_enabled': report['tracking_enabled'],
        'all_scales_valid': valid_scales == total_scales,
        'overall_status': 'PASS' if (success and quant_counts['quantized_linear'] > 0) else 'FAIL'
    }
    report['summary'] = summary
    
    print(f"Quantization Applied: {'✓ YES' if summary['quantization_applied'] else '✗ NO'}")
    print(f"Forward Pass Works: {'✓ YES' if summary['forward_pass_works'] else '✗ NO'}")
    print(f"Distribution Tracking: {'✓ YES' if summary['tracking_enabled'] else '✗ NO'}")
    print(f"All Scales Valid: {'✓ YES' if summary['all_scales_valid'] else '✗ NO'}")
    print(f"\nOVERALL STATUS: {summary['overall_status']}")
    
    # Save report
    if output_file:
        with open(output_file, 'w') as f:
            # Convert non-serializable items
            report_serializable = {
                k: v for k, v in report.items() 
                if isinstance(v, (dict, list, str, int, float, bool, type(None)))
            }
            json.dump(report_serializable, f, indent=2)
        print(f"\nReport saved to {output_file}")
    
    print("="*60 + "\n")
    
    return report


# Usage example
if __name__ == '__main__':
    """
    Example validation script usage
    """
    print("Example validation:")
    print("  from models.quantization import create_quantized_lwdetr_model")
    print("  from quantization_validation import run_full_validation")
    print()
    print("  fp32_model = load_fp32_model('checkpoint.pt')")
    print("  quant_model = create_quantized_lwdetr_model(fp32_model)")
    print()
    print("  report = run_full_validation(fp32_model, quant_model,")
    print("                              output_file='validation_report.json')")
