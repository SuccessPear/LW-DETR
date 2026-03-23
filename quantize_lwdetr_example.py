"""
Example script demonstrating how to quantize LW-DETR models to INT8 with distribution tracking.

Usage:
    python quantize_lwdetr_example.py --config <config_file> --checkpoint <checkpoint_path> \\
                                      --output <output_dir> --benchmark

The script:
1. Loads a trained FP32 LW-DETR model
2. Converts it to INT8 quantized version
3. Initializes quantization ranges with sample data
4. Benchmarks and compares FP32 vs INT8 performance
5. Generates distribution visualization plots
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import sys

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from LW-DETR
from models.quantization import (
    create_quantized_lwdetr_model,
    initialize_quantization_ranges,
    save_quantized_model,
    QuantizationBenchmark,
)


def load_fp32_model(checkpoint_path: str, device='cuda') -> nn.Module:
    """
    Load a trained FP32 LW-DETR model from checkpoint.
    
    Args:
        checkpoint_path: Path to the FP32 checkpoint
        device: Device to load on
    
    Returns:
        Loaded FP32 model
    """
    print(f"Loading FP32 model from {checkpoint_path}...")
    
    # Try to load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model weights
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        model_state = checkpoint
    
    # Create model instance
    # Note: You need to modify this to match your LW-DETR model initialization
    from models.lwdetr import build
    
    # Assuming you have a config or args object
    class Args:
        # Fill these based on your model architecture
        d_model = 256
        nhead = 8
        num_decoder_layers = 6
        dim_feedforward = 2048
        dropout = 0.1
    
    args = Args()
    fp32_model = build(args)
    fp32_model.load_state_dict(model_state, strict=False)
    fp32_model = fp32_model.to(device)
    
    print("FP32 model loaded successfully!")
    return fp32_model


def create_quantized_model(fp32_model: nn.Module, track_distributions=True,
                          device='cuda') -> nn.Module:
    """
    Convert FP32 model to INT8 quantized model.
    
    Args:
        fp32_model: Original FP32 model
        track_distributions: Whether to track activation distributions
        device: Device to use
    
    Returns:
        Quantized INT8 model
    """
    print("\nCreating quantized INT8 model...")
    
    quant_model = create_quantized_lwdetr_model(
        fp32_model,
        track_distributions=track_distributions,
        device=device
    )
    
    print("Quantized model created successfully!")
    print(f"Number of quantized linear layers: {count_quantized_layers(quant_model)}")
    
    return quant_model


def count_quantized_layers(model: nn.Module) -> int:
    """Count number of quantized layers in model."""
    count = 0
    for module in model.modules():
        if 'Quantized' in module.__class__.__name__:
            count += 1
    return count


def create_dummy_batch(batch_size=1, img_size=800, device='cuda'):
    """
    Create dummy batch for initialization and benchmarking.
    
    Args:
        batch_size: Batch size
        img_size: Image size
        device: Device to create on
    
    Returns:
        Dummy batch tensor
    """
    # Create dummy image batch
    # Adjust dimensions based on your model's input format
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
    return dummy_input


def initialize_quantization(quant_model: nn.Module, num_batches=10, 
                           batch_size=1, device='cuda'):
    """
    Initialize quantization ranges using representative data.
    
    Args:
        quant_model: Quantized model
        num_batches: Number of batches to use for initialization
        batch_size: Batch size
        device: Device to use
    """
    print(f"\nInitializing quantization ranges with {num_batches} batches...")
    
    quant_model.eval()
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Create dummy batch (replace with real data in production)
            dummy_input = create_dummy_batch(batch_size, device=device)
            
            # Forward pass to collect statistics
            try:
                _ = quant_model(dummy_input)
            except Exception as e:
                print(f"Warning: {e}")
                pass
            
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                print(f"  Processed {batch_idx + 1}/{num_batches} batches")
    
    print("Quantization initialization complete!")


def benchmark_models(fp32_model: nn.Module, quant_model: nn.Module,
                    num_batches=10, batch_size=1, device='cuda',
                    output_dir='./quant_benchmark') -> Dict:
    """
    Benchmark and compare FP32 vs INT8 models.
    
    Args:
        fp32_model: FP32 reference model
        quant_model: INT8 quantized model
        num_batches: Number of batches to test
        batch_size: Batch size
        device: Device to use
        output_dir: Directory to save results
    
    Returns:
        Benchmark results dictionary
    """
    print(f"\nStarting benchmarking (testing on {num_batches} batches)...")
    
    benchmark = QuantizationBenchmark(
        fp32_model, quant_model,
        batch_size=batch_size,
        output_dir=output_dir
    )
    
    # Run test batches
    mse_values = []
    mae_values = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Create dummy batch
            dummy_input = create_dummy_batch(batch_size, device=device)
            
            # Compare models
            stats, out_fp32, out_quant = benchmark.compare_activations(dummy_input, device)
            
            mse_values.append(stats['output_mse'])
            mae_values.append(stats['output_mae'])
            
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches}")
                print(f"    MSE: {stats['output_mse']:.6f}")
                print(f"    MAE: {stats['output_mae']:.6f}")
                print(f"    Max Diff: {stats['output_max_diff']:.6f}")
            
            # Generate plots from first batch
            if batch_idx == 0:
                benchmark.plot_weight_distribution()
                benchmark.plot_activation_distribution()
                benchmark.plot_comparison_stats(out_fp32, out_quant)
    
    # Compute aggregated statistics
    results = {
        'avg_mse': sum(mse_values) / len(mse_values),
        'avg_mae': sum(mae_values) / len(mae_values),
        'mse_values': mse_values,
        'mae_values': mae_values,
        'num_batches_tested': num_batches,
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Average MSE: {results['avg_mse']:.6f}")
    print(f"  Average MAE: {results['avg_mae']:.6f}")
    print(f"  Plots saved to: {Path(output_dir)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Quantize LW-DETR model to INT8')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained FP32 LW-DETR checkpoint')
    parser.add_argument('--output-dir', type=str, default='./quantized_models',
                       help='Directory to save quantized model')
    parser.add_argument('--benchmark-dir', type=str, default='./quant_benchmark',
                       help='Directory to save benchmark results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for benchmarking')
    parser.add_argument('--num-init-batches', type=int, default=10,
                       help='Number of batches for quantization initialization')
    parser.add_argument('--num-benchmark-batches', type=int, default=10,
                       help='Number of batches for benchmarking')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmarking after quantization')
    parser.add_argument('--track-distributions', action='store_true', default=True,
                       help='Track activation distributions (default: True)')
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    Path(args.benchmark_dir).mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("LW-DETR INT8 Quantization Pipeline")
    print("="*60)
    
    # Load FP32 model
    fp32_model = load_fp32_model(args.checkpoint, device=args.device)
    
    # Create quantized model
    quant_model = create_quantized_model(fp32_model, 
                                        track_distributions=args.track_distributions,
                                        device=args.device)
    
    # Initialize quantization ranges
    initialize_quantization(quant_model, 
                          num_batches=args.num_init_batches,
                          batch_size=args.batch_size,
                          device=args.device)
    
    # Save quantized model
    quant_checkpoint = Path(args.output_dir) / 'lwdetr_int8_quantized.pt'
    save_quantized_model(quant_model, str(quant_checkpoint), 
                        include_distributions=args.track_distributions)
    
    print(f"\nQuantized model saved to {quant_checkpoint}")
    
    # Run benchmarking if requested
    if args.benchmark:
        results = benchmark_models(fp32_model, quant_model,
                                 num_batches=args.num_benchmark_batches,
                                 batch_size=args.batch_size,
                                 device=args.device,
                                 output_dir=args.benchmark_dir)
        
        # Save results
        import json
        results_file = Path(args.benchmark_dir) / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Benchmark results saved to {results_file}")
    
    print("\n" + "="*60)
    print("Quantization pipeline complete!")
    print("="*60)


if __name__ == '__main__':
    main()
