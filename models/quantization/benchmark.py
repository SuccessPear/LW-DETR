"""
Benchmarking and visualization script for quantized LW-DETR.
Tracks and plots distributions of activations and weights in quantized blocks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict


class QuantizationBenchmark:
    """
    Benchmarking tool for quantized models with distribution tracking and visualization.
    """
    
    def __init__(self, model_fp32, model_quant, batch_size=1, 
                 output_dir='./quant_benchmark'):
        """
        Initialize benchmark with FP32 and quantized models.
        
        Args:
            model_fp32: Original FP32 model
            model_quant: Quantized INT8 model
            batch_size: Batch size for testing
            output_dir: Directory to save benchmark results
        """
        self.model_fp32 = model_fp32
        self.model_quant = model_quant
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage for statistics
        self.stats = defaultdict(dict)
        self.layer_names = []
        
        self._extract_layer_names()
    
    def _extract_layer_names(self):
        """Extract layer names from models for tracking."""
        for name, module in self.model_quant.named_modules():
            if hasattr(module, 'distributions'):
                self.layer_names.append(name)
    
    def compare_activations(self, input_tensor, device='cuda'):
        """
        Compare activations between FP32 and quantized models.
        
        Args:
            input_tensor: Input data (batch, features...)
            device: Device to run on
        
        Returns:
            Dictionary with comparison statistics
        """
        self.model_fp32.eval()
        self.model_quant.eval()
        
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            # FP32 forward pass
            out_fp32 = self.model_fp32(input_tensor)
            
            # Quantized forward pass
            out_quant = self.model_quant(input_tensor)
        
        # Calculate statistics
        if isinstance(out_fp32, (tuple, list)):
            out_fp32 = out_fp32[0] if isinstance(out_fp32, tuple) else out_fp32[0]
        if isinstance(out_quant, (tuple, list)):
            out_quant = out_quant[0] if isinstance(out_quant, tuple) else out_quant[0]
        
        # Compute differences
        diff = (out_fp32 - out_quant).abs()
        
        stats = {
            'output_mse': torch.mean((out_fp32 - out_quant) ** 2).item(),
            'output_mae': torch.mean(diff).item(),
            'output_max_diff': torch.max(diff).item(),
            'output_fp32_range': (out_fp32.min().item(), out_fp32.max().item()),
            'output_quant_range': (out_quant.min().item(), out_quant.max().item()),
        }
        
        return stats, out_fp32, out_quant
    
    def plot_weight_distribution(self, save_name='weight_distribution.png'):
        """
        Plot weight distributions for all quantized linear layers.
        
        Args:
            save_name: Name of output image file
        """
        linear_layers = []
        for name, module in self.model_quant.named_modules():
            if name and hasattr(module, 'weight') and hasattr(module, 'weight_scale'):
                if isinstance(module, nn.Linear) or 'QuantizedLinear' in module.__class__.__name__:
                    linear_layers.append((name, module))
        
        n_layers = len(linear_layers)
        if n_layers == 0:
            print("No quantized linear layers found")
            return
        
        # Create subplots
        n_cols = min(3, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 5 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        for idx, (name, module) in enumerate(linear_layers):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Get weights
            weights = module.weight.data.cpu().numpy().flatten()
            
            # Plot histogram
            ax.hist(weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f'Layer: {name.split(".")[-1]}', fontsize=10)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'μ={weights.mean():.4f}\nσ={weights.std():.4f}\nmin={weights.min():.4f}\nmax={weights.max():.4f}'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Weight distribution plot saved to {save_path}")
        plt.close()
    
    def plot_activation_distribution(self, save_name='activation_distribution.png', 
                                     max_layers=12):
        """
        Plot activation distributions from tracked quantization modules.
        
        Args:
            save_name: Name of output image file
            max_layers: Maximum number of layers to plot
        """
        tracked_modules = []
        for name, module in self.model_quant.named_modules():
            if hasattr(module, 'distributions') and module.distributions:
                tracked_modules.append((name, module))
        
        n_layers = min(len(tracked_modules), max_layers)
        if n_layers == 0:
            print("No tracked activations found. Run model forward pass first.")
            return
        
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(16, 3.5 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        for idx in range(n_layers):
            name, module = tracked_modules[idx]
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Collect activations
            if hasattr(module, 'distributions'):
                dists = module.distributions
                
                # Plot FP32 vs Quantized if both available
                if 'activation_fp32' in dists and dists['activation_fp32']:
                    fp32_acts = np.concatenate([a.flatten() for a in dists['activation_fp32'][-5:]])  # Last 5
                    ax.hist(fp32_acts, bins=40, alpha=0.6, label='FP32', color='blue', edgecolor='black')
                
                if 'activation_quant' in dists and dists['activation_quant']:
                    quant_acts = np.concatenate([a.flatten() for a in dists['activation_quant'][-5:]])
                    ax.hist(quant_acts, bins=40, alpha=0.6, label='INT8', color='red', edgecolor='black')
                elif 'fp32' in dists and dists['fp32']:
                    acts = np.concatenate([a.flatten() for a in dists['fp32'][-5:]])
                    ax.hist(acts, bins=40, alpha=0.7, color='green', edgecolor='black')
            
            ax.set_title(f'{name.split(".")[-1]}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Activation distribution plot saved to {save_path}")
        plt.close()
    
    def plot_comparison_stats(self, fp32_outputs, quant_outputs, 
                             save_name='output_comparison.png'):
        """
        Plot comparison statistics between FP32 and quantized outputs.
        
        Args:
            fp32_outputs: FP32 model outputs
            quant_outputs: Quantized model outputs
            save_name: Output image file name
        """
        fp32_out = fp32_outputs.cpu().numpy().flatten()
        quant_out = quant_outputs.cpu().numpy().flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Output distributions
        axes[0, 0].hist([fp32_out, quant_out], bins=50, label=['FP32', 'INT8'],
                        alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Output Distributions')
        axes[0, 0].set_xlabel('Output Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Difference distribution
        diff = fp32_out - quant_out
        axes[0, 1].hist(diff, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_title('FP32 - INT8 Difference Distribution')
        axes[0, 1].set_xlabel('Difference')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot
        sample_size = min(5000, len(fp32_out))
        sample_idx = np.random.choice(len(fp32_out), sample_size, replace=False)
        axes[1, 0].scatter(fp32_out[sample_idx], quant_out[sample_idx], alpha=0.5, s=1)
        axes[1, 0].plot([fp32_out.min(), fp32_out.max()], 
                       [fp32_out.min(), fp32_out.max()], 'r--', lw=2)
        axes[1, 0].set_title('FP32 vs INT8 Outputs')
        axes[1, 0].set_xlabel('FP32 Output')
        axes[1, 0].set_ylabel('INT8 Output')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Statistics
        axes[1, 1].axis('off')
        stats_text = f"""
        FP32 Stats:
          Mean: {fp32_out.mean():.6f}
          Std: {fp32_out.std():.6f}
          Min: {fp32_out.min():.6f}
          Max: {fp32_out.max():.6f}
        
        INT8 Stats:
          Mean: {quant_out.mean():.6f}
          Std: {quant_out.std():.6f}
          Min: {quant_out.min():.6f}
          Max: {quant_out.max():.6f}
        
        Comparison:
          MSE: {((fp32_out - quant_out) ** 2).mean():.6f}
          MAE: {np.abs(fp32_out - quant_out).mean():.6f}
          Max Diff: {np.abs(fp32_out - quant_out).max():.6f}
          Correlation: {np.corrcoef(fp32_out, quant_out)[0, 1]:.6f}
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Comparison statistics plot saved to {save_path}")
        plt.close()
    
    def generate_report(self, fps_model, quant_model, input_sample, device='cuda'):
        """
        Generate comprehensive benchmark report.
        
        Args:
            fps_model: FP32 model
            quant_model: Quantized model
            input_sample: Sample input for testing
            device: Device to run on
        
        Returns:
            Report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_fp32': str(fps_model.__class__.__name__),
            'model_quant': str(quant_model.__class__.__name__),
            'quantization_method': 'INT8 Symmetric Quantization',
        }
        
        # Run comparison
        stats, out_fp32, out_quant = self.compare_activations(input_sample, device)
        report['metrics'] = stats
        
        # Generate plots
        self.plot_weight_distribution()
        self.plot_activation_distribution()
        self.plot_comparison_stats(out_fp32, out_quant)
        
        # Save report
        report_path = self.output_dir / 'benchmark_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nBenchmark Report:")
        print(f"Output MSE: {stats['output_mse']:.6f}")
        print(f"Output MAE: {stats['output_mae']:.6f}")
        print(f"Output Max Diff: {stats['output_max_diff']:.6f}")
        print(f"\nFull report saved to {report_path}")
        
        return report


def benchmark_quantization(fp32_model, quant_model, dataloader, 
                          num_batches=5, output_dir='./quant_benchmark', device='cuda'):
    """
    Convenience function to run quantization benchmark.
    
    Args:
        fp32_model: FP32 model
        quant_model: Quantized model
        dataloader: DataLoader for test data
        num_batches: Number of batches to test
        output_dir: Output directory for results
        device: Device to run on
    """
    benchmark = QuantizationBenchmark(fp32_model, quant_model, output_dir=output_dir)
    
    print(f"Running quantization benchmark...")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"Processing batch {batch_idx + 1}/{num_batches}...")
        
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if isinstance(batch[0], torch.Tensor):
                input_tensor = batch[0]
            else:
                input_tensor = batch[0]['image'] if isinstance(batch[0], dict) else batch[0]
        else:
            input_tensor = batch
        
        # Run benchmark
        stats, out_fp32, out_quant = benchmark.compare_activations(input_tensor, device)
        
        # Print batch statistics
        print(f"  Batch {batch_idx} MSE: {stats['output_mse']:.6f}")
        print(f"  Batch {batch_idx} MAE: {stats['output_mae']:.6f}")
    
    # Generate plots
    print("\nGenerating visualization plots...")
    benchmark.plot_weight_distribution()
    benchmark.plot_activation_distribution()
    
    print(f"\nBenchmark complete! Results saved to {output_dir}")
    
    return benchmark
