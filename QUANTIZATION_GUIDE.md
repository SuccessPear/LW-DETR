# LW-DETR INT8 Quantization Guide

This guide explains how to quantize LW-DETR models from FP32 to INT8 with distribution tracking for benchmarking.

## Overview

The quantization pipeline consists of:
1. **Quantized Modules** - INT8 implementations of linear layers, attention, activations
2. **Deformable Attention Quantization** - Quantized MS-Deform-Attn blocks
3. **Distribution Tracking** - Tracks FP32 vs INT8 activations for analysis
4. **Model Conversion** - Converts trained FP32 models to INT8
5. **Benchmarking & Visualization** - Generates plots comparing distributions

## Architecture

### Quantized Modules (`models/quantization/quant_modules.py`)
- `QuantizedLinear`: INT8 linear layers with symmetric quantization
- `QuantizedAct`: INT8 activation quantization
- `QuantizedMatMul`: Quantized matrix multiplication
- `QuantizedLayerNorm`: INT8-friendly layer normalization
- `QuantizedReLU`, `QuantizedGELU`: Quantized activation functions

### Transformer Layers (`models/quantization/quant_transformer.py`)
- `QuantizedMultiheadAttention`: Full INT8 attention mechanism
- `QuantizedMLP`: Quantized feedforward network
- `QuantizedTransformerEncoderLayer`: Complete encoder block
- `QuantizedDecoderLayer`: Complete decoder block with cross-attention

### Deformable Attention (`models/quantization/quant_deformable_attn.py`)
- `QuantizedMSDeformAttn`: INT8 multi-scale deformable attention (primary)
- `QuantizedDeformableAttention`: Simplified quantized deformable attention

### Utilities

#### Conversion (`models/quantization/conversion.py`)
```python
# Main entry point for quantization
quant_model = create_quantized_lwdetr_model(
    original_model,
    track_distributions=True,  # Enable distribution tracking
    device='cuda'
)

# Initialize quantization ranges with representative data
quant_model = initialize_quantization_ranges(
    quant_model,
    dataloader,
    num_batches=10
)

# Save quantized model
save_quantized_model(quant_model, 'model_int8.pt')
```

#### Benchmarking (`models/quantization/benchmark.py`)
```python
benchmark = QuantizationBenchmark(
    fp32_model,
    quant_model,
    output_dir='./quant_benchmark'
)

# Generate comparison plots
benchmark.plot_weight_distribution()
benchmark.plot_activation_distribution()
benchmark.plot_comparison_stats(fp32_outputs, int8_outputs)

# Run comprehensive benchmark
results = benchmark_quantization(
    fp32_model,
    quant_model,
    dataloader,
    num_batches=10
)
```

## Usage Examples

### Basic Quantization

```python
import torch
from models.quantization import create_quantized_lwdetr_model

# Load your trained FP32 model
fp32_model = torch.load('checkpoint_fp32.pt')

# Convert to INT8
quant_model = create_quantized_lwdetr_model(
    fp32_model,
    track_distributions=True
)

# Run sample forward pass for quantization calibration
sample_input = torch.randn(1, 3, 800, 800)
with torch.no_grad():
    _ = quant_model(sample_input)

print(f"Model quantized! Quantized layers: {len([m for m in quant_model.modules() if 'Quantized' in m.__class__.__name__])}")
```

### With Benchmarking

```python
from models.quantization import (
    create_quantized_lwdetr_model,
    QuantizationBenchmark
)

# Create quantized model
quant_model = create_quantized_lwdetr_model(fp32_model)

# Create benchmark tool
benchmark = QuantizationBenchmark(fp32_model, quant_model, output_dir='./results')

# Compare on sample input
sample_input = torch.randn(2, 3, 800, 800).cuda()
stats, out_fp32, out_quant = benchmark.compare_activations(sample_input)

print(f"MSE: {stats['output_mse']:.6f}")
print(f"MAE: {stats['output_mae']:.6f}")

# Generate visualization plots
benchmark.plot_weight_distribution()
benchmark.plot_activation_distribution()
benchmark.plot_comparison_stats(out_fp32, out_quant)
```

### Command Line Usage

```bash
# Full pipeline with benchmarking
python quantize_lwdetr_example.py \
    --checkpoint models/lwdetr_checkpoint.pt \
    --output-dir ./quantized_models \
    --benchmark-dir ./benchmark_results \
    --num-init-batches 20 \
    --num-benchmark-batches 20 \
    --benchmark

# Only quantization without benchmarking
python quantize_lwdetr_example.py \
    --checkpoint models/lwdetr_checkpoint.pt \
    --output-dir ./quantized_models \
    --device cuda
```

## Features

### Distribution Tracking

Each quantized module can track:
- **Weight distributions** (FP32 and quantized)
- **Activation distributions** (FP32 and quantized)
- **Output distributions** for comparison

```python
# Access tracked distributions
for name, module in quant_model.named_modules():
    if hasattr(module, 'distributions'):
        print(f"{name}:")
        print(f"  FP32 activations: {len(module.distributions['activation_fp32'])} samples")
        print(f"  INT8 activations: {len(module.distributions['activation_quant'])} samples")
```

### Visualization

The benchmark generates several plots:

1. **weight_distribution.png** - Histogram of weights in all quantized layers
2. **activation_distribution.png** - FP32 vs INT8 activation distributions
3. **output_comparison.png** - Output statistics and scatter plots

## Quantization Method

**Symmetric INT8 Quantization**:
$$Q(x) = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right), -127, 127\right)$$

Where $S$ is the scaling factor:
$$S = \frac{\max(|x|)}{127}$$

**Features**:
- Per-channel quantization for weights
- Learnable quantization ranges during initialization
- Straight-through estimator (STE) for gradients
- Compatible with deployment on INT8 hardware

## Performance Expectations

When converting from FP32 to INT8:
- **Inference speed**: 2-4x faster (depending on hardware)
- **Model size**: 4x smaller
- **Accuracy loss**: Typically < 1-2% mAP for object detection

The benchmark reports:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- Output correlation between FP32 and INT8

## Integration with LW-DETR

### Step 1: Replace Attention (Optional)

```python
from models.quantization import replace_attention_with_quantized

# Automatically replace attention layers
quant_model = replace_attention_with_quantized(
    quant_model,
    use_deformable=True,
    track_distributions=True
)
```

### Step 2: Use in Training

For fine-tuning with quantization awareness:

```python
# Enable training mode while keeping quantization scales fixed
for module in quant_model.modules():
    if hasattr(module, 'fix'):
        module.fix()

# Train model
quant_model.train()
for images, targets in dataloader:
    outputs = quant_model(images, targets)
    loss = compute_loss(outputs, targets)
    loss.backward()
    optimizer.step()
```

### Step 3: Inference

For pure INT8 inference:

```python
quant_model.eval()

with torch.no_grad():
    # No special handling needed - model runs in INT8 internally
    detections = quant_model(images)
```

## File Structure

```
LW-DETR/
├── models/
│   └── quantization/
│       ├── __init__.py                 # Main export
│       ├── quant_modules.py            # Base quantized layers
│       ├── quant_transformer.py        # Quantized attention/transformer
│       ├── quant_deformable_attn.py    # Quantized deformable attention
│       ├── conversion.py               # Model conversion utilities
│       └── benchmark.py                # Benchmarking and visualization
├── quantize_lwdetr_example.py          # Full quantization pipeline example
└── QUANTIZATION_GUIDE.md               # This file
```

## Advanced Configurations

### Custom Quantization Settings

```python
from models.quantization import QuantizedLinear

# Create with custom bit-width
layer = QuantizedLinear(
    in_features=256,
    out_features=256,
    weight_bit=8,        # INT8 weights
    bias_bit=32,         # FP32 bias
    per_channel=True,    # Per-channel quantization
    quant_mode='symmetric',
    track_distributions=True
)
```

### Layer-Specific Tracking

```python
# Enable tracking only for specific layers
for name, module in quant_model.named_modules():
    if 'decoder' in name:
        module.track_distributions = True
    else:
        module.track_distributions = False
```

## Troubleshooting

### Q: Model inference is slower after quantization?
A: Ensure your deployment hardware supports INT8 operations. Pure Python evaluation won't show speed benefits - use ONNX or compiled backends.

### Q: Large accuracy drop after quantization?
A: 
- Increase `num_init_batches` for better calibration
- Use quantization-aware training (QAT) instead of post-quantization
- Check that distributions are being tracked properly

### Q: Memory usage not reduced?
A: The PyTorch model still stores FP32 for backward compatibility. For deployment, use ONNX export or model compression tools.

## Performance Metrics

The benchmark reports these metrics:

| Metric | Description | Good Range |
|--------|-------------|-----------|
| MSE    | Mean squared error between FP32 and INT8 outputs | < 0.01 |
| MAE    | Mean absolute error | < 0.1 |
| Max Diff | Maximum difference between any outputs | < 1.0 |
| Correlation | Pearson correlation of output distributions | > 0.99 |

## References

- I-ViT Quantization: https://github.com/THU-MIG/INT-ViT
- LW-DETR: Lightweight Detection Transformer
- INT8 Quantization: https://arxiv.org/abs/1906.06822

## Citation

If you use this quantization framework, please cite:

```bibtex
@article{lwdetr2024,
  title={LW-DETR: Lightweight Detection Transformer},
  year={2024}
}
```

## License

Licensed under the Apache License Version 2.0 [see LICENSE for details]
