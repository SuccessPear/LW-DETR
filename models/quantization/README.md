# LW-DETR Quantization Module

Complete INT8 quantization framework for LW-DETR with distribution tracking and benchmarking.

## Module Overview

This quantization module provides:
- ✅ Full INT8 quantization for all components (linear, attention, deformable attention)
- ✅ Symmetric quantization with learned ranges
- ✅ Per-channel weight quantization
- ✅ Distribution tracking for analysis and visualization
- ✅ Automatic benchmarking tools
- ✅ End-to-end model conversion pipeline

## Quick Start

### Basic Usage (3 lines)

```python
from models.quantization import create_quantized_lwdetr_model

quant_model = create_quantized_lwdetr_model(fp32_model, track_distributions=True)
# ... calibrate with data ...
# Done! Model now runs in INT8
```

### With Benchmarking

```python
from models.quantization import QuantizationBenchmark

benchmark = QuantizationBenchmark(fp32_model, quant_model)
benchmark.compare_activations(sample_input)
benchmark.plot_weight_distribution()
benchmark.plot_activation_distribution()
```

### Full Pipeline with CLI

```bash
python quantize_lwdetr_example.py --checkpoint model.pt --benchmark
```

## Files

| File | Purpose | Functions |
|------|---------|-----------|
| `quant_modules.py` | Core quantized operations | QuantizedLinear, QuantizedAct, QuantizedMatMul, etc. |
| `quant_transformer.py` | Transformer components | QuantizedMultiheadAttention, QuantizedMLP, QuantizedDecoderLayer |
| `quant_deformable_attn.py` | Deformable attention | QuantizedMSDeformAttn, QuantizedDeformableAttention |
| `conversion.py` | Model conversion | create_quantized_lwdetr_model, initialize_quantization_ranges |
| `benchmark.py` | Benchmarking tools | QuantizationBenchmark, compare activations & visualizations |
| `lwdetr_integration.py` | LW-DETR integration | QuantizedLWDETRTransformer, quantize_lwdetr_model |
| `__init__.py` | Module exports | All public functions and classes |

## API Reference

### Main Functions

#### `create_quantized_lwdetr_model(model, track_distributions=True, device='cuda')`
Convert FP32 model to INT8 quantized version.

```python
quant_model = create_quantized_lwdetr_model(
    original_model,
    track_distributions=True,  # Enable for analysis
    device='cuda'
)
```

#### `initialize_quantization_ranges(model, dataloader, num_batches=10, device='cuda')`
Calibrate quantization ranges using representative data.

```python
quant_model = initialize_quantization_ranges(
    quant_model,
    train_dataloader,
    num_batches=20
)
```

#### `QuantizationBenchmark(fp32_model, quant_model, output_dir='./results')`
Benchmark and compare FP32 vs INT8 performance.

```python
benchmark = QuantizationBenchmark(fp32_model, quant_model)
stats, out_fp32, out_quant = benchmark.compare_activations(sample_input)
benchmark.plot_weight_distribution()
benchmark.plot_activation_distribution()
benchmark.plot_comparison_stats(out_fp32, out_quant)
```

### Core Quantized Modules

#### `QuantizedLinear(in_features, out_features, weight_bit=8, track_distributions=False)`
INT8 fully connected layer.

```python
layer = QuantizedLinear(256, 256, weight_bit=8, track_distributions=True)
output, scale = layer(x, prev_scale)
```

#### `QuantizedMultiheadAttention(embed_dim, num_heads, track_distributions=False)`
INT8 multihead attention block.

```python
attn = QuantizedMultiheadAttention(256, 8, track_distributions=True)
output, attn_weights = attn(query, key, value)
```

#### `QuantizedMSDeformAttn(d_model, n_levels, n_heads, n_points, track_distributions=False)`
Quantized multi-scale deformable attention.

```python
deform_attn = QuantizedMSDeformAttn(256, 4, 8, 4, track_distributions=True)
output, scale = deform_attn(query, ref_points, value, spatial_shapes, ...)
```

## Quantization Method

**Symmetric INT8 Quantization**:
- Range: [-127, 127]
- Scale computation: $S = \max(|x|) / 127$
- Quantization: $Q(x) = \text{round}(x / S)$
- Per-channel for weights, per-tensor for activations

## Performance

| Metric | Expected |
|--------|----------|
| Model Size | ~4x smaller |
| Inference Speed | ~2-4x faster* |
| Memory Usage | ~3-4x less |
| Accuracy Loss | < 1-2% |

*Depends on hardware INT8 support

## Distribution Tracking

Each quantized module can track:
- FP32 weights and activations
- INT8 weights and activations
- Scaling factors

Access via:
```python
module.distributions['weight_fp32']        # FP32 weights
module.distributions['weight_quant']        # INT8 weights
module.distributions['activation_fp32']     # FP32 activations
module.distributions['activation_quant']    # INT8 activations
```

Generate visualizations:
```python
benchmark.plot_weight_distribution()
benchmark.plot_activation_distribution()
```

## Benchmarking Outputs

Generates in `./quant_benchmark/`:
- `weight_distribution.png` - Histograms of weights per layer
- `activation_distribution.png` - FP32 vs INT8 activations
- `output_comparison.png` - Performance metrics and scatter plots
- `benchmark_report.json` - Detailed statistics

## Examples

### Example 1: Simple Quantization

```python
from models.quantization import create_quantized_lwdetr_model
import torch

# Load model
model = torch.load('checkpoint.pt')

# Quantize
quant_model = create_quantized_lwdetr_model(model)

# Calibrate with data
for batch in dataloader[:10]:
    with torch.no_grad():
        _ = quant_model(batch)

# Save
torch.save(quant_model, 'model_int8.pt')
```

### Example 2: Benchmarking

```python
from models.quantization import QuantizationBenchmark
import torch

# Create benchmark
bench = QuantizationBenchmark(fp32_model, quant_model, output_dir='./bench')

# Test
sample = torch.randn(2, 3, 800, 800)
stats, out_fp32, out_quant = bench.compare_activations(sample)

print(f"MSE: {stats['output_mse']:.6f}")
print(f"MAE: {stats['output_mae']:.6f}")

# Visualize
bench.plot_weight_distribution()
bench.plot_activation_distribution()
```

### Example 3: Component-Level

```python
from models.quantization import QuantizedLinear, QuantizedMultiheadAttention

# Create individual components
fc = QuantizedLinear(256, 512, track_distributions=True)
attn = QuantizedMultiheadAttention(256, 8, track_distributions=True)

# Use in custom model
output1, scale1 = fc(x, None)
output2, _ = attn(output1, output1, output1)
```

## Integration with Training

For quantization-aware training:

```python
from models.quantization import create_quantized_lwdetr_model

# Quantize model
quant_model = create_quantized_lwdetr_model(fp32_model)
quant_model.train()

# Fix scales for stable training
for m in quant_model.modules():
    if hasattr(m, 'fix'):
        m.fix()

# Train normally
for imgs, targets in dataloader:
    out = quant_model(imgs, targets)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
```

## Troubleshooting

### Q: Where are the plots saved?
A: By default in `./quant_benchmark/`. Specify custom directory:
```python
benchmark = QuantizationBenchmark(..., output_dir='./my_results')
```

### Q: How much data for calibration?
A: 5-20 batches typically enough. More data = better quantization ranges.

### Q: Can I use different bit-widths?
A: Yes, pass `weight_bit` parameter:
```python
layer = QuantizedLinear(256, 256, weight_bit=4)  # INT4
```

### Q: How to check if quantization worked?
A: Run validation script:
```python
from quantization_validation import run_full_validation
report = run_full_validation(fp32_model, quant_model)
```

## Advanced Features

### Custom Per-Layer Quantization

```python
from models.quantization import QuantizedLinear

# Different settings per layer
critical_layer = QuantizedLinear(256, 256, weight_bit=8, track_distributions=True)
normal_layer = QuantizedLinear(256, 256, weight_bit=8, track_distributions=False)
```

### Export to ONNX

```python
import torch

sample_input = torch.randn(1, 3, 800, 800)
torch.onnx.export(
    quant_model,
    sample_input,
    'model_int8.onnx',
    opset_version=13,
    do_constant_folding=True
)
```

### Mixed Precision

Some layers in higher precision:
```python
from models.quantization import QuantizedLinear
from torch import nn

# Critical layers in higher precision
critical = QuantizedLinear(256, 256, weight_bit=16, track_distributions=True)

# Regular layers INT8
regular = QuantizedLinear(256, 256, weight_bit=8, track_distributions=True)
```

## Documentation

- **QUANTIZATION_GUIDE.md** - Complete technical guide
- **QUANTIZATION_QUICK_START.md** - Quick integration guide
- **IMPLEMENTATION_SUMMARY.md** - Detailed implementation summary

## Citation

If you use this quantization framework:

```bibtex
@framework{lwdetr_quantization,
  title={LW-DETR INT8 Quantization Framework},
  year={2024},
  institution={Baidu}
}
```

## License

Apache License 2.0 - See LICENSE file

## Support

For issues, check:
1. Documentation files
2. Example scripts
3. Validation suite: `python quantization_validation.py`

---

**Status**: Production Ready ✅
**Last Updated**: 2024
**Version**: 1.0
