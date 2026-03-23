# LW-DETR INT8 Quantization - Quick Integration Guide

## Summary

This quantization framework provides complete INT8 conversion for LW-DETR with:
- ✅ All layers quantized (Linear, Attention, Deformable Attention, MLP)
- ✅ Distribution tracking for FP32 vs INT8 comparison
- ✅ Automatic benchmarking and visualization
- ✅ Symmetric INT8 quantization (compatible with hardware deployment)
- ✅ Per-channel weight quantization
- ✅ Learned quantization scales via calibration

## Quick Start (3 steps)

### Step 1: Convert Your FP32 Model to INT8

```python
from models.quantization import create_quantized_lwdetr_model, initialize_quantization_ranges
import torch

# Load your trained FP32 model
fp32_model = torch.load('trained_lwdetr.pt')

# Convert to INT8 (all layers quantized)
quant_model = create_quantized_lwdetr_model(fp32_model, track_distributions=True)

# Initialize quantization with representative data
# Feed 5-20 batches to calibrate quantization ranges
for batch in dataloader[:20]:
    with torch.no_grad():
        _ = quant_model(batch)

print("✓ Model quantized to INT8!")
```

### Step 2: Benchmark FP32 vs INT8

```python
from models.quantization import QuantizationBenchmark
import torch

# Create benchmark tool
benchmark = QuantizationBenchmark(fp32_model, quant_model, output_dir='./results')

# Test on sample data
sample_input = torch.randn(2, 3, 800, 800).cuda()
stats, out_fp32, out_quant = benchmark.compare_activations(sample_input)

print(f"Output MSE: {stats['output_mse']:.6f}")
print(f"Output MAE: {stats['output_mae']:.6f}")

# Generate comparison plots
benchmark.plot_weight_distribution()
benchmark.plot_activation_distribution()  
benchmark.plot_comparison_stats(out_fp32, out_quant)

# Checks
# ✓ Plots saved to ./results/
# ✓ weight_distribution.png - shows weight ranges in each layer
# ✓ activation_distribution.png - shows FP32 vs INT8 activations
# ✓ output_comparison.png - overall performance comparison
```

### Step 3: Use in Inference

```python
# Inference - no changes needed!
quant_model.eval()

with torch.no_grad():
    # Model runs in INT8 internally
    detections = quant_model(images)
    
# Expected improvements:
# - 2-4x faster inference (on INT8-capable hardware)
# - 4x smaller model
# - Minor accuracy loss (typically < 1-2%)
```

## File Reference

| File | Purpose | What It Does |
|------|---------|-------------|
| `quant_modules.py` | Core quantized operations | Symmetric INT8 linear, activation, matmul layers |
| `quant_transformer.py` | Quantized attention blocks | MultiheadAttention, MLP, Transformer blocks |
| `quant_deformable_attn.py` | Deformable attention | Quantized MS-Deform-Attn with bilinear sampling |
| `conversion.py` | Model conversion | FP32→INT8 conversion, model loading/saving |
| `benchmark.py` | Benchmarking tools | Distribution tracking, visualization, comparison |
| `quantize_lwdetr_example.py` | End-to-end example | Complete pipeline with CLI |

## What Gets Quantized

✅ **Everything to INT8**:
- All Linear layers (fully connected)
- Attention projections (Q, K, V, output)
- Deformable attention offset/weight projections
- MLP layers (first and second linear)
- Matrix multiplications in attention
- Activation functions (ReLU, GELU)

✅ **Kept in higher precision** (selective):
- LayerNorm (typically FP32)
- Softmax (computed in FP32, then quantized)
- Position embeddings (can be FP32 or INT8)

## Distribution Tracking

Each quantized module tracks:
- **FP32 weights/activations** - original values
- **INT8 weights/activations** - quantized values
- **Scaling factors** - for reverse quantization

Access tracked data:
```python
for name, module in quant_model.named_modules():
    if hasattr(module, 'distributions'):
        print(f"{name}:")
        # module.distributions contains:
        # - 'weight_fp32': FP32 weights
        # - 'weight_quant': INT8 weights
        # - 'activation_fp32': FP32 activations
        # - 'activation_quant': INT8 activations
```

## Expected Results

After quantization, you should see:

1. **Weight Distributions** plots showing:
   - Typical range: [-127, 127] for INT8
   - Per-channel scales in plot titles
   - Smooth histograms (good quantization)

2. **Activation Distributions** showing:
   - Blue: Original FP32 activations
   - Red: Quantized INT8 activations
   - Good overlap = good quantization quality

3. **Output Comparison** metrics:
   - MSE < 0.01: Excellent
   - MSE < 0.05: Good
   - MAE < 0.1: Acceptable
   - Correlation > 0.99: Very good match

## Performance Estimates

| Metric | Expected Value |
|--------|-----------------|
| Model Size | ~4x smaller |
| Inference Speed | ~2-4x faster |
| Memory Usage | ~3-4x less |
| Accuracy Loss | < 1-2% mAP |

## Troubleshooting

### Model accuracy drops significantly after quantization?

```python
# Use more calibration data
initialize_quantization_ranges(quant_model, dataloader, num_batches=50)

# Or use Quantization-Aware Training (QAT) instead
# (fine-tune with quantization for better accuracy)
```

### Distributions don't overlap well in plots?

```python
# Use more diverse calibration data
# Ensure data represents full range of model inputs
# Try different layer quantization settings
```

### Can't run inference due to shape mismatches?

```python
# Check that quantized model accepts same input format as original
# Verify all intermediate layers are correctly quantized
# Use debug mode:
model.eval()
with torch.no_grad():
    try:
        output = model(input_sample)
    except Exception as e:
        print(f"Error: {e}")
        # Fix incompatibility in conversion code
```

## Advanced: Custom Quantization Strategy

Replace default uniform quantization with per-layer settings:

```python
from models.quantization import QuantizedLinear, QuantizedLayerNorm

# Create custom layer with higher precision
custom_layer = QuantizedLinear(
    in_features=256,
    out_features=256,
    weight_bit=8,        # INT8
    bias_bit=16,         # INT16 (optional mixed precision)
    per_channel=True,
    track_distributions=True
)
```

## Integration Checklist

- [ ] Prepare trained FP32 model checkpoint
- [ ] Run quantization conversion (`create_quantized_lwdetr_model`)
- [ ] Calibrate with representative data (5-20 batches)
- [ ] Run benchmarking on validation set
- [ ] Review distribution plots
- [ ] Check MSE/MAE metrics
- [ ] Validate inference output matches expected format
- [ ] Test inference speed on target hardware
- [ ] Deploy quantized model

## For Production Deployment

For actual deployment optimization, consider:

```python
# Export to ONNX for optimized inference
import onnx
torch.onnx.export(quant_model, sample_input, 'model.onnx',
                 input_names=['images'],
                 output_names=['predictions'],
                 opset_version=13,
                 do_constant_folding=True)

# Or use TorchScript
scripted_model = torch.jit.trace(quant_model, sample_input)
scripted_model.save('model_scripted.pt')
```

## Getting Help

Check these files for more info:
- `QUANTIZATION_GUIDE.md` - Complete guide
- `quantize_lwdetr_example.py` - Example usage
- `models/quantization/__init__.py` - Available functions

## Key Takeaways

1. **Automatic Quantization**: All layers converted with one call
2. **Distribution Tracking**: See exactly what's changing during quantization
3. **Benchmarking Built-in**: Automatic comparison and visualization
4. **Production Ready**: INT8 compatible with hardware deployment
5. **Easy Integration**: Drop-in model replacement for inference

---

**Next Steps**:
1. Run: `python quantize_lwdetr_example.py --checkpoint model.pt --benchmark`
2. Check generated plots in `./quant_benchmark/`
3. Compare metrics with your expectations
4. Fine-tune quantization ranges if needed
5. Deploy quantized model!
