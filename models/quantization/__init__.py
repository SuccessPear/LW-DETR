"""
Quantization package for LW-DETR with INT8 support and distribution tracking.
"""

from .quant_modules import (
    SymmetricQuantFunction,
    symmetric_linear_quantization_params,
    QuantizedLinear,
    QuantizedAct,
    QuantizedMatMul,
    QuantizedLayerNorm,
    QuantizedReLU,
    QuantizedGELU,
)

from .quant_transformer import (
    QuantizedMultiheadAttention,
    QuantizedMLP,
    QuantizedTransformerEncoderLayer,
    QuantizedDecoderLayer,
)

from .quant_deformable_attn import (
    QuantizedMSDeformAttn,
    QuantizedDeformableAttention,
)

from .conversion import (
    convert_module_to_quantized,
    convert_transformer_to_quantized,
    replace_attention_with_quantized,
    initialize_quantization_ranges,
    create_quantized_lwdetr_model,
    save_quantized_model,
    load_quantized_model,
)

from .benchmark import (
    QuantizationBenchmark,
    benchmark_quantization,
)

__all__ = [
    # Quantization modules
    'SymmetricQuantFunction',
    'symmetric_linear_quantization_params',
    'QuantizedLinear',
    'QuantizedAct',
    'QuantizedMatMul',
    'QuantizedLayerNorm',
    'QuantizedReLU',
    'QuantizedGELU',
    
    # Transformer layers
    'QuantizedMultiheadAttention',
    'QuantizedMLP',
    'QuantizedTransformerEncoderLayer',
    'QuantizedDecoderLayer',
    
    # Deformable attention
    'QuantizedMSDeformAttn',
    'QuantizedDeformableAttention',
    
    # Conversion utilities
    'convert_module_to_quantized',
    'convert_transformer_to_quantized',
    'replace_attention_with_quantized',
    'initialize_quantization_ranges',
    'create_quantized_lwdetr_model',
    'save_quantized_model',
    'load_quantized_model',
    
    # Benchmarking
    'QuantizationBenchmark',
    'benchmark_quantization',
]
