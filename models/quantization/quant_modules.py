"""
Quantized modules for LW-DETR with distribution tracking for benchmarking.
Adapted from I-ViT quantization approach with INT8 inference support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


class SymmetricQuantFunction(Function):
    """Symmetric quantization function for INT8."""
    
    @staticmethod
    def forward(ctx, x, k, specified_scale, is_weight):
        scale = specified_scale
        zero_point = torch.tensor(0., device=x.device)
        
        n = 2 ** (k - 1) - 1
        new_quant_x = torch.round(x / scale) + zero_point
        new_quant_x = torch.clamp(new_quant_x, -n-1, n)
        
        ctx.scale = scale
        ctx.is_weight = is_weight
        return new_quant_x
    
    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        return grad_output.clone() / scale, None, None, None


def symmetric_linear_quantization_params(num_bits, min_val, max_val):
    """Compute symmetric quantization scaling factor."""
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        eps = torch.finfo(torch.float32).eps
        
        max_val_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
        scale = max_val_abs / float(n)
        scale = torch.clamp(scale, min=eps)
    
    return scale


class QuantizedLinear(nn.Module):
    """Quantized linear layer with INT8 support and distribution tracking."""
    
    def __init__(self, in_features, out_features, bias=True, 
                 weight_bit=8, bias_bit=32, per_channel=True,
                 quant_mode='symmetric', track_distributions=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.per_channel = per_channel
        self.quant_mode = quant_mode
        self.track_distributions = track_distributions
        
        # Original parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Quantization buffers
        self.register_buffer('weight_scale', torch.ones(out_features))
        self.register_buffer('act_scale', torch.ones(1))
        
        # Distribution tracking
        if track_distributions:
            self.distributions = {
                'weight_fp32': [],
                'weight_quant': [],
                'activation_fp32': [],
                'activation_quant': []
            }
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x, prev_act_scaling_factor=None):
        # Quantize weights
        with torch.no_grad():
            if self.per_channel:
                w_reshaped = self.weight.reshape(self.weight.shape[0], -1)
                w_min = w_reshaped.min(dim=1)[0]
                w_max = w_reshaped.max(dim=1)[0]
            else:
                w_min = self.weight.min()
                w_max = self.weight.max()
            
            self.weight_scale = symmetric_linear_quantization_params(
                self.weight_bit, w_min, w_max)
        
        # Track distributions if enabled
        if self.track_distributions:
            self.distributions['weight_fp32'].append(self.weight.detach().cpu().numpy())
        
        # Quantize weight
        w_quant = SymmetricQuantFunction.apply(
            self.weight, self.weight_bit, self.weight_scale, True)
        
        if self.track_distributions:
            self.distributions['weight_quant'].append(w_quant.detach().cpu().numpy())
        
        # Handle activations
        if prev_act_scaling_factor is None:
            # First layer - quantize input
            with torch.no_grad():
                if len(x.shape) == 2:
                    x_min = x.min()
                    x_max = x.max()
                else:
                    x_flat = x.reshape(-1, x.shape[-1])
                    x_min = x_flat.min(dim=0)[0].min()
                    x_max = x_flat.max(dim=0)[0].max()
                
                self.act_scale = symmetric_linear_quantization_params(
                    8, x_min, x_max)
            
            x_quant = SymmetricQuantFunction.apply(
                x, 8, self.act_scale, False)
            bias_scale = self.weight_scale * self.act_scale
            
            if self.track_distributions:
                self.distributions['activation_fp32'].append(x.detach().cpu().numpy())
                self.distributions['activation_quant'].append(x_quant.detach().cpu().numpy())
        else:
            x_quant = x / prev_act_scaling_factor
            bias_scale = self.weight_scale * prev_act_scaling_factor
        
        # Compute output
        output = F.linear(x_quant, w_quant, None)
        
        # Quantize bias if present
        if self.bias is not None:
            if prev_act_scaling_factor is not None:
                bias_scale = self.weight_scale * prev_act_scaling_factor
            bias_quant = (self.bias / bias_scale).round().clamp(-128, 127)
            output = output + bias_quant
        
        output = output * bias_scale
        
        if self.track_distributions:
            self.distributions['activation_fp32'].append(output.detach().cpu().numpy())
            self.distributions['activation_quant'].append(output.detach().cpu().numpy() / bias_scale.item())
        
        return output, bias_scale


class QuantizedAct(nn.Module):
    """Quantized activation with INT8 support and distribution tracking."""
    
    def __init__(self, activation_bit=8, act_range_momentum=0.95,
                 running_stat=True, quant_mode='symmetric', 
                 track_distributions=False):
        super().__init__()
        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.track_distributions = track_distributions
        
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('act_scaling_factor', torch.zeros(1))
        
        if track_distributions:
            self.distributions = {
                'fp32': [],
                'quant': []
            }
    
    def forward(self, x, prev_act_scaling_factor=None):
        # Compute scaling factor
        with torch.no_grad():
            if self.running_stat:
                if len(x.shape) == 2:
                    cur_min = x.min()
                    cur_max = x.max()
                else:
                    cur_min = x.min()
                    cur_max = x.max()
                
                if self.min_val.sum() == 0 and self.max_val.sum() == 0:
                    self.min_val = cur_min.clone()
                    self.max_val = cur_max.clone()
                else:
                    self.min_val = self.min_val * self.act_range_momentum + \
                                   cur_min * (1 - self.act_range_momentum)
                    self.max_val = self.max_val * self.act_range_momentum + \
                                   cur_max * (1 - self.act_range_momentum)
            
            self.act_scaling_factor = symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val)
        
        if self.track_distributions:
            self.distributions['fp32'].append(x.detach().cpu().numpy())
        
        # Quantize
        x_quant = SymmetricQuantFunction.apply(
            x, self.activation_bit, self.act_scaling_factor, False)
        
        if self.track_distributions:
            self.distributions['quant'].append(x_quant.detach().cpu().numpy())
        
        return x_quant * self.act_scaling_factor, self.act_scaling_factor


class QuantizedMatMul(nn.Module):
    """Quantized matrix multiplication with distribution tracking."""
    
    def __init__(self, track_distributions=False):
        super().__init__()
        self.track_distributions = track_distributions
        self.register_buffer('act_scaling_factor', torch.ones(1))
        
        if track_distributions:
            self.distributions = {
                'output_fp32': [],
                'output_quant': []
            }
    
    def forward(self, A, scale_A, B, scale_B):
        # Quantized matrix multiplication
        A_int = A / scale_A
        B_int = B / scale_B
        
        output_scale = scale_A * scale_B
        output = (A_int @ B_int) * output_scale
        
        if self.track_distributions:
            self.distributions['output_fp32'].append(output.detach().cpu().numpy())
        
        self.act_scaling_factor = output_scale
        return output, output_scale


class QuantizedLayerNorm(nn.Module):
    """INT8-friendly LayerNorm (typically kept in FP32 or light quantization)."""
    
    def __init__(self, normalized_shape, eps=1e-5, track_distributions=False):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.track_distributions = track_distributions
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        if track_distributions:
            self.distributions = {
                'output': []
            }
    
    def forward(self, x):
        output = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        
        if self.track_distributions:
            self.distributions['output'].append(output.detach().cpu().numpy())
        
        return output, None  # Return scale as None (kept in FP32)


class QuantizedReLU(nn.Module):
    """Quantized ReLU with distribution tracking."""
    
    def __init__(self, track_distributions=False):
        super().__init__()
        self.track_distributions = track_distributions
        self.register_buffer('act_scaling_factor', torch.ones(1))
        
        if track_distributions:
            self.distributions = {
                'output': []
            }
    
    def forward(self, x, prev_scale=None):
        output = F.relu(x)
        
        if self.track_distributions:
            self.distributions['output'].append(output.detach().cpu().numpy())
        
        return output, prev_scale


class QuantizedGELU(nn.Module):
    """Quantized GELU approximation with distribution tracking."""
    
    def __init__(self, track_distributions=False):
        super().__init__()
        self.track_distributions = track_distributions
        
        if track_distributions:
            self.distributions = {
                'output': []
            }
    
    def forward(self, x, prev_scale=None):
        # Use approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / np.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
        output = x * cdf
        
        if self.track_distributions:
            self.distributions['output'].append(output.detach().cpu().numpy())
        
        return output, prev_scale
