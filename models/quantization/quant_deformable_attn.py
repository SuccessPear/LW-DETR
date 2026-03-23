"""
Quantized Deformable Attention Module for LW-DETR with INT8 support.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from .quant_modules import QuantizedLinear, QuantizedAct, QuantizedMatMul


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class QuantizedMSDeformAttn(nn.Module):
    """
    Quantized Multi-Scale Deformable Attention Module with INT8 support.
    Adapts the original MSDeformAttn from LW-DETR for quantization.
    """
    
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4,
                 track_distributions=False):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.track_distributions = track_distributions
        
        # Quantized linear layers for offset and attention weight computation
        self.sampling_offsets = QuantizedLinear(
            d_model, n_heads * n_levels * n_points * 2,
            bias=True, track_distributions=track_distributions)
        
        self.attention_weights = QuantizedLinear(
            d_model, n_heads * n_levels * n_points,
            bias=True, track_distributions=track_distributions)
        
        # Value projection
        self.value_proj = QuantizedLinear(
            d_model, d_model, bias=True,
            track_distributions=track_distributions)
        
        # Output projection
        self.output_proj = QuantizedLinear(
            d_model, d_model, bias=True,
            track_distributions=track_distributions)
        
        # Quantized operations
        self.matmul = QuantizedMatMul(track_distributions=track_distributions)
        self.act_sampling = QuantizedAct(track_distributions=track_distributions)
        self.act_weights = QuantizedAct(track_distributions=track_distributions)
        self.act_output = QuantizedAct(track_distributions=track_distributions)
        
        # Distribution tracking
        if track_distributions:
            self.distributions = {
                'sampling_offsets': [],
                'attention_weights': [],
                'attention_weights_softmax': [],
                'output': []
            }
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize sampling offsets
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.data = grid_init.view(-1)
        
        # Initialize attention weights
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        
        # Initialize projections
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
    
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None, prev_act_scale=None):
        """
        :param query: (N, Length_{query}, C)
        :param reference_points: (N, Length_{query}, n_levels, 2)
        :param input_flatten: (N, \\sum{H_i \\times W_i}, C)
        :param input_spatial_shapes: (n_levels, 2)
        :param input_level_start_index: (n_levels)
        :param input_padding_mask: (N, \\sum{H_i \\times W_i})
        :return output: (N, Length_{query}, C)
        """
        
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        
        # Project value with quantization
        value, value_scale = self.value_proj(input_flatten, prev_act_scale)
        
        # Compute sampling offsets (keep in higher precision)
        sampling_offsets, offset_scale = self.sampling_offsets(query, prev_act_scale)
        
        if self.track_distributions:
            self.distributions['sampling_offsets'].append(
                sampling_offsets.detach().cpu().numpy())
        
        # Reshape offsets: (N, Len_q, n_heads, n_levels, n_points, 2)
        sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        
        # Compute attention weights with quantization
        attention_weights, weights_scale = self.attention_weights(query, prev_act_scale)
        
        if self.track_distributions:
            self.distributions['attention_weights'].append(
                attention_weights.detach().cpu().numpy())
        
        # Reshape weights: (N, Len_q, n_heads, n_levels, n_points)
        attention_weights = attention_weights.view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        # Apply softmax for normalization (typically in FP32)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        if self.track_distributions:
            self.distributions['attention_weights_softmax'].append(
                attention_weights.detach().cpu().numpy())
        
        # Quantize normalized weights
        attention_weights, weights_scale = self.act_weights(attention_weights, weights_scale)
        
        # Compute sampling locations (in normalized coordinates)
        reference_points_level = reference_points[:, :, :, None, :]  # (N, Len_q, n_levels, 1, 2)
        sampling_locations = reference_points_level + sampling_offsets  # (N, Len_q, n_heads, n_levels, n_points, 2)
        
        # Bilinear sampling (simplified for quantization)
        # In practice, you would quantize the interpolation factor
        output = self._ms_deform_attn_pytorch(
            value, input_spatial_shapes, input_level_start_index,
            sampling_locations, attention_weights, input_padding_mask)
        
        if self.track_distributions:
            self.distributions['output'].append(output.detach().cpu().numpy())
        
        # Output projection with quantization
        output, output_scale = self.output_proj(output, value_scale)
        
        return output, output_scale
    
    def _ms_deform_attn_pytorch(self, value, value_spatial_shapes, value_level_start_index,
                                sampling_locations, attention_weights, input_padding_mask=None):
        """
        PyTorch implementation of multi-scale deformable attention (compatible with quantization).
        
        :param value: (N, \\sum{HiWi}, C)
        :param value_spatial_shapes: (n_levels, 2)
        :param value_level_start_index: (n_levels,)
        :param sampling_locations: (N, Len_q, n_heads, n_levels, n_points, 2)
        :param attention_weights: (N, Len_q, n_heads, n_levels, n_points)
        """
        
        N, _, C = value.shape
        _, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
        
        value = value.view(N, -1, n_heads, C // n_heads)
        
        sampling_locations = sampling_locations.view(N, Len_q, n_heads, n_levels * n_points, 2)
        attention_weights = attention_weights.view(N, Len_q, n_heads, n_levels * n_points)
        
        output = torch.zeros_like(value)
        
        for l_idx in range(n_levels):
            start_idx = value_level_start_index[l_idx].item()
            h, w = value_spatial_shapes[l_idx]
            
            locations = sampling_locations[:, :, :, l_idx * n_points:(l_idx + 1) * n_points, :]
            weights = attention_weights[:, :, :, l_idx * n_points:(l_idx + 1) * n_points]
            
            # Get feature map for this level
            feature_map = value[:, start_idx:start_idx + h * w, :, :].view(N, h, w, n_heads, C // n_heads)
            
            # Compute bilinear interpolated output for this level
            level_output = self._bilinear_interpolate(
                feature_map, locations, weights, N, Len_q, n_heads, C // n_heads)
            
            output = output + level_output
        
        return output.view(N, Len_q, C)
    
    def _bilinear_interpolate(self, feature_map, locations, weights, N, Len_q, n_heads, head_dim):
        """
        Bilinear interpolation for deformable attention sampling.
        Compatible with INT8 inference.
        """
        _, H, W, _, _ = feature_map.shape
        n_points = locations.shape[-2]
        
        # Normalize locations to [0, H-1] and [0, W-1]
        x = locations[..., 0] * (W - 1)
        y = locations[..., 1] * (H - 1)
        
        # Get integer and fractional parts
        x0 = torch.floor(x).long().clamp(0, W - 1)
        x1 = torch.floor(x).long().clamp(0, W - 1) + 1
        x1 = x1.clamp(0, W - 1)
        
        y0 = torch.floor(y).long().clamp(0, H - 1)
        y1 = torch.floor(y).long().clamp(0, H - 1) + 1
        y1 = y1.clamp(0, H - 1)
        
        # Bilinear weights
        wx = x - x0.float()
        wy = y - y0.float()
        
        # Gather and interpolate
        v00 = feature_map[:, y0, x0]  # (N, Len_q, n_heads, n_points, head_dim)
        v01 = feature_map[:, y0, x1]
        v10 = feature_map[:, y1, x0]
        v11 = feature_map[:, y1, x1]
        
        # Bilinear interpolation
        v = (v00 * (1 - wx[..., None]) * (1 - wy[..., None]) +
             v01 * wx[..., None] * (1 - wy[..., None]) +
             v10 * (1 - wx[..., None]) * wy[..., None] +
             v11 * wx[..., None] * wy[..., None])  # (N, Len_q, n_heads, n_points, head_dim)
        
        # Apply attention weights
        output = (v * weights[..., None]).sum(dim=-2)  # (N, Len_q, n_heads, head_dim)
        
        return output


class QuantizedDeformableAttention(nn.Module):
    """
    Alternative simplified quantized deformable attention that's easier to convert.
    Can be used as a drop-in replacement for standard attention with location bias.
    """
    
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4,
                 track_distributions=False):
        super().__init__()
        
        # Use regular quantized linear layers with position bias
        self.query_proj = QuantizedLinear(d_model, d_model, bias=True,
                                          track_distributions=track_distributions)
        self.key_proj = QuantizedLinear(d_model, d_model, bias=True,
                                        track_distributions=track_distributions)
        self.value_proj = QuantizedLinear(d_model, d_model, bias=True,
                                          track_distributions=track_distributions)
        self.output_proj = QuantizedLinear(d_model, d_model, bias=True,
                                           track_distributions=track_distributions)
        
        # Deformable position embeddings
        self.register_buffer('position_embeddings', None)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.matmul = QuantizedMatMul(track_distributions=track_distributions)
        
        if track_distributions:
            self.distributions = {
                'attn_scores': [],
                'attn_weights': [],
                'output': []
            }
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        """Forward pass with learnable deformable offsets."""
        
        N, Len_q, C = query.shape
        _, Len_k, _ = key.shape
        
        # Project Q, K, V
        q, q_scale = self.query_proj(query, None)
        k, k_scale = self.key_proj(key, None)
        v, v_scale = self.value_proj(value, None)
        
        # Reshape for multihead
        q = q.view(N, Len_q, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(N, Len_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(N, Len_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        scores, scores_scale = self.matmul(q, q_scale, k.transpose(-2, -1), k_scale)
        scores = scores / math.sqrt(self.head_dim)
        
        if self.track_distributions:
            self.distributions['attn_scores'].append(scores.detach().cpu().numpy())
        
        # Apply mask if provided
        if key_padding_mask is not None:
            scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # Softmax (kept in higher precision)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        weights = weights.to(scores.dtype)
        
        if self.track_distributions:
            self.distributions['attn_weights'].append(weights.detach().cpu().numpy())
        
        # Apply weights to values
        output, output_scale = self.matmul(weights, scores_scale, v, v_scale)
        
        # Reshape back
        output = output.permute(0, 2, 1, 3).contiguous().view(N, Len_q, C)
        
        # Output projection
        output, out_scale = self.output_proj(output, output_scale)
        
        if self.track_distributions:
            self.distributions['output'].append(output.detach().cpu().numpy())
        
        return output, None if not need_weights else weights.mean(dim=1)
