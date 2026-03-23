"""
Quantized transformer layers for LW-DETR with distribution tracking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .quant_modules import (
    QuantizedLinear, QuantizedAct, QuantizedMatMul, 
    QuantizedLayerNorm, QuantizedReLU, QuantizedGELU
)


class QuantizedMultiheadAttention(nn.Module):
    """Quantized multihead attention with INT8 support and distribution tracking."""
    
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False,
                 track_distributions=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.track_distributions = track_distributions
        
        assert self.head_dim * num_heads == self.embed_dim
        
        # Quantized projections
        self.q_proj = QuantizedLinear(embed_dim, embed_dim, bias=bias,
                                      track_distributions=track_distributions)
        self.k_proj = QuantizedLinear(self.kdim, embed_dim, bias=bias,
                                      track_distributions=track_distributions)
        self.v_proj = QuantizedLinear(self.vdim, embed_dim, bias=bias,
                                      track_distributions=track_distributions)
        
        # Quantized operations
        self.matmul_qk = QuantizedMatMul(track_distributions=track_distributions)
        self.matmul_attn_v = QuantizedMatMul(track_distributions=track_distributions)
        
        # Output projection
        self.out_proj = QuantizedLinear(embed_dim, embed_dim, bias=bias,
                                        track_distributions=track_distributions)
        
        # Activation quantization
        self.act_q = QuantizedAct(track_distributions=track_distributions)
        self.act_k = QuantizedAct(track_distributions=track_distributions)
        self.act_v = QuantizedAct(track_distributions=track_distributions)
        self.act_attn = QuantizedAct(track_distributions=track_distributions)
        self.act_out = QuantizedAct(track_distributions=track_distributions)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, query, key, value, key_padding_mask=None, 
                need_weights=False, attn_mask=None, average_attn_weights=True):
        """
        Forward pass for quantized multihead attention.
        Args simplified compared to standard attention for quantization.
        """
        bsz, tgt_len, embed_dim = query.shape
        src_len = key.shape[1]
        
        # Project Q, K, V with quantization
        q, q_scale = self.q_proj(query, None)
        k, k_scale = self.k_proj(key, None)
        v, v_scale = self.v_proj(value, None)
        
        # Reshape for multihead attention
        # (bsz, seq_len, embed_dim) -> (bsz * num_heads, seq_len, head_dim)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)
        
        # Compute attention scores: (Q @ K^T) / sqrt(d_k)
        # Using quantized matmul
        scores, scores_scale = self.matmul_qk(q, q_scale, k.transpose(-2, -1), k_scale)
        scores = scores * self.scale
        scores_scale = scores_scale * self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
            scores.masked_fill_(attn_mask, float('-inf'))
        
        if key_padding_mask is not None:
            scores = scores.view(bsz, self.num_heads, tgt_len, src_len)
            scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            scores = scores.view(bsz * self.num_heads, tgt_len, src_len)
        
        # Softmax (kept in higher precision)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(scores.dtype)
        attn_weights, attn_scale = self.act_attn(attn_weights, scores_scale)
        attn_weights = self.attn_drop(attn_weights)
        
        # Apply attention to values: Attn(Q,K,V) = softmax(Q@K^T/sqrt(d_k)) @ V
        attn_output, attn_output_scale = self.matmul_attn_v(
            attn_weights, attn_scale, v, v_scale)
        
        # Reshape back
        attn_output = attn_output.view(bsz, tgt_len, embed_dim)
        
        # Output projection with quantization
        attn_output, out_scale = self.out_proj(attn_output, attn_output_scale)
        attn_output = self.proj_drop(attn_output)
        
        if need_weights:
            return attn_output, attn_weights.view(bsz, self.num_heads, tgt_len, src_len).sum(dim=1) / self.num_heads
        else:
            return attn_output, None


class QuantizedMLP(nn.Module):
    """Quantized feedforward network (MLP) with INT8 support."""
    
    def __init__(self, in_features, hidden_features, out_features=None, 
                 activation='gelu', dropout=0., track_distributions=False):
        super().__init__()
        out_features = out_features or in_features
        
        self.fc1 = QuantizedLinear(in_features, hidden_features, bias=True,
                                   track_distributions=track_distributions)
        self.fc2 = QuantizedLinear(hidden_features, out_features, bias=True,
                                   track_distributions=track_distributions)
        
        if activation == 'gelu':
            self.act = QuantizedGELU(track_distributions=track_distributions)
        elif activation == 'relu':
            self.act = QuantizedReLU(track_distributions=track_distributions)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x, prev_scale=None):
        x, scale1 = self.fc1(x, prev_scale)
        x = self.drop1(x)
        x, _ = self.act(x, scale1)
        x, scale2 = self.fc2(x, scale1)
        x = self.drop2(x)
        return x, scale2


class QuantizedTransformerEncoderLayer(nn.Module):
    """Quantized transformer encoder layer with distribution tracking."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False,
                 track_distributions=False):
        super().__init__()
        self.self_attn = QuantizedMultiheadAttention(
            d_model, nhead, dropout=dropout,
            track_distributions=track_distributions)
        
        # MLP
        self.linear1 = QuantizedLinear(d_model, dim_feedforward, bias=True,
                                       track_distributions=track_distributions)
        self.linear2 = QuantizedLinear(dim_feedforward, d_model, bias=True,
                                       track_distributions=track_distributions)
        
        # Activation
        if activation == 'relu':
            self.activation = QuantizedReLU(track_distributions=track_distributions)
        elif activation == 'gelu':
            self.activation = QuantizedGELU(track_distributions=track_distributions)
        else:
            self.activation = QuantizedReLU(track_distributions=track_distributions)
        
        self.norm1 = QuantizedLayerNorm(d_model, track_distributions=track_distributions)
        self.norm2 = QuantizedLayerNorm(d_model, track_distributions=track_distributions)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.normalize_before = normalize_before
    
    def with_pos_embed(self, tensor, pos):
        return tensor + pos if pos is not None else tensor
    
    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        # Self attention block
        src2, _ = self.self_attn(
            self.with_pos_embed(src, pos),
            self.with_pos_embed(src, pos),
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        src = src + self.dropout1(src2)
        src, _ = self.norm1(src)
        
        # FFN block
        src2, _ = self.linear1(src, None)
        src2, _ = self.activation(src2, None)
        src2 = self.dropout2(src2)
        src2, _ = self.linear2(src2, None)
        src = src + self.dropout3(src2)
        src, _ = self.norm2(src)
        
        return src
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class QuantizedDecoderLayer(nn.Module):
    """Quantized transformer decoder layer with quantized attention."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False,
                 use_deformable_attn=False,
                 track_distributions=False):
        super().__init__()
        self.self_attn = QuantizedMultiheadAttention(
            d_model, nhead, dropout=dropout,
            track_distributions=track_distributions)
        
        self.cross_attn = QuantizedMultiheadAttention(
            d_model, nhead, dropout=dropout,
            track_distributions=track_distributions) if not use_deformable_attn else None
        
        # MLP
        self.linear1 = QuantizedLinear(d_model, dim_feedforward, bias=True,
                                       track_distributions=track_distributions)
        self.linear2 = QuantizedLinear(dim_feedforward, d_model, bias=True,
                                       track_distributions=track_distributions)
        
        if activation == 'relu':
            self.activation = QuantizedReLU(track_distributions=track_distributions)
        elif activation == 'gelu':
            self.activation = QuantizedGELU(track_distributions=track_distributions)
        else:
            self.activation = QuantizedReLU(track_distributions=track_distributions)
        
        self.norm1 = QuantizedLayerNorm(d_model, track_distributions=track_distributions)
        self.norm2 = QuantizedLayerNorm(d_model, track_distributions=track_distributions)
        self.norm3 = QuantizedLayerNorm(d_model, track_distributions=track_distributions)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        
        self.normalize_before = normalize_before
        self.use_deformable_attn = use_deformable_attn
    
    def with_pos_embed(self, tensor, pos):
        return tensor + pos if pos is not None else tensor
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        # Self attention on target
        tgt2, _ = self.self_attn(
            self.with_pos_embed(tgt, query_pos),
            self.with_pos_embed(tgt, query_pos),
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt, _ = self.norm1(tgt)
        
        # Cross attention
        if self.cross_attn is not None:
            tgt2, _ = self.cross_attn(
                self.with_pos_embed(tgt, query_pos),
                self.with_pos_embed(memory, pos),
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False
            )
            tgt = tgt + self.dropout2(tgt2)
            tgt, _ = self.norm2(tgt)
        
        # FFN
        tgt2, _ = self.linear1(tgt, None)
        tgt2, _ = self.activation(tgt2, None)
        tgt2 = self.dropout3(tgt2)
        tgt2, _ = self.linear2(tgt2, None)
        tgt = tgt + self.dropout4(tgt2)
        tgt, _ = self.norm3(tgt)
        
        return tgt
