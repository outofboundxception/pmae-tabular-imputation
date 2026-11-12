"""Transformer and MLP-Mixer blocks used by the PMAE implementation.

This module intentionally implements the small subset of timm.layers used by
this project locally. That keeps the original model architecture usable without
requiring a runtime dependency on timm internals.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from einops import rearrange


class Mlp(nn.Module):
    """Two-layer feed-forward network, matching timm's Mlp defaults used here."""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class DropPath(nn.Module):
    """Per-sample stochastic depth. A no-op when drop_prob is zero."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep_prob


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_norm: bool = False, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, norm_layer=nn.LayerNorm,
                 fused: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = bool(fused and hasattr(F, "scaled_dot_product_attention"))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.save_attn = False

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
        else:
            weights = (q * self.scale) @ k.transpose(-2, -1)
            weights = self.attn_drop(weights.softmax(dim=-1))
            if self.save_attn:
                self.attn_res = weights
            x = weights @ v
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_norm: bool = False,
                 proj_drop: float = 0.0, attn_drop: float = 0.0,
                 init_values: Optional[float] = None, drop_path: float = 0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 mlp_layer=Mlp, fused: bool = True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_norm, attn_drop,
                              proj_drop, norm_layer, fused=fused)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Block_mlp(nn.Module):
    """MLP-Mixer block retained from the original PMAE implementation."""
    def __init__(self, d_dim: int, c_dim: int, block_config: int = 0,
                 mlp_ratio: float = 4.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, mlp_layer=Mlp, p_drop: float = 0.1):
        super().__init__()
        self.norm_c = norm_layer(c_dim)
        self.norm_d = norm_layer(d_dim)
        self.mlp_c = mlp_layer(c_dim, int(c_dim * mlp_ratio), act_layer=act_layer, drop=p_drop)
        self.mlp_d = mlp_layer(d_dim, int(d_dim * mlp_ratio), act_layer=act_layer, drop=p_drop)
        self.block_config = block_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_config == 0:
            x = rearrange(x, 'n d c -> n c d')
            x = x + self.mlp_d(self.norm_d(x))
            x = rearrange(x, 'n c d -> n d c')
            x = x + self.mlp_c(self.norm_c(x))
        elif self.block_config == 1:
            x = x + self.mlp_c(self.norm_c(x))
            x = rearrange(x, 'n d c -> n c d')
            x = x + self.mlp_d(self.norm_d(x))
            x = rearrange(x, 'n c d -> n d c')
        elif self.block_config == 2:
            x = rearrange(x, 'n d c -> n c d')
            x = x + self.mlp_d(self.norm_d(x))
            x = rearrange(x, 'n c d -> n d c')
        elif self.block_config == 3:
            x = x + self.mlp_c(self.norm_c(x))
        return x
