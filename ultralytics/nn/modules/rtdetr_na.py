# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""ResNet18-vd and a gated P4 refinement for the RT-DETR NA ablation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .transformer import LayerNorm2d

__all__ = ("NeighborhoodAttention2d", "ResNet18Stem", "ResNet18Stage", "ResidualLocalRefine")


class ResNet18Stem(nn.Module):
    """Three 3x3 convolutions followed by max pooling, as in RT-DETR's ResNet-vd."""

    def __init__(self, c1: int, c2: int = 64):
        """Map input images to stride-four features."""
        super().__init__()
        self.layers = nn.Sequential(
            Conv(c1, c2 // 2, 3, 2, act=nn.ReLU(inplace=True)),
            Conv(c2 // 2, c2 // 2, 3, 1, act=nn.ReLU(inplace=True)),
            Conv(c2 // 2, c2, 3, 1, act=nn.ReLU(inplace=True)),
            nn.MaxPool2d(3, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return stem features."""
        return self.layers(x)


class ResNet18BasicBlock(nn.Module):
    """Two 3x3 convolutions with the variant-d downsampling shortcut."""

    def __init__(self, c1: int, c2: int, stride: int = 1, project: bool = False):
        """Use average pooling before a stride-one projection when downsampling."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 3, stride, act=nn.ReLU(inplace=True))
        self.conv2 = Conv(c2, c2, 3, 1, act=False)
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(2, 2, ceil_mode=True), Conv(c1, c2, 1, 1, act=False))
        elif project or c1 != c2:
            self.shortcut = Conv(c1, c2, 1, 1, act=False)
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return residual features without channel expansion."""
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class ResNet18Stage(nn.Module):
    """A ResNet18 stage; four stages contain two basic blocks each."""

    def __init__(self, c1: int, c2: int, stride: int = 1, n: int = 2):
        """Project the first shortcut, matching the RT-DETR PResNet implementation."""
        super().__init__()
        self.blocks = nn.Sequential(
            ResNet18BasicBlock(c1, c2, stride, project=True),
            *(ResNet18BasicBlock(c2, c2) for _ in range(n - 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return stage features."""
        return self.blocks(x)


class NeighborhoodAttention2d(nn.Module):
    """Masked sliding-window attention; padded locations never participate in softmax."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        kernel_size: int = 5,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rpb: bool = False,
    ):
        """Optionally enable RPB for later experiments; R1 leaves it disabled."""
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim and num_heads must be positive, and dim must be divisible by num_heads")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        if use_rpb:
            self.relative_position_bias = nn.Parameter(torch.zeros(num_heads, kernel_size * kernel_size))
            nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        else:
            self.register_parameter("relative_position_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve BCHW shape and attend only to valid image neighbors."""
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, self.head_dim, h * w) * self.scale
        shape = (b, self.num_heads, self.head_dim, self.kernel_size**2, h * w)
        k = F.unfold(k, self.kernel_size, padding=self.padding).reshape(shape)
        v = F.unfold(v, self.kernel_size, padding=self.padding).reshape(shape)
        attn = torch.einsum("bhcn,bhckn->bhkn", q, k)
        if self.relative_position_bias is not None:
            attn = attn + self.relative_position_bias[None, :, :, None]
        if self.padding:
            valid = F.unfold(x.new_ones(1, 1, h, w), self.kernel_size, padding=self.padding)
            valid = valid.reshape(1, 1, self.kernel_size**2, h * w).bool()
            attn = attn.masked_fill(~valid, float("-inf"))
        attn = self.attn_drop(attn.softmax(dim=2))
        out = torch.einsum("bhkn,bhckn->bhcn", attn, v).reshape(b, c, h, w)
        return self.proj_drop(self.proj(out))


class ResidualLocalRefine(nn.Module):
    """P4 + gamma * project(NA(reduce(LayerNorm(P4)))); no FFN."""

    def __init__(
        self,
        c1: int,
        inner_dim: int = 128,
        num_heads: int = 4,
        kernel_size: int = 5,
        gamma_init: float = 0.0,
        use_rpb: bool = False,
    ):
        """Start as an exact identity and learn how much local refinement to use."""
        super().__init__()
        self.norm = LayerNorm2d(c1)
        self.reduce = nn.Conv2d(c1, inner_dim, 1, bias=False)
        self.attn = NeighborhoodAttention2d(inner_dim, num_heads, kernel_size, use_rpb=use_rpb)
        self.project = nn.Conv2d(inner_dim, c1, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated refinement without modifying the input tensor in place."""
        return x + self.gamma * self.project(self.attn(self.reduce(self.norm(x))))
