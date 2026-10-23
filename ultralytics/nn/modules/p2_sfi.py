# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pure spatial P2 downsampling for the controlled R18 SFI experiments."""

import torch
import torch.nn as nn

from .conv import Conv


class SpaceToDepth2(nn.Module):
    """Preserve every pixel using SO-DETR's four-phase channel order."""

    def forward(self, x):
        """Map BCHW to B(4C)(H/2)(W/2); require even spatial dimensions."""
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("SpaceToDepth2 requires even height and width")
        return torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2],
                          x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1)


class SPDConv(nn.Module):
    """Space-to-depth followed by a standard 3x3 Conv/BN/SiLU."""

    def __init__(self, c1, c2):
        super().__init__()
        self.rearrange = SpaceToDepth2()
        self.conv = Conv(4 * c1, c2, 3)

    def forward(self, x):
        return self.conv(self.rearrange(x))


class SPDLite(nn.Module):
    """Space-to-depth, 1x1 reduction, depthwise 3x3, then 1x1 expansion."""

    def __init__(self, c1, c2, hidden=64):
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.rearrange = SpaceToDepth2()
        self.reduce = Conv(4 * c1, hidden, 1)
        self.depthwise = Conv(hidden, hidden, 3, g=hidden)
        self.expand = Conv(hidden, c2, 1)

    def forward(self, x):
        return self.expand(self.depthwise(self.reduce(self.rearrange(x))))
