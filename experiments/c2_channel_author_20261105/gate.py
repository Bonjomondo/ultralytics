"""Static, identity-initialized P2 scalar/channel calibration before C2 concat."""
import torch
from torch import nn


class P2GateConcat(nn.Module):
    def __init__(self, dimension=1, channels=128, mode="channel"):
        super().__init__()
        if mode not in {"scalar", "channel"}:
            raise ValueError(mode)
        self.dimension, self.channels, self.mode = dimension, channels, mode
        self.logits = nn.Parameter(torch.zeros(1 if mode == "scalar" else channels))

    def alpha(self):
        return 2 * self.logits.sigmoid()

    def forward(self, xs):
        assert xs[1].shape[1] == self.channels
        return torch.cat([xs[0], xs[1] * self.alpha().view(1, -1, 1, 1), xs[2]], self.dimension)
