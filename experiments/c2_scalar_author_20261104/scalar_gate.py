"""Scalar P2 gate: preserve [P4up,P2down,P3] concat and identity initialization."""
import torch
from torch import nn

class ScalarP2Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension
        self.logit = nn.Parameter(torch.zeros(()))

    def forward(self, xs):
        return torch.cat([xs[0], xs[1] * (2 * self.logit.sigmoid()), xs[2]], self.dimension)
