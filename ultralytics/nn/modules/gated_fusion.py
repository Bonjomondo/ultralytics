"""Identity-initialized alternatives to C2's [P4up, P2down, P3] concat."""
import torch
from torch import nn


class C2GatedConcat(nn.Module):
    """Gate P2 only, or learn three branch weights; preserve C2 tensor layout."""

    def __init__(self, channels, mode="spatial", hidden=8):
        super().__init__()
        if len(channels) != 3 or mode not in {"scalar", "spatial", "fusion"}:
            raise ValueError("Expected three branches and scalar/spatial/fusion mode")
        self.mode = mode
        if mode == "spatial":
            self.gate = nn.Sequential(nn.Conv2d(6, hidden, 1), nn.SiLU(),
                                      nn.Conv2d(hidden, 1, 3, padding=1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        else:
            self.logits = nn.Parameter(torch.zeros(3 if mode == "fusion" else 1))
        self.capture = False
        self.stats = {}

    def forward(self, xs):
        context, detail, original = xs
        if self.mode == "fusion":
            weights = 3 * self.logits.softmax(0)
            ys = [x * weights[i] for i, x in enumerate(xs)]
            alpha = weights[1]
        else:
            if self.mode == "spatial":
                descriptors = torch.cat([v for x in xs for v in
                                         (x.mean(1, keepdim=True), x.amax(1, keepdim=True))], 1)
                alpha = 2 * self.gate(descriptors).sigmoid()
            else:
                alpha = 2 * self.logits.sigmoid()
            ys = [context, detail * alpha, original]
        if self.capture:
            a = alpha.detach().float()
            self.stats = {"gate_mean": a.mean(), "gate_std": a.std(unbiased=False),
                          "gate_low_fraction": (a < .1).float().mean(),
                          "gate_high_fraction": (a > 1.9).float().mean()}
            if self.mode == "fusion":
                self.stats.update({f"branch_weight_{i}": w.detach() for i, w in enumerate(weights)})
        return torch.cat(ys, 1)
