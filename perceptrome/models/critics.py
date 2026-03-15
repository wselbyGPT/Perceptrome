from typing import Dict

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class CriticHeads(nn.Module):  # type: ignore[misc]
    def __init__(self, hidden_dim: int):
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.validity = nn.Linear(hidden_dim, 1)
        self.novelty = nn.Linear(hidden_dim, 1)
        self.gc_fraction = nn.Linear(hidden_dim, 1)

    def forward(self, fused_global: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        return {
            "validity": self.validity(fused_global).squeeze(-1),
            "novelty": self.novelty(fused_global).squeeze(-1),
            "gc_fraction": self.gc_fraction(fused_global).squeeze(-1),
        }
