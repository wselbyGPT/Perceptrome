from typing import Dict, Optional, Tuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class TreeRvnnEncoder(nn.Module):  # type: ignore[misc]
    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.1):
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.compose = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(float(dropout))) for _ in range(max(1, int(layers)))]
        )

    def _aggregate_edges(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
        bsz, n_nodes, h = x.shape
        src = edge_index[0].long().clamp(min=0, max=n_nodes - 1)
        dst = edge_index[1].long().clamp(min=0, max=n_nodes - 1)
        agg = torch.zeros_like(x)
        for b in range(bsz):
            agg[b].index_add_(0, dst, x[b, src, :])
        return agg

    def forward(self, node_embeddings: "torch.Tensor", tree: Optional[Dict[str, "torch.Tensor"]] = None) -> Tuple["torch.Tensor", "torch.Tensor"]:
        x = node_embeddings
        edge_index = None if tree is None else tree.get("edge_index")
        for layer in self.compose:
            if edge_index is None or edge_index.numel() == 0:
                agg = x
            else:
                agg = self._aggregate_edges(x, edge_index)
            x = x + layer(torch.cat([x, agg], dim=-1))
        root = x[:, 0, :] if x.size(1) > 0 else x.mean(dim=1)
        return x, root
