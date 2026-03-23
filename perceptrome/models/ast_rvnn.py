from typing import Dict, List, Optional, Tuple

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
        bsz, n_nodes, _ = x.shape
        src = edge_index[0].long().clamp(min=0, max=n_nodes - 1)
        dst = edge_index[1].long().clamp(min=0, max=n_nodes - 1)
        agg = torch.zeros_like(x)
        for b in range(bsz):
            agg[b].index_add_(0, dst, x[b, src, :])
        return agg

    def _normalize_edge_index(self, edge_index: Optional[object], batch_size: int) -> List[Optional["torch.Tensor"]]:
        if edge_index is None:
            return [None] * batch_size
        if isinstance(edge_index, list):
            return [item if item is not None else None for item in edge_index]
        return [edge_index for _ in range(batch_size)]

    def forward(self, node_embeddings: "torch.Tensor", tree: Optional[Dict[str, "torch.Tensor"]] = None) -> Tuple["torch.Tensor", "torch.Tensor"]:
        x = node_embeddings
        edge_index = None if tree is None else tree.get("edge_index")
        node_mask = None if tree is None else tree.get("node_mask")
        edges_per_sample = self._normalize_edge_index(edge_index, x.size(0))
        for layer in self.compose:
            agg = torch.zeros_like(x)
            for b, sample_edges in enumerate(edges_per_sample):
                if sample_edges is None or sample_edges.numel() == 0:
                    agg[b] = x[b]
                else:
                    agg[b : b + 1] = self._aggregate_edges(x[b : b + 1], sample_edges)
            x = x + layer(torch.cat([x, agg], dim=-1))
            if node_mask is not None:
                x = x * node_mask.unsqueeze(-1).to(dtype=x.dtype)
        if node_mask is None:
            root = x[:, 0, :] if x.size(1) > 0 else x.mean(dim=1)
        else:
            denom = node_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=x.dtype)
            root = (x * node_mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1) / denom
        return x, root
