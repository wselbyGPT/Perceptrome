from typing import Optional

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class ASTNodeEncoder(nn.Module):  # type: ignore[misc]
    def __init__(self, hidden_dim: int, node_type_vocab_size: int = 64, dropout: float = 0.1):
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.type_embed = nn.Embedding(int(node_type_vocab_size), int(hidden_dim))
        self.coord_proj = nn.Linear(2, int(hidden_dim))
        self.strand_embed = nn.Embedding(3, int(hidden_dim))
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.drop = nn.Dropout(float(dropout))

    def forward(
        self,
        node_type_ids: "torch.Tensor",
        coords: Optional["torch.Tensor"] = None,
        strand: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        node_ids = node_type_ids.clamp(min=0, max=self.type_embed.num_embeddings - 1)
        x = self.type_embed(node_ids)
        if coords is not None:
            x = x + self.coord_proj(coords.to(dtype=x.dtype))
        if strand is not None:
            x = x + self.strand_embed((strand + 1).clamp(min=0, max=2))
        return self.drop(self.norm(x))
