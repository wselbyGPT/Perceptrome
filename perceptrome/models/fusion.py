from typing import Tuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class CrossScaleFusion(nn.Module):  # type: ignore[misc]
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.token_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.global_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        seq_token: "torch.Tensor",
        seq_global: "torch.Tensor",
        ast_node: "torch.Tensor",
        ast_global: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        ast_summary = ast_node.mean(dim=1, keepdim=True).expand(-1, seq_token.size(1), -1)
        t_gate = torch.sigmoid(self.token_gate(torch.cat([seq_token, ast_summary], dim=-1)))
        fused_token = self.dropout(t_gate * seq_token + (1.0 - t_gate) * ast_summary)

        g_gate = torch.sigmoid(self.global_gate(torch.cat([seq_global, ast_global], dim=-1)))
        fused_global = self.dropout(g_gate * seq_global + (1.0 - g_gate) * ast_global)
        return fused_token, fused_global


class PooledASTFusion(nn.Module):  # type: ignore[misc]
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.token_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.global_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        seq_token: "torch.Tensor",
        seq_global: "torch.Tensor",
        ast_node: "torch.Tensor",
        ast_global: "torch.Tensor",
        node_mask: "torch.Tensor | None" = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if node_mask is None:
            ast_summary = ast_node.mean(dim=1, keepdim=True)
        else:
            weights = node_mask.unsqueeze(-1).to(dtype=ast_node.dtype)
            denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            ast_summary = (ast_node * weights).sum(dim=1, keepdim=True) / denom
        ast_summary = ast_summary.expand(-1, seq_token.size(1), -1)
        fused_token = self.dropout(self.token_proj(torch.cat([seq_token, ast_summary], dim=-1)))
        fused_global = self.dropout(self.global_proj(torch.cat([seq_global, ast_global], dim=-1)))
        return fused_token, fused_global
