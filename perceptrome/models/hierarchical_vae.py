from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from .cnn_local import LocalCNNEncoder
from .ast_nodes import ASTNodeEncoder
from .ast_rvnn import TreeRvnnEncoder
from .fusion import PooledASTFusion
from .critics import CriticHeads


def _first_present(mapping: Dict[str, "torch.Tensor"], *keys: str):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


@dataclass
class HierarchicalOutput:
    recon_logits: "torch.Tensor"
    mu: "torch.Tensor"
    logvar: "torch.Tensor"
    fused_token: "torch.Tensor"
    fused_global: "torch.Tensor"
    critics: Dict[str, "torch.Tensor"]
    ast_used: bool


class HierarchicalVAE(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        hidden_dim: int,
        latent_dim: int,
        ast_tree_layers: int = 2,
        ast_node_type_vocab_size: int = 64,
        dropout: float = 0.1,
        ablation_mode: str = "pooled_fusion",
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.ablation_mode = str(ablation_mode).lower()

        self.seq_encoder = LocalCNNEncoder(vocab_size, hidden_dim, channels=hidden_dim, dropout=dropout)
        self.ast_node_encoder = ASTNodeEncoder(hidden_dim=hidden_dim, node_type_vocab_size=ast_node_type_vocab_size, dropout=dropout)
        self.ast_tree_encoder = TreeRvnnEncoder(hidden_dim=hidden_dim, layers=ast_tree_layers, dropout=dropout)
        self.pooled_fusion = PooledASTFusion(hidden_dim=hidden_dim, dropout=dropout)

        self.fc_mu = nn.Linear(hidden_dim, int(latent_dim))
        self.fc_logvar = nn.Linear(hidden_dim, int(latent_dim))
        self.dec1 = nn.Linear(int(latent_dim), hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, self.seq_len * self.vocab_size)
        self.critics = CriticHeads(hidden_dim=hidden_dim)

    def encode_hierarchical(self, x_seq: "torch.Tensor", ast_batch: Optional[Dict[str, "torch.Tensor"]] = None) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", Dict[str, "torch.Tensor"], bool]:
        seq_token, seq_global = self.seq_encoder(x_seq)
        ast_used = ast_batch is not None and ast_batch.get("node_type_ids") is not None

        mode = self.ablation_mode
        if mode == "ast_only" and not ast_used:
            raise ValueError("ast_only ablation requires ast_batch inputs")

        if not ast_used:
            fused_token, fused_global = seq_token, seq_global
        else:
            ast_nodes = self.ast_node_encoder(
                ast_batch["node_type_ids"],
                coords=_first_present(ast_batch, "coords", "span_coords"),
                strand=_first_present(ast_batch, "strand"),
                frame=_first_present(ast_batch, "frame"),
            )
            ast_node_struct, ast_root = self.ast_tree_encoder(
                ast_nodes,
                tree={
                    "edge_index": _first_present(ast_batch, "edge_index", "tree_edge_index"),
                    "node_mask": _first_present(ast_batch, "node_mask"),
                },
            )
            if mode == "cnn_only":
                fused_token, fused_global = seq_token, seq_global
            elif mode == "ast_only":
                ast_summary = ast_root.unsqueeze(1).expand(-1, seq_token.size(1), -1)
                fused_token, fused_global = ast_summary, ast_root
            else:
                fused_token, fused_global = self.pooled_fusion(
                    seq_token,
                    seq_global,
                    ast_node_struct,
                    ast_root,
                    node_mask=_first_present(ast_batch, "node_mask"),
                )

        mu = self.fc_mu(fused_global)
        logvar = self.fc_logvar(fused_global)
        return mu, logvar, fused_token, fused_global, self.critics(fused_global), ast_used

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h = F.gelu(self.dec1(z))
        return self.dec_out(h)

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        x_seq = x.view(x.size(0), self.seq_len, self.vocab_size)
        mu, logvar, *_ = self.encode_hierarchical(x_seq)
        return mu, logvar

    def forward_with_aux(self, x: "torch.Tensor", ast_batch: Optional[Dict[str, "torch.Tensor"]] = None) -> HierarchicalOutput:
        x_seq = x.view(x.size(0), self.seq_len, self.vocab_size)
        mu, logvar, fused_token, fused_global, critic_outputs, ast_used = self.encode_hierarchical(x_seq, ast_batch=ast_batch)
        z = self.reparameterize(mu, logvar)
        return HierarchicalOutput(self.decode(z), mu, logvar, fused_token, fused_global, critic_outputs, ast_used)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        out = self.forward_with_aux(x)
        return out.recon_logits, out.mu, out.logvar
