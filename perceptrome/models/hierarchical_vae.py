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
from .fusion import CrossScaleFusion
from .critics import CriticHeads


@dataclass
class HierarchicalOutput:
    recon_logits: "torch.Tensor"
    mu: "torch.Tensor"
    logvar: "torch.Tensor"
    fused_token: "torch.Tensor"
    fused_global: "torch.Tensor"
    critics: Dict[str, "torch.Tensor"]


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
        ablation_mode: str = "hierarchical",
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
        self.fusion = CrossScaleFusion(hidden_dim=hidden_dim, dropout=dropout)

        self.fc_mu = nn.Linear(hidden_dim, int(latent_dim))
        self.fc_logvar = nn.Linear(hidden_dim, int(latent_dim))
        self.dec1 = nn.Linear(int(latent_dim), hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, self.seq_len * self.vocab_size)
        self.critics = CriticHeads(hidden_dim=hidden_dim)

    def _default_ast(self, x_seq: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        bsz, seq_len, _ = x_seq.shape
        node_count = max(2, min(16, seq_len // 8))
        device = x_seq.device
        node_type_ids = torch.zeros((bsz, node_count), dtype=torch.long, device=device)
        coords = torch.zeros((bsz, node_count, 2), dtype=x_seq.dtype, device=device)
        coords[..., 0] = torch.arange(node_count, device=device, dtype=x_seq.dtype)
        coords[..., 1] = coords[..., 0] + 1.0
        src = torch.arange(1, node_count, device=device, dtype=torch.long)
        dst = torch.arange(0, node_count - 1, device=device, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        return {"node_type_ids": node_type_ids, "coords": coords, "edge_index": edge_index}

    def encode_hierarchical(self, x_seq: "torch.Tensor", ast_batch: Optional[Dict[str, "torch.Tensor"]] = None) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", Dict[str, "torch.Tensor"]]:
        ast = ast_batch or self._default_ast(x_seq)
        seq_token, seq_global = self.seq_encoder(x_seq)
        ast_nodes = self.ast_node_encoder(ast["node_type_ids"], coords=ast.get("coords"), strand=ast.get("strand"))
        ast_node_struct, ast_root = self.ast_tree_encoder(ast_nodes, tree={"edge_index": ast.get("edge_index")})

        mode = self.ablation_mode
        if mode == "cnn_only":
            fused_token, fused_global = seq_token, seq_global
        elif mode == "ast_only":
            ast_summary = ast_node_struct.mean(dim=1, keepdim=True).expand(-1, seq_token.size(1), -1)
            fused_token, fused_global = ast_summary, ast_root
        else:
            fused_token, fused_global = self.fusion(seq_token, seq_global, ast_node_struct, ast_root)

        mu = self.fc_mu(fused_global)
        logvar = self.fc_logvar(fused_global)
        return mu, logvar, fused_token, fused_global, self.critics(fused_global)

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
        mu, logvar, fused_token, fused_global, critic_outputs = self.encode_hierarchical(x_seq, ast_batch=ast_batch)
        z = self.reparameterize(mu, logvar)
        return HierarchicalOutput(self.decode(z), mu, logvar, fused_token, fused_global, critic_outputs)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        out = self.forward_with_aux(x)
        return out.recon_logits, out.mu, out.logvar
