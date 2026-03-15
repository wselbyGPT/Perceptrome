from typing import Any

from .hierarchical_vae import HierarchicalVAE


def build_hierarchical_model(
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
    latent_dim: int,
    ast_tree_layers: int,
    ast_node_type_vocab_size: int,
    dropout: float,
    ablation_mode: str,
) -> Any:
    return HierarchicalVAE(
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        ast_tree_layers=ast_tree_layers,
        ast_node_type_vocab_size=ast_node_type_vocab_size,
        dropout=dropout,
        ablation_mode=ablation_mode,
    )
