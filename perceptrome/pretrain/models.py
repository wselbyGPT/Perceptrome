from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from perceptrome.models.ast_nodes import ASTNodeEncoder
from perceptrome.models.ast_rvnn import TreeRvnnEncoder
from perceptrome.models.fusion import PooledASTFusion
from .interfaces import EncoderOutput


def _first_present(mapping: Dict[str, "torch.Tensor"], *keys: str):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


@dataclass
class BackboneConfig:
    vocab_size: int
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    ast_node_type_vocab_size: int = 64
    ast_tree_layers: int = 2
    fusion_strategy: str = "pooled_fusion"


class SequenceBackbone(nn.Module):  # type: ignore[misc]
    """Default pretraining backbone for tokenized sequence inputs with an optional Bio-AST branch."""

    def __init__(self, cfg: BackboneConfig):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for SequenceBackbone")
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(int(cfg.vocab_size), int(cfg.hidden_size), padding_idx=0)
        layers = []
        for _ in range(max(1, int(cfg.num_layers))):
            layers.extend(
                [
                    nn.LayerNorm(int(cfg.hidden_size)),
                    nn.Linear(int(cfg.hidden_size), int(cfg.hidden_size)),
                    nn.GELU(),
                    nn.Dropout(float(cfg.dropout)),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.ast_node_encoder = ASTNodeEncoder(
            hidden_dim=int(cfg.hidden_size),
            node_type_vocab_size=int(cfg.ast_node_type_vocab_size),
            dropout=float(cfg.dropout),
        )
        self.ast_tree_encoder = TreeRvnnEncoder(hidden_dim=int(cfg.hidden_size), layers=int(cfg.ast_tree_layers), dropout=float(cfg.dropout))
        self.pooled_fusion = PooledASTFusion(hidden_dim=int(cfg.hidden_size), dropout=float(cfg.dropout))

    def get_hidden_size(self) -> int:
        return int(self.cfg.hidden_size)

    def _sequence_encode(self, batch: Dict[str, "torch.Tensor"]) -> EncoderOutput:
        input_ids = batch["input_ids"].long()
        mask = batch.get("input_ids_mask")
        if mask is None:
            mask = (input_ids != 0).to(input_ids.dtype)
        x = self.emb(input_ids)
        x = self.encoder(x)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return EncoderOutput(token_embeddings=x, pooled_embedding=pooled, valid_mask=mask, aux={"ast_used": False, "fusion_strategy": self.cfg.fusion_strategy})

    def encode(self, batch: Dict[str, "torch.Tensor"]) -> EncoderOutput:
        seq_out = self._sequence_encode(batch)
        ast_ids = _first_present(batch, "ast_node_type_ids", "node_type_ids")
        if ast_ids is None:
            return seq_out
        ast_nodes = self.ast_node_encoder(
            ast_ids.long(),
            coords=_first_present(batch, "ast_span_coords", "span_coords", "coords"),
            strand=_first_present(batch, "ast_strand", "strand"),
            frame=_first_present(batch, "ast_frame", "frame"),
        )
        ast_node_struct, ast_root = self.ast_tree_encoder(
            ast_nodes,
            tree={
                "edge_index": _first_present(batch, "ast_tree_edge_index", "tree_edge_index", "edge_index"),
                "node_mask": _first_present(batch, "ast_node_mask", "node_mask"),
            },
        )
        fused_token, fused_global = self.pooled_fusion(
            seq_out.token_embeddings,
            seq_out.pooled_embedding,
            ast_node_struct,
            ast_root,
            node_mask=_first_present(batch, "ast_node_mask", "node_mask"),
        )
        aux = dict(seq_out.aux)
        aux.update({"ast_used": True, "fusion_strategy": self.cfg.fusion_strategy, "ast_node_embeddings": ast_node_struct, "ast_pooled_embedding": ast_root})
        return EncoderOutput(token_embeddings=fused_token, pooled_embedding=fused_global, valid_mask=seq_out.valid_mask, aux=aux)
