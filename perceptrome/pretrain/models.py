from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from .interfaces import EncoderOutput


@dataclass
class BackboneConfig:
    vocab_size: int
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1


class SequenceBackbone(nn.Module):  # type: ignore[misc]
    """Default pretraining backbone for tokenized sequence inputs."""

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

    def get_hidden_size(self) -> int:
        return int(self.cfg.hidden_size)

    def encode(self, batch: Dict[str, "torch.Tensor"]) -> EncoderOutput:
        input_ids = batch["input_ids"].long()
        mask = batch.get("input_ids_mask")
        if mask is None:
            mask = (input_ids != 0).to(input_ids.dtype)
        x = self.emb(input_ids)
        x = self.encoder(x)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return EncoderOutput(token_embeddings=x, pooled_embedding=pooled, valid_mask=mask)
