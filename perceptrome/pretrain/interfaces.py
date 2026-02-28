from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass
class EncoderOutput:
    token_embeddings: Any
    pooled_embedding: Any
    valid_mask: Any
    aux: Dict[str, Any] = field(default_factory=dict)


class PretrainBackbone(Protocol):
    def encode(self, batch: Dict[str, Any]) -> EncoderOutput:
        ...

    def get_hidden_size(self) -> int:
        ...


class PretrainObjective(Protocol):
    name: str

    def __call__(self, enc: EncoderOutput, batch: Dict[str, Any]):
        """Returns tuple(loss, metrics_dict)."""
        ...
