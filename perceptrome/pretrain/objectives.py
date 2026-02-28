from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from .interfaces import EncoderOutput


@dataclass
class ObjectiveWeights:
    mlm: float = 1.0
    sme: float = 1.0
    contrastive: float = 1.0


class MaskedTokenObjective(nn.Module):  # type: ignore[misc]
    name = "masked_token"

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(int(hidden_size), int(vocab_size))

    def forward(self, enc: EncoderOutput, batch: Dict[str, "torch.Tensor"]) -> Tuple["torch.Tensor", Dict[str, float]]:
        logits = self.head(enc.token_embeddings)
        labels = batch["mlm_labels"].long()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        with torch.no_grad():
            mask = labels != -100
            preds = logits.argmax(dim=-1)
            correct = ((preds == labels) & mask).sum().item()
            total = mask.sum().item() or 1
        return loss, {"mlm_acc": float(correct / total)}


class MaskedSMEObjective(nn.Module):  # type: ignore[misc]
    name = "masked_sme"

    def __init__(self, hidden_size: int, num_s: int = 8, num_m: int = 64):
        super().__init__()
        self.s_head = nn.Linear(int(hidden_size), int(num_s))
        self.m_head = nn.Linear(int(hidden_size), int(num_m))
        self.e_head = nn.Linear(int(hidden_size), 1)

    def forward(self, enc: EncoderOutput, batch: Dict[str, "torch.Tensor"]) -> Tuple["torch.Tensor", Dict[str, float]]:
        token_h = enc.token_embeddings
        s_logits = self.s_head(token_h)
        m_logits = self.m_head(token_h)
        e_pred = self.e_head(token_h).squeeze(-1)

        s_labels = batch["sme_s_labels"].long()
        m_labels = batch["sme_m_labels"].long()
        e_labels = batch["sme_e_labels"].float()
        s_mask = batch["sme_s_mask"].bool()
        m_mask = batch["sme_m_mask"].bool()
        e_mask = batch["sme_e_mask"].bool()

        s_loss = F.cross_entropy(s_logits[s_mask], s_labels[s_mask]) if bool(s_mask.any()) else token_h.new_tensor(0.0)
        m_loss = F.cross_entropy(m_logits[m_mask], m_labels[m_mask]) if bool(m_mask.any()) else token_h.new_tensor(0.0)
        e_loss = F.smooth_l1_loss(e_pred[e_mask], e_labels[e_mask]) if bool(e_mask.any()) else token_h.new_tensor(0.0)
        loss = s_loss + m_loss + e_loss
        return loss, {"s_loss": float(s_loss.detach()), "m_loss": float(m_loss.detach()), "e_loss": float(e_loss.detach())}


class ContrastiveObjective(nn.Module):  # type: ignore[misc]
    name = "contrastive"

    def __init__(self, hidden_size: int, temperature: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(int(hidden_size), int(hidden_size))
        self.temperature = float(temperature)

    def forward(self, enc: EncoderOutput, batch: Dict[str, "torch.Tensor"]) -> Tuple["torch.Tensor", Dict[str, float]]:
        z = F.normalize(self.proj(enc.pooled_embedding), dim=-1)
        z2 = F.normalize(batch["contrastive_pos"], dim=-1)
        logits = (z @ z2.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        return loss, {"contrastive_acc": float(acc)}
