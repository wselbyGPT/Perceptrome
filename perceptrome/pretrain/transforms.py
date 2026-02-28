from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class MaskedLanguageModelTransform:
    mask_prob: float = 0.15
    mask_token_id: int = 0
    vocab_size: int = 32
    seed: int = 1337

    def __call__(self, input_ids: np.ndarray) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        input_ids = np.asarray(input_ids, dtype=np.int64)
        labels = np.full_like(input_ids, fill_value=-100)
        candidate_mask = rng.random(input_ids.shape) < float(self.mask_prob)
        labels[candidate_mask] = input_ids[candidate_mask]

        masked = input_ids.copy()
        replace_mask = candidate_mask & (rng.random(input_ids.shape) < 0.80)
        random_mask = candidate_mask & ~replace_mask & (rng.random(input_ids.shape) < 0.50)

        masked[replace_mask] = int(self.mask_token_id)
        if random_mask.any():
            masked[random_mask] = rng.integers(0, self.vocab_size, size=int(random_mask.sum()))

        return {"input_ids": masked, "mlm_labels": labels, "mlm_mask": candidate_mask.astype(np.int8)}


@dataclass
class MaskSMETransform:
    mask_prob_s: float = 0.2
    mask_prob_m: float = 0.2
    mask_prob_e: float = 0.2
    seed: int = 1337

    def __call__(self, sme_s: np.ndarray, sme_m: np.ndarray, sme_e: np.ndarray) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        sme_s = np.asarray(sme_s)
        sme_m = np.asarray(sme_m)
        sme_e = np.asarray(sme_e)

        mask_s = rng.random(sme_s.shape) < float(self.mask_prob_s)
        mask_m = rng.random(sme_m.shape) < float(self.mask_prob_m)
        mask_e = rng.random(sme_e.shape) < float(self.mask_prob_e)

        masked_s = sme_s.copy()
        masked_m = sme_m.copy()
        masked_e = sme_e.copy()

        if masked_s.dtype.kind in ("i", "u"):
            masked_s[mask_s] = -1
        else:
            masked_s[mask_s] = np.nan

        if masked_m.dtype.kind in ("i", "u"):
            masked_m[mask_m] = -1
        else:
            masked_m[mask_m] = np.nan

        masked_e[mask_e] = 0.0

        return {
            "sme_s": masked_s,
            "sme_m": masked_m,
            "sme_e": masked_e,
            "sme_s_labels": sme_s,
            "sme_m_labels": sme_m,
            "sme_e_labels": sme_e,
            "sme_s_mask": mask_s.astype(np.int8),
            "sme_m_mask": mask_m.astype(np.int8),
            "sme_e_mask": mask_e.astype(np.int8),
        }


@dataclass
class ContrastivePairTransform:
    crop_size: Optional[int] = None
    noise_std: float = 0.0
    seed: int = 1337

    def __call__(self, input_ids: np.ndarray) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(input_ids, dtype=np.int64)
        L = x.shape[0]
        crop = int(self.crop_size or L)
        crop = max(1, min(crop, L))

        def _view() -> np.ndarray:
            start = int(rng.integers(0, max(1, L - crop + 1)))
            out = x[start : start + crop].copy()
            if self.noise_std > 0:
                noise = rng.normal(0.0, self.noise_std, size=out.shape)
                out = np.clip(np.rint(out + noise), 0, None).astype(np.int64)
            return out

        return {"view_a": _view(), "view_b": _view()}
