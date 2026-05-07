from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ELBORecord:
    accession: str
    status: str           # "ok" | "error"
    elbo: float           # -(recon_loss + kl), higher = better fit
    recon_loss: float     # mean CE (or MSE) per position, lower = better
    kl: float             # mean KL per latent unit, lower = closer to prior
    n_windows: int
    sequence_length: int  # total tokens covered (windows * seq_len)
    error: Optional[str] = None


@dataclass(frozen=True)
class ELBOSummary:
    run_id: str
    catalog_path: str
    n_scored: int
    n_failed: int
    tokenizer: str
    model_type: str
    window_size: int
    loss_type: str
    started_at: str
    completed_at: str
    mean_elbo: float
    mean_recon_loss: float
    mean_kl: float
    metadata: Dict[str, Any]


def score_windows(
    model: Any,
    encoded_np: "np.ndarray",
    seq_len: int,
    vocab_size: int,
    loss_type: str,
    device: Any,
) -> Tuple[float, float, float]:
    """Forward-pass all windows of one accession and return (elbo, recon_loss, kl).

    Uses z = mu (deterministic posterior mean) so scores are reproducible.
    All values are means over windows and over positions/latent units so they
    are comparable across sequences of different lengths.

    elbo = -(recon_loss + kl)   higher is better
    """
    import numpy as np

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise RuntimeError("PyTorch is required for score_windows.")

    wt = torch.from_numpy(encoded_np).to(device)
    N = int(wt.size(0))
    x_flat = wt.view(N, -1)

    with torch.no_grad():
        mu, logvar = model.encode(x_flat)
        recon_logits = model.decode(mu)

        lt = str(loss_type).lower()
        if lt == "ce":
            logits = recon_logits.view(N, int(seq_len), int(vocab_size))
            targets = x_flat.view(N, int(seq_len), int(vocab_size)).argmax(dim=2)
            ce_per_sample = (
                F.cross_entropy(
                    logits.view(-1, int(vocab_size)),
                    targets.view(-1),
                    reduction="none",
                )
                .view(N, int(seq_len))
                .mean(dim=1)
            )
            recon_per_sample = ce_per_sample.cpu().numpy()
        else:
            recon_t = torch.sigmoid(recon_logits)
            recon_per_sample = ((recon_t - x_flat) ** 2).mean(dim=1).cpu().numpy()

        kl_per_sample = (
            (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()))
            .mean(dim=1)
            .cpu()
            .numpy()
        )

    recon_mean = float(np.mean(recon_per_sample))
    kl_mean = float(np.mean(kl_per_sample))
    elbo = -(recon_mean + kl_mean)
    return elbo, recon_mean, kl_mean


def write_elbo_json(
    path: str,
    records: List[ELBORecord],
    summary: Optional[ELBOSummary] = None,
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {"records": [asdict(r) for r in records]}
    if summary is not None:
        payload["summary"] = asdict(summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_elbo_tsv(path: str, records: List[ELBORecord]) -> str:
    cols = [
        "accession", "status", "elbo", "recon_loss", "kl",
        "n_windows", "sequence_length", "error",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in records:
            d = asdict(r)
            f.write("\t".join(str(d.get(c, "") or "") for c in cols) + "\n")
    return path
