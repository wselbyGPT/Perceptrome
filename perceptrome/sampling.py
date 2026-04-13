"""Advanced token-level sampling strategies for genome generation.

Each strategy takes raw logits (numpy) and returns a sampled token index.
All strategies can be composed with GC bias and AST vocabulary masks applied
upstream — they operate on the final log-probability vector.

Strategies
----------
- ``temperature`` : standard softmax(logits / T) sampling (baseline)
- ``top_k``       : zero out all but the *k* highest-scoring tokens, then sample
- ``top_p``       : nucleus sampling — keep the smallest set of tokens whose
                    cumulative probability exceeds *p*
- ``beam``        : beam search over a full window, returning the highest-scoring
                    complete sequence
- ``anneal``      : linearly anneal temperature from T_start → T_end across positions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingConfig:
    """Unified configuration for all sampling strategies."""

    strategy: str = "temperature"     # temperature | top_k | top_p | beam | anneal
    temperature: float = 1.0
    top_k: int = 0                    # 0 = disabled; >0 keeps only top-k tokens
    top_p: float = 1.0               # 1.0 = disabled; <1.0 enables nucleus sampling
    beam_width: int = 1              # 1 = greedy; >1 for beam search
    anneal_start: float = 1.2        # starting temperature for anneal strategy
    anneal_end: float = 0.5          # ending temperature for anneal strategy


def default_sampling_config() -> SamplingConfig:
    return SamplingConfig()


# ---------------------------------------------------------------------------
# Core sampling primitives
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Stable softmax with temperature scaling."""
    T = max(1e-6, float(temperature))
    x = logits.astype(np.float64) / T
    x = x - np.max(x)
    w = np.exp(x)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / w.shape[0]
    return w / s


def sample_temperature(logits: np.ndarray, temperature: float) -> int:
    """Standard temperature-scaled categorical sampling."""
    probs = _softmax(logits, temperature)
    return int(np.random.choice(probs.shape[0], p=probs))


def sample_top_k(logits: np.ndarray, k: int, temperature: float) -> int:
    """Top-k sampling: zero out all but the *k* highest logits, then sample.

    Prevents sampling from the long tail of low-probability tokens which
    can produce nonsense codons or invalid amino acids.
    """
    k = max(1, min(k, logits.shape[0]))
    indices = np.argpartition(logits, -k)[-k:]
    masked = np.full_like(logits, -1e9)
    masked[indices] = logits[indices]
    probs = _softmax(masked, temperature)
    return int(np.random.choice(probs.shape[0], p=probs))


def sample_top_p(logits: np.ndarray, p: float, temperature: float) -> int:
    """Nucleus (top-p) sampling: keep the smallest token set whose cumulative
    probability exceeds *p*, then re-normalize and sample.

    Dynamically adapts the effective vocabulary size per position — tight
    distributions use few tokens, ambiguous positions use many.
    """
    p = max(0.01, min(1.0, float(p)))
    probs = _softmax(logits, temperature)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    # Find cutoff: keep tokens up to where cumsum first exceeds p
    cutoff = int(np.searchsorted(cumsum, p)) + 1
    cutoff = min(cutoff, probs.shape[0])
    # Zero out tokens below the cutoff
    keep = set(sorted_idx[:cutoff].tolist())
    masked = np.where(np.isin(np.arange(probs.shape[0]), list(keep)), probs, 0.0)
    s = masked.sum()
    if s <= 0 or not np.isfinite(s):
        return int(sorted_idx[0])
    masked /= s
    return int(np.random.choice(masked.shape[0], p=masked))


def sample_anneal(logits: np.ndarray, position: int, total_positions: int,
                  t_start: float, t_end: float) -> int:
    """Temperature-annealed sampling: linearly interpolate temperature from
    t_start to t_end based on position in the sequence.

    Early positions use higher temperature for diversity; later positions
    use lower temperature for coherence — mimicking how biological sequences
    have more constrained 3' ends.
    """
    frac = float(position) / max(1, total_positions - 1) if total_positions > 1 else 0.0
    t = t_start + (t_end - t_start) * frac
    return sample_temperature(logits, t)


# ---------------------------------------------------------------------------
# Beam search (window-level)
# ---------------------------------------------------------------------------

@dataclass
class _Beam:
    """Single beam hypothesis during beam search."""
    tokens: List[int] = field(default_factory=list)
    log_prob: float = 0.0


def beam_search_window(
    logits_matrix: np.ndarray,
    beam_width: int,
    temperature: float = 1.0,
) -> List[int]:
    """Beam search over a full window of logits.

    Parameters
    ----------
    logits_matrix : (seq_len, vocab_size)
        Pre-computed logits for each position in the window.
    beam_width : int
        Number of hypotheses to maintain. 1 = greedy decoding.
    temperature : float
        Temperature applied before scoring.

    Returns
    -------
    List[int]
        Token indices for the highest-scoring complete sequence.
    """
    seq_len, vocab_size = logits_matrix.shape
    beam_width = max(1, int(beam_width))

    beams: List[_Beam] = [_Beam()]

    for pos in range(seq_len):
        probs = _softmax(logits_matrix[pos], temperature)
        log_probs = np.log(np.clip(probs, 1e-12, None))
        candidates: List[_Beam] = []
        for beam in beams:
            # Expand each beam with top-k tokens (k = 2 * beam_width for diversity)
            expand_k = min(vocab_size, 2 * beam_width)
            top_indices = np.argpartition(log_probs, -expand_k)[-expand_k:]
            for idx in top_indices:
                candidates.append(_Beam(
                    tokens=beam.tokens + [int(idx)],
                    log_prob=beam.log_prob + float(log_probs[idx]),
                ))
        # Keep top beam_width candidates
        candidates.sort(key=lambda b: b.log_prob, reverse=True)
        beams = candidates[:beam_width]

    return beams[0].tokens if beams else list(np.argmax(logits_matrix, axis=1))


# ---------------------------------------------------------------------------
# Dispatcher: route SamplingConfig to the right strategy
# ---------------------------------------------------------------------------

def sample_token(
    logits: np.ndarray,
    cfg: SamplingConfig,
    position: int = 0,
    total_positions: int = 1,
) -> int:
    """Sample a single token using the configured strategy.

    For beam search, use :func:`beam_search_window` instead (it operates
    on the full logits matrix).
    """
    strategy = cfg.strategy.lower()
    if strategy == "top_k" and cfg.top_k > 0:
        return sample_top_k(logits, cfg.top_k, cfg.temperature)
    elif strategy == "top_p" and cfg.top_p < 1.0:
        return sample_top_p(logits, cfg.top_p, cfg.temperature)
    elif strategy == "anneal":
        return sample_anneal(logits, position, total_positions, cfg.anneal_start, cfg.anneal_end)
    else:
        return sample_temperature(logits, cfg.temperature)


def sample_window(
    logits_matrix: np.ndarray,
    cfg: SamplingConfig,
    bias_weights: Optional[np.ndarray] = None,
) -> List[int]:
    """Sample a full window of tokens using the configured strategy.

    Parameters
    ----------
    logits_matrix : (seq_len, vocab_size)
    cfg : SamplingConfig
    bias_weights : optional (vocab_size,) multiplicative bias applied to
                   probabilities before sampling (e.g., GC bias, AST mask).

    Returns
    -------
    List of token indices, length = seq_len.
    """
    seq_len, vocab_size = logits_matrix.shape

    # Apply bias weights to logits (convert multiplicative to additive)
    if bias_weights is not None:
        bw = np.clip(bias_weights[:vocab_size], 1e-9, None)
        logits_matrix = logits_matrix + np.log(bw)

    if cfg.strategy.lower() == "beam" and cfg.beam_width > 1:
        return beam_search_window(logits_matrix, cfg.beam_width, cfg.temperature)

    tokens: List[int] = []
    for pos in range(seq_len):
        idx = sample_token(logits_matrix[pos], cfg, position=pos, total_positions=seq_len)
        tokens.append(idx)
    return tokens
