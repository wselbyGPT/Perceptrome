from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from perceptrome.encoding_main import IDX_TO_AA, IDX_TO_CODON


_IDX_TO_BASE = ["A", "C", "G", "T"]


def tokens_to_sequence(tokens: List[int], tokenizer: str) -> str:
    tok = tokenizer.lower()
    if tok == "aa":
        return "".join(IDX_TO_AA[t] if t < len(IDX_TO_AA) else "X" for t in tokens)
    if tok == "codon":
        return "".join(IDX_TO_CODON[t] if t < len(IDX_TO_CODON) else "NNN" for t in tokens)
    return "".join(_IDX_TO_BASE[t % 4] for t in tokens)


@dataclass(frozen=True)
class InterpStep:
    step: int
    t: float
    sequence: str


@dataclass(frozen=True)
class InterpSummary:
    run_id: str
    acc_a: str
    acc_b: str
    steps: int
    latent_dim: int
    tokenizer: str
    n_windows: int
    window_size: int
    output_length: int
    output_fasta: str
    started_at: str
    completed_at: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _fasta_lines(seq: str, width: int = 60) -> List[str]:
    return [seq[i : i + width] for i in range(0, len(seq), width)]


def write_interp_fasta(
    path: str,
    acc_a: str,
    acc_b: str,
    steps: List[InterpStep],
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(steps)
    with open(path, "w", encoding="utf-8") as f:
        for s in steps:
            header = (
                f">interp_step{s.step}_of{n}"
                f"|t={s.t:.4f}|a={acc_a}|b={acc_b}"
            )
            f.write(header + "\n")
            for line in _fasta_lines(s.sequence):
                f.write(line + "\n")
    return path


def write_interp_summary_json(
    path: str,
    summary: InterpSummary,
    steps: List[InterpStep],
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = summary.to_dict()
    payload["steps"] = [asdict(s) for s in steps]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
