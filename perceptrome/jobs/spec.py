"""Lightweight job spec types — no ML dependencies.

This module exists so the TUI and CLI helpers can import :class:`JobSpec`
without pulling in the full torch/ML stack that lives in ``engine.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

JobKind = Literal[
    "train_one",
    "stream",
    "generate_plasmid",
    "generate_protein",
    "validate_plasmid",
    "pretrain",
    "design_loop",
]


@dataclass(slots=True)
class JobSpec:
    kind: "JobKind"
    config_path: str = "config/stream_config.yaml"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobEvent:
    stage: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobResult:
    ok: bool
    exit_code: int
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    events: list[JobEvent] = field(default_factory=list)


class JobCancelledError(Exception):
    pass
