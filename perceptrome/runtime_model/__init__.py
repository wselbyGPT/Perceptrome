"""Hybrid Bio-AST runtime model primitives for Perceptrome.

This package is intentionally lightweight and dependency-free.  It gives
Perceptrome a first executable bridge from static Bio-AST structure to dynamic
cell-state simulation without pretending to be a full molecular simulator.
"""

from perceptrome.runtime_model.demos import build_p53_p21_demo
from perceptrome.runtime_model.engine import RuntimeEngine, run_simulation
from perceptrome.runtime_model.schema import (
    BioASTDocument,
    BioAstEdge,
    BioAstNode,
    EvidenceRecord,
    PerceptromeRuntimeBundle,
    PerturbationEvent,
    RuntimeFrame,
    RuntimeModel,
    RuntimeProfile,
    RuntimeRule,
    RuntimeVariable,
    SimulationRun,
)

__all__ = [
    "BioASTDocument",
    "BioAstEdge",
    "BioAstNode",
    "EvidenceRecord",
    "PerceptromeRuntimeBundle",
    "PerturbationEvent",
    "RuntimeEngine",
    "RuntimeFrame",
    "RuntimeModel",
    "RuntimeProfile",
    "RuntimeRule",
    "RuntimeVariable",
    "SimulationRun",
    "build_p53_p21_demo",
    "run_simulation",
]
