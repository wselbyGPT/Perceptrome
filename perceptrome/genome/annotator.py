from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional

from perceptrome.bio_ast import BioAST
from perceptrome.bio_reg_graph import infer_regulatory_features
from perceptrome.encoding.bio_ast_builder import BioASTBuilder
from perceptrome.genome.parsers import GenomeInputContents, load_genome_input


@dataclass(frozen=True)
class GenomeAnnotationCounts:
    node_counts: Dict[str, int]
    edge_counts: Dict[str, int]
    cds_count: int
    gene_count: int
    promoter_count: int
    operator_count: int
    rbs_count: int
    terminator_count: int
    total_nodes: int
    total_edges: int


@dataclass(frozen=True)
class GenomeAnnotationResult:
    accession: str
    sequence_length: int
    source_format: str
    cds_source: str
    counts: GenomeAnnotationCounts


def _node_kind(node: Any) -> str:
    name = type(node).__name__
    if name.endswith("Node"):
        name = name[:-4]
    return name.lower() or "node"


def _edge_kind(edge: Any) -> str:
    return str(getattr(edge, "kind", None) or "unknown")


def compute_annotation_counts(ast: BioAST) -> GenomeAnnotationCounts:
    node_counter: Counter = Counter()
    for node in ast.nodes:
        node_counter[_node_kind(node)] += 1
    edge_counter: Counter = Counter()
    for edge in ast.semantic_edges:
        edge_counter[_edge_kind(edge)] += 1
    return GenomeAnnotationCounts(
        node_counts=dict(node_counter),
        edge_counts=dict(edge_counter),
        cds_count=int(node_counter.get("cds", 0)),
        gene_count=int(node_counter.get("gene", 0)),
        promoter_count=int(node_counter.get("promoter", 0)),
        operator_count=int(node_counter.get("operator", 0)),
        rbs_count=int(node_counter.get("rbs", 0)),
        terminator_count=int(node_counter.get("terminator", 0)),
        total_nodes=sum(node_counter.values()),
        total_edges=sum(edge_counter.values()),
    )


def annotate_genome_input(
    *,
    input_path: str,
    accession: Optional[str] = None,
) -> GenomeAnnotationResult:
    contents: GenomeInputContents = load_genome_input(input_path)
    accession_value = accession or os.path.splitext(os.path.basename(input_path))[0]

    builder = BioASTBuilder()
    built = builder.build(
        sequence=contents.sequence,
        cds_features=contents.cds_features,
        accession=accession_value,
        source_format=contents.source_format,
    )
    annotated_ast = infer_regulatory_features(built.ast, sequence=built.sequence)
    counts = compute_annotation_counts(annotated_ast)

    if contents.cds_features is None:
        cds_source = "orf_fallback"
    elif contents.cds_features:
        cds_source = "genbank_features"
    else:
        cds_source = "no_features"

    return GenomeAnnotationResult(
        accession=accession_value,
        sequence_length=len(contents.sequence),
        source_format=contents.source_format,
        cds_source=cds_source,
        counts=counts,
    )


def write_annotation_json(path: str, result: GenomeAnnotationResult) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "accession": result.accession,
        "sequence_length": result.sequence_length,
        "source_format": result.source_format,
        "cds_source": result.cds_source,
        "counts": {
            "node_counts": result.counts.node_counts,
            "edge_counts": result.counts.edge_counts,
            "cds_count": result.counts.cds_count,
            "gene_count": result.counts.gene_count,
            "promoter_count": result.counts.promoter_count,
            "operator_count": result.counts.operator_count,
            "rbs_count": result.counts.rbs_count,
            "terminator_count": result.counts.terminator_count,
            "total_nodes": result.counts.total_nodes,
            "total_edges": result.counts.total_edges,
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def read_annotation_json(path: str) -> GenomeAnnotationResult:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    counts_payload = payload.get("counts") or {}
    counts = GenomeAnnotationCounts(
        node_counts=dict(counts_payload.get("node_counts") or {}),
        edge_counts=dict(counts_payload.get("edge_counts") or {}),
        cds_count=int(counts_payload.get("cds_count", 0)),
        gene_count=int(counts_payload.get("gene_count", 0)),
        promoter_count=int(counts_payload.get("promoter_count", 0)),
        operator_count=int(counts_payload.get("operator_count", 0)),
        rbs_count=int(counts_payload.get("rbs_count", 0)),
        terminator_count=int(counts_payload.get("terminator_count", 0)),
        total_nodes=int(counts_payload.get("total_nodes", 0)),
        total_edges=int(counts_payload.get("total_edges", 0)),
    )
    return GenomeAnnotationResult(
        accession=str(payload.get("accession", "")),
        sequence_length=int(payload.get("sequence_length", 0)),
        source_format=str(payload.get("source_format", "")),
        cds_source=str(payload.get("cds_source", "")),
        counts=counts,
    )
