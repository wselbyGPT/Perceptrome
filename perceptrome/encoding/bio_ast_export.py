from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional

from perceptrome.bio_ast import BioAST
from perceptrome.encoding.bio_ast_builder import BuiltBioAST
from perceptrome.encoding.bio_ast_viz import ast_to_graph_json, ast_to_tree_json
from perceptrome.encoding.storage_map import build_storage_map_payload

EXPORT_LAYER_VERSION = 1
CANONICAL_DOCUMENT_SCHEMA = "bio_ast_canonical_document_v1"
TREE_TENSORS_SCHEMA = "bio_ast_tree_tensors_v1"
GRAPH_EDGES_SCHEMA = "bio_ast_graph_edges_v1"
MOTIF_FEATURES_SCHEMA = "bio_ast_motif_features_v1"
SUMMARY_SCHEMA = "bio_ast_summary_v1"


_EXPORT_FILENAMES = {
    "canonical_ast": "canonical.document.json",
    "motif_features": "features.motif.json",
    "tree_tensors": "message-passing.tree-tensors.json",
    "graph_edges": "message-passing.graph-edges.json",
    "tree_json": "view.tree.json",
    "graph_json": "view.graph.json",
    "storage_map": "view.storage-map.json",
    "summary_json": "view.summary.json",
}


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def stable_json_sha256(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def export_filenames() -> Dict[str, str]:
    return dict(_EXPORT_FILENAMES)


def _canonical_ast_json(built: BuiltBioAST, *, accession: str, source: str) -> Dict[str, Any]:
    payload = built.ast.to_dict()
    payload.update(
        {
            "schema": CANONICAL_DOCUMENT_SCHEMA,
            "export_version": EXPORT_LAYER_VERSION,
            "accession": str(accession),
            "source": str(source),
        }
    )
    return payload


def _motif_level_features(ast: BioAST, *, accession: str) -> Dict[str, Any]:
    motif_node_types = {"region", "domain", "sme", "microfeature", "residue", "kmer"}
    rows: List[Dict[str, Any]] = []
    for node in ast.nodes:
        if node.node_type not in motif_node_types:
            continue
        rows.append(
            {
                "node_id": node.canonical_id,
                "node_type": node.node_type,
                "parent_id": node.parent_id,
                "start": node.start,
                "end": node.end,
                "length": (int(node.end) - int(node.start) + 1) if node.start is not None and node.end is not None else None,
                "metadata": dict(node.metadata),
            }
        )
    return {
        "schema": MOTIF_FEATURES_SCHEMA,
        "export_version": EXPORT_LAYER_VERSION,
        "accession": str(accession),
        "row_count": len(rows),
        "rows": rows,
    }


def _tree_tensors_with_ids(built: BuiltBioAST, *, accession: str) -> Dict[str, Any]:
    base = built.to_tree_message_passing_tensors()
    node_ids = [node.canonical_id for node in built.ast.nodes]
    edge_index = base["edge_index"].tolist()
    return {
        "schema": TREE_TENSORS_SCHEMA,
        "export_version": EXPORT_LAYER_VERSION,
        "accession": str(accession),
        "node_count": len(node_ids),
        "edge_count": len(edge_index[0]) if edge_index else 0,
        "node_ids": node_ids,
        **{key: value.tolist() for key, value in base.items()},
    }


def _graph_edge_list(ast: BioAST, *, accession: str) -> Dict[str, Any]:
    id_to_idx = {node.canonical_id: idx for idx, node in enumerate(ast.nodes)}
    edges: List[Dict[str, Any]] = []
    for node in ast.nodes:
        if not node.parent_id or node.parent_id not in id_to_idx:
            continue
        edges.append(
            {
                "parent_id": node.parent_id,
                "child_id": node.canonical_id,
                "parent_index": id_to_idx[node.parent_id],
                "child_index": id_to_idx[node.canonical_id],
            }
        )
    return {
        "schema": GRAPH_EDGES_SCHEMA,
        "export_version": EXPORT_LAYER_VERSION,
        "accession": str(accession),
        "edge_count": len(edges),
        "edges": edges,
    }


def _summary_payload(ast: BioAST, sequence_length: int, *, accession: str, canonical_sha256: str) -> Dict[str, Any]:
    node_type_counts: Dict[str, int] = {}
    relation_counts: Dict[str, int] = {}
    for node in ast.nodes:
        node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1
    for edge in ast.edges:
        relation_counts[edge.kind] = relation_counts.get(edge.kind, 0) + 1
    root_ids = [node.canonical_id for node in ast.nodes if not node.parent_id]
    return {
        "schema": SUMMARY_SCHEMA,
        "export_version": EXPORT_LAYER_VERSION,
        "accession": str(accession),
        "sequence_length": int(sequence_length),
        "canonical_sha256": canonical_sha256,
        "node_count": len(ast.nodes),
        "edge_count": len(ast.edges),
        "semantic_edge_count": len(ast.relationships),
        "root_ids": sorted(root_ids),
        "node_type_counts": node_type_counts,
        "relation_counts": relation_counts,
    }


def _annotate_derived_payload(payload: Dict[str, Any], *, accession: str, canonical_sha256: str) -> Dict[str, Any]:
    enriched = dict(payload)
    enriched["export_version"] = EXPORT_LAYER_VERSION
    enriched["accession"] = str(accession)
    enriched["derived_from"] = {
        "schema": CANONICAL_DOCUMENT_SCHEMA,
        "canonical_sha256": canonical_sha256,
    }
    return enriched


def build_bio_ast_export_artifacts(built: BuiltBioAST, *, accession: str, source: str) -> Dict[str, Any]:
    canonical = _canonical_ast_json(built, accession=accession, source=source)
    canonical_sha256 = stable_json_sha256(canonical)
    return {
        "canonical_ast": canonical,
        "motif_features": _annotate_derived_payload(_motif_level_features(built.ast, accession=accession), accession=accession, canonical_sha256=canonical_sha256),
        "tree_tensors": _annotate_derived_payload(_tree_tensors_with_ids(built, accession=accession), accession=accession, canonical_sha256=canonical_sha256),
        "graph_edges": _annotate_derived_payload(_graph_edge_list(built.ast, accession=accession), accession=accession, canonical_sha256=canonical_sha256),
        "tree_json": _annotate_derived_payload(ast_to_tree_json(built.ast, accession=str(accession)), accession=accession, canonical_sha256=canonical_sha256),
        "graph_json": _annotate_derived_payload(ast_to_graph_json(built.ast, accession=str(accession)), accession=accession, canonical_sha256=canonical_sha256),
        "storage_map": _annotate_derived_payload(build_storage_map_payload(built.ast, len(built.sequence), accession=str(accession)), accession=accession, canonical_sha256=canonical_sha256),
        "summary_json": _summary_payload(built.ast, len(built.sequence), accession=accession, canonical_sha256=canonical_sha256),
    }


def normalize_visualization_loader_payload(payload: Mapping[str, Any]) -> Any:
    if payload.get("schema") == GRAPH_EDGES_SCHEMA:
        return list(payload.get("edges", []))
    if payload.get("schema") == MOTIF_FEATURES_SCHEMA:
        return list(payload.get("rows", []))
    return payload
