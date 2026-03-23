from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from perceptrome.bio_ast import BioAST, CONTAINMENT_EDGE_KIND


SortKey = Tuple[int, int, int, int, str]

NODE_TYPE_ORDER = {
    "genome": 0,
    "plasmid": 0,
    "virus": 0,
    "gene": 1,
    "orf": 2,
    "cds": 3,
    "region": 4,
    "domain": 5,
    "sme": 6,
    "microfeature": 7,
    "residue": 8,
    "kmer": 9,
}


def _node_sort_key(node: Any) -> SortKey:
    start = int(node.start) if node.start is not None else 10**12
    end = int(node.end) if node.end is not None else 10**12
    type_rank = int(NODE_TYPE_ORDER.get(str(node.node_type), 100))
    return (start, end, type_rank, len(str(node.canonical_id)), str(node.canonical_id))


def _label_for_node(node: Any) -> str:
    if getattr(node, "node_type", "") == "gene" and getattr(node, "gene_id", None):
        return f"gene:{node.gene_id}"
    return f"{node.node_type}:{node.canonical_id}"


def _node_payload(node: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": node.canonical_id,
        "label": _label_for_node(node),
        "node_type": node.node_type,
        "parent_id": node.parent_id,
        "span": {
            "start": node.start,
            "end": node.end,
            "strand": node.strand,
            "frame": node.frame,
        },
        "metadata": dict(getattr(node, "metadata", {}) or {}),
    }
    if node.start is not None and node.end is not None:
        payload["coordinates"] = {"x": int(node.start), "y": int(node.end)}
    return payload


def ast_to_tree_json(ast: BioAST, *, accession: Optional[str] = None) -> Dict[str, Any]:
    node_by_id = {node.canonical_id: node for node in ast.nodes}
    children: Dict[str, List[str]] = {node_id: [] for node_id in node_by_id}
    roots: List[str] = []

    for node in ast.nodes:
        if node.parent_id and node.parent_id in children:
            children[node.parent_id].append(node.canonical_id)
        else:
            roots.append(node.canonical_id)

    def _visit(node_id: str) -> Dict[str, Any]:
        node = node_by_id[node_id]
        sorted_children = sorted(children.get(node_id, []), key=lambda child_id: _node_sort_key(node_by_id[child_id]))
        return {
            **_node_payload(node),
            "children": [_visit(child_id) for child_id in sorted_children],
        }

    ordered_roots = sorted(set(roots), key=lambda node_id: _node_sort_key(node_by_id[node_id]))
    hierarchy = [_visit(root_id) for root_id in ordered_roots]

    return {
        "accession": accession,
        "schema": "bio_ast_tree_v1",
        "node_count": len(ast.nodes),
        "sequence_metadata": ast.sequence_metadata.to_dict(),
        "roots": ordered_roots,
        "hierarchy": hierarchy,
    }


def ast_to_graph_json(ast: BioAST, *, accession: Optional[str] = None) -> Dict[str, Any]:
    ordered_nodes = sorted(ast.nodes, key=_node_sort_key)
    node_index = {node.canonical_id: idx for idx, node in enumerate(ordered_nodes)}

    nodes_payload = [
        {
            **_node_payload(node),
            "index": idx,
        }
        for idx, node in enumerate(ordered_nodes)
    ]

    edges: List[Dict[str, Any]] = []
    hierarchy_edges: List[Dict[str, Any]] = []
    semantic_edges: List[Dict[str, Any]] = []
    for edge in ast.edges:
        if edge.source_id not in node_index or edge.target_id not in node_index:
            continue
        relation_type = "hierarchy" if edge.kind == CONTAINMENT_EDGE_KIND else "semantic"
        edge_payload = {
            "id": f"{relation_type}:{edge.kind}:{edge.source_id}->{edge.target_id}",
            "source": edge.source_id,
            "target": edge.target_id,
            "source_index": node_index[edge.source_id],
            "target_index": node_index[edge.target_id],
            "relation": edge.kind,
            "edge_kind": edge.kind,
            "relation_type": relation_type,
            "evidence": dict(edge.metadata).get("evidence", "curated" if relation_type == "hierarchy" else "unspecified"),
            "metadata": dict(edge.metadata),
        }
        edges.append(edge_payload)
        if relation_type == "hierarchy":
            hierarchy_edges.append(edge_payload)
        else:
            semantic_edges.append(edge_payload)

    edges.sort(
        key=lambda item: (
            str(item["relation_type"]),
            str(item["relation"]),
            int(item["source_index"]),
            int(item["target_index"]),
        )
    )

    return {
        "accession": accession,
        "schema": "bio_ast_graph_v1",
        "node_count": len(nodes_payload),
        "edge_count": len(edges),
        "hierarchy_edge_count": len(hierarchy_edges),
        "semantic_edge_count": len(semantic_edges),
        "sequence_metadata": ast.sequence_metadata.to_dict(),
        "nodes": nodes_payload,
        "edges": edges,
        "hierarchy_edges": sorted(hierarchy_edges, key=lambda item: str(item["id"])),
        "semantic_edges": sorted(semantic_edges, key=lambda item: str(item["id"])),
    }
