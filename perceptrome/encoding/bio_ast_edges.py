from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from perceptrome.bio_ast import BioAST, RelationshipEdge, SMENode


REGULATORY_FEATURE_TYPES = frozenset(
    {
        "promoter",
        "operator",
        "enhancer",
        "silencer",
        "terminator",
        "riboswitch",
        "regulatory_region",
        "tf_binding_site",
    }
)


def derive_semantic_edges(
    ast: BioAST,
    *,
    feature_annotations: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[RelationshipEdge, ...]:
    derived: List[RelationshipEdge] = []
    seen: Set[Tuple[str, str, str]] = {(edge.source_id, edge.target_id, edge.kind) for edge in ast.semantic_edges}

    for edge in _derive_coordinate_edges(ast):
        key = (edge.source_id, edge.target_id, edge.kind)
        if key not in seen:
            seen.add(key)
            derived.append(edge)

    for edge in _derive_regulatory_edges(ast, feature_annotations=feature_annotations):
        key = (edge.source_id, edge.target_id, edge.kind)
        if key not in seen:
            seen.add(key)
            derived.append(edge)

    return tuple(derived)


def _derive_coordinate_edges(ast: BioAST) -> Iterable[RelationshipEdge]:
    nodes = [node for node in ast.nodes if node.start is not None and node.end is not None]
    node_by_id = ast.node_by_id
    ancestor_cache: Dict[str, Set[str]] = {}

    def ancestors(node_id: str) -> Set[str]:
        cached = ancestor_cache.get(node_id)
        if cached is not None:
            return cached
        lineage: Set[str] = set()
        cursor = node_by_id.get(node_id)
        while cursor is not None and cursor.parent_id:
            lineage.add(cursor.parent_id)
            cursor = node_by_id.get(cursor.parent_id)
        ancestor_cache[node_id] = lineage
        return lineage

    ordered = sorted(nodes, key=lambda node: (int(node.start), int(node.end), str(node.node_type), str(node.canonical_id)))
    for idx, left in enumerate(ordered):
        left_ancestors = ancestors(left.canonical_id)
        for right in ordered[idx + 1 :]:
            if left.canonical_id in ancestors(right.canonical_id) or right.canonical_id in left_ancestors:
                continue
            left_start = int(left.start)
            left_end = int(left.end)
            right_start = int(right.start)
            right_end = int(right.end)
            if left_end >= right_start and right_end >= left_start:
                yield RelationshipEdge(
                    source_id=left.canonical_id,
                    target_id=right.canonical_id,
                    kind="overlaps",
                    metadata=_coordinate_provenance(
                        evidence="inferred",
                        method="interval_overlap",
                        coordinates={
                            "source": {"start": left_start, "end": left_end},
                            "target": {"start": right_start, "end": right_end},
                        },
                    ),
                )
            elif left_end + 1 == right_start:
                yield RelationshipEdge(
                    source_id=left.canonical_id,
                    target_id=right.canonical_id,
                    kind="adjacent_to",
                    metadata=_coordinate_provenance(
                        evidence="inferred",
                        method="interval_adjacency",
                        coordinates={
                            "source": {"start": left_start, "end": left_end},
                            "target": {"start": right_start, "end": right_end},
                        },
                    ),
                )


def _derive_regulatory_edges(
    ast: BioAST,
    *,
    feature_annotations: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Iterable[RelationshipEdge]:
    annotations = feature_annotations or {}
    node_by_id = ast.node_by_id
    for node in ast.nodes:
        annotation = _annotation_for_node(node, annotations)
        targets = _normalize_targets(annotation.get("regulates"))
        if not targets:
            continue
        if _annotation_supports_regulatory_edge(node, annotation):
            evidence = "curated" if annotation.get("regulates") else "heuristic"
            feature_type = _annotation_feature_type(annotation)
            for target_id in targets:
                if target_id not in node_by_id or target_id == node.canonical_id:
                    continue
                yield RelationshipEdge(
                    source_id=node.canonical_id,
                    target_id=target_id,
                    kind="regulates",
                    metadata={
                        "evidence": evidence,
                        "inferred": evidence != "curated",
                        "provenance": {
                            "source": "annotation" if evidence == "curated" else "feature_type_heuristic",
                            "feature_type": feature_type,
                        },
                    },
                )


def _annotation_for_node(node: Any, annotations: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    keys = [str(getattr(node, "canonical_id", ""))]
    gene_id = getattr(node, "gene_id", None)
    if gene_id:
        keys.append(str(gene_id))
    for key in keys:
        annotation = annotations.get(key)
        if isinstance(annotation, Mapping):
            return annotation
    return {}


def _normalize_targets(raw: Any) -> Tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if isinstance(raw, Sequence):
        return tuple(str(item) for item in raw if item)
    return ()


def _annotation_feature_type(annotation: Mapping[str, Any]) -> Optional[str]:
    raw = annotation.get("feature_type", annotation.get("type"))
    token = str(raw).strip().lower() if raw is not None else ""
    return token or None


def _annotation_supports_regulatory_edge(node: Any, annotation: Mapping[str, Any]) -> bool:
    if annotation.get("regulates"):
        return True
    feature_type = _annotation_feature_type(annotation)
    if feature_type in REGULATORY_FEATURE_TYPES:
        return True
    if isinstance(node, SMENode) and node.motif_family == "REGULATORY":
        return True
    node_feature_type = str(getattr(node, "metadata", {}).get("feature_type", "")).strip().lower()
    return node_feature_type in REGULATORY_FEATURE_TYPES


def _coordinate_provenance(*, evidence: str, method: str, coordinates: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "evidence": evidence,
        "inferred": True,
        "provenance": {
            "source": "coordinates",
            "method": method,
            "coordinates": dict(coordinates),
        },
    }
