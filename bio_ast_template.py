from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from perceptrome.bio_ast import BioAST, RelationshipEdge


_TEMPLATE_SCHEMA_VERSION = 1


def _node_sort_key(node: Any) -> tuple:
    start = int(node.start) if getattr(node, "start", None) is not None else 10**12
    end = int(node.end) if getattr(node, "end", None) is not None else 10**12
    return (str(getattr(node, "node_type", "")), start, end, str(getattr(node, "canonical_id", "")))


@dataclass(frozen=True)
class BioASTTemplateNode:
    template_id: str
    node_type: str
    ordinal: int
    start: Optional[int] = None
    end: Optional[int] = None
    start_tolerance: int = 0
    end_tolerance: int = 0
    parent_type: Optional[str] = None
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "node_type": self.node_type,
            "ordinal": self.ordinal,
            "start": self.start,
            "end": self.end,
            "start_tolerance": int(self.start_tolerance),
            "end_tolerance": int(self.end_tolerance),
            "parent_type": self.parent_type,
            "required": bool(self.required),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BioASTTemplateNode":
        return cls(
            template_id=str(payload.get("template_id") or ""),
            node_type=str(payload.get("node_type") or "").lower(),
            ordinal=int(payload.get("ordinal") or 0),
            start=int(payload["start"]) if payload.get("start") is not None else None,
            end=int(payload["end"]) if payload.get("end") is not None else None,
            start_tolerance=max(0, int(payload.get("start_tolerance") or 0)),
            end_tolerance=max(0, int(payload.get("end_tolerance") or 0)),
            parent_type=str(payload["parent_type"]).lower() if payload.get("parent_type") is not None else None,
            required=bool(payload.get("required", True)),
        )


@dataclass(frozen=True)
class BioASTTemplateOrderingConstraint:
    before: str
    after: str

    def to_dict(self) -> Dict[str, str]:
        return {"before": self.before, "after": self.after}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BioASTTemplateOrderingConstraint":
        return cls(before=str(payload.get("before") or ""), after=str(payload.get("after") or ""))


@dataclass(frozen=True)
class BioASTTemplateSemanticEdge:
    source: str
    target: str
    kind: str
    required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target, "kind": self.kind, "required": bool(self.required)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BioASTTemplateSemanticEdge":
        return cls(
            source=str(payload.get("source") or ""),
            target=str(payload.get("target") or ""),
            kind=str(payload.get("kind") or "").lower(),
            required=bool(payload.get("required", False)),
        )


@dataclass(frozen=True)
class BioASTTemplate:
    schema_version: int = _TEMPLATE_SCHEMA_VERSION
    source_kind: str = "bio_ast_template"
    top_level_type: Optional[str] = None
    topology: Optional[str] = None
    sequence_length: Optional[int] = None
    nodes: Tuple[BioASTTemplateNode, ...] = ()
    ordering_constraints: Tuple[BioASTTemplateOrderingConstraint, ...] = ()
    semantic_edges: Tuple[BioASTTemplateSemanticEdge, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "source_kind": self.source_kind,
            "top_level_type": self.top_level_type,
            "topology": self.topology,
            "sequence_length": self.sequence_length,
            "nodes": [node.to_dict() for node in self.nodes],
            "ordering_constraints": [item.to_dict() for item in self.ordering_constraints],
            "semantic_edges": [item.to_dict() for item in self.semantic_edges],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BioASTTemplate":
        return cls(
            schema_version=int(payload.get("schema_version") or _TEMPLATE_SCHEMA_VERSION),
            source_kind=str(payload.get("source_kind") or "bio_ast_template"),
            top_level_type=str(payload["top_level_type"]).lower() if payload.get("top_level_type") is not None else None,
            topology=str(payload["topology"]).lower() if payload.get("topology") is not None else None,
            sequence_length=int(payload["sequence_length"]) if payload.get("sequence_length") is not None else None,
            nodes=tuple(BioASTTemplateNode.from_dict(item) for item in (payload.get("nodes") or []) if isinstance(item, Mapping)),
            ordering_constraints=tuple(BioASTTemplateOrderingConstraint.from_dict(item) for item in (payload.get("ordering_constraints") or []) if isinstance(item, Mapping)),
            semantic_edges=tuple(BioASTTemplateSemanticEdge.from_dict(item) for item in (payload.get("semantic_edges") or []) if isinstance(item, Mapping)),
            metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {},
        )


def derive_bio_ast_template(ast: BioAST, *, span_tolerance: int = 0, include_semantic_edges: bool = False) -> BioASTTemplate:
    ordered_nodes = sorted(ast.nodes, key=_node_sort_key)
    per_type: Dict[str, int] = {}
    node_templates: List[BioASTTemplateNode] = []
    canonical_to_template: Dict[str, str] = {}
    for node in ordered_nodes:
        node_type = str(node.node_type).lower()
        ordinal = int(per_type.get(node_type, 0))
        per_type[node_type] = ordinal + 1
        template_id = f"{node_type}#{ordinal}"
        canonical_to_template[str(node.canonical_id)] = template_id
        parent_type = None
        if getattr(node, "parent_id", None):
            for parent in ast.nodes:
                if parent.canonical_id == node.parent_id:
                    parent_type = str(parent.node_type).lower()
                    break
        node_templates.append(
            BioASTTemplateNode(
                template_id=template_id,
                node_type=node_type,
                ordinal=ordinal,
                start=node.start,
                end=node.end,
                start_tolerance=int(span_tolerance),
                end_tolerance=int(span_tolerance),
                parent_type=parent_type,
            )
        )

    ordering_constraints: List[BioASTTemplateOrderingConstraint] = []
    for left, right in zip(node_templates, node_templates[1:]):
        if left.node_type == right.node_type:
            ordering_constraints.append(BioASTTemplateOrderingConstraint(before=left.template_id, after=right.template_id))

    semantic_edges: List[BioASTTemplateSemanticEdge] = []
    if include_semantic_edges:
        for edge in ast.semantic_edges:
            src = canonical_to_template.get(edge.source_id)
            dst = canonical_to_template.get(edge.target_id)
            if src and dst:
                semantic_edges.append(BioASTTemplateSemanticEdge(source=src, target=dst, kind=str(edge.kind).lower(), required=True))

    meta = ast.sequence_metadata
    top_level_type = None
    if ordered_nodes:
        top_level_type = str(ordered_nodes[0].node_type).lower()
    return BioASTTemplate(
        top_level_type=top_level_type,
        topology=str(meta.topology).lower() if meta.topology else None,
        sequence_length=int(meta.length) if meta.length else None,
        nodes=tuple(node_templates),
        ordering_constraints=tuple(ordering_constraints),
        semantic_edges=tuple(semantic_edges),
        metadata={"derived_from_schema_version": int(ast.schema_version)},
    )


def parse_bio_ast_template(payload: Mapping[str, Any], *, span_tolerance: int = 0, include_semantic_edges: bool = False) -> BioASTTemplate:
    if payload.get("source_kind") == "bio_ast_template" or payload.get("nodes") and isinstance((payload.get("nodes") or [None])[0], Mapping) and "template_id" in (payload.get("nodes") or [{}])[0]:
        return BioASTTemplate.from_dict(payload)
    return derive_bio_ast_template(BioAST.from_dict(payload), span_tolerance=span_tolerance, include_semantic_edges=include_semantic_edges)


def load_bio_ast_template(path: str, *, span_tolerance: int = 0, include_semantic_edges: bool = False) -> BioASTTemplate:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Bio-AST template artifact must be a JSON object: {path}")
    return parse_bio_ast_template(payload, span_tolerance=span_tolerance, include_semantic_edges=include_semantic_edges)


def compare_bio_ast_to_template(ast: BioAST, template: BioASTTemplate) -> Dict[str, Any]:
    ordered_nodes = sorted(ast.nodes, key=_node_sort_key)
    per_type: Dict[str, int] = {}
    template_index = {node.template_id: node for node in template.nodes}
    candidate_index: Dict[str, Any] = {}
    template_positions: Dict[str, int] = {}
    matched_nodes: List[Dict[str, Any]] = []
    failed_nodes: List[Dict[str, Any]] = []
    extra_nodes: List[Dict[str, Any]] = []

    for node in ordered_nodes:
        node_type = str(node.node_type).lower()
        ordinal = int(per_type.get(node_type, 0))
        per_type[node_type] = ordinal + 1
        template_id = f"{node_type}#{ordinal}"
        candidate_index[template_id] = node
        template_positions[template_id] = len(template_positions)
        spec = template_index.get(template_id)
        if spec is None:
            extra_nodes.append({"template_id": template_id, "canonical_id": node.canonical_id, "node_type": node_type})
            continue
        start_ok = spec.start is None or node.start is None or abs(int(node.start) - int(spec.start)) <= int(spec.start_tolerance)
        end_ok = spec.end is None or node.end is None or abs(int(node.end) - int(spec.end)) <= int(spec.end_tolerance)
        parent_type = None
        if getattr(node, "parent_id", None):
            parent = next((item for item in ast.nodes if item.canonical_id == node.parent_id), None)
            parent_type = str(parent.node_type).lower() if parent is not None else None
        parent_ok = spec.parent_type is None or spec.parent_type == parent_type
        payload = {
            "template_id": template_id,
            "canonical_id": node.canonical_id,
            "node_type": node_type,
            "span_expected": [spec.start, spec.end],
            "span_actual": [node.start, node.end],
            "span_match": bool(start_ok and end_ok),
            "parent_type_expected": spec.parent_type,
            "parent_type_actual": parent_type,
            "parent_match": bool(parent_ok),
            "matched": bool(start_ok and end_ok and parent_ok),
        }
        if payload["matched"]:
            matched_nodes.append(payload)
        else:
            failed_nodes.append(payload)

    missing_nodes: List[Dict[str, Any]] = []
    for spec in template.nodes:
        if spec.template_id not in candidate_index:
            missing_nodes.append({"template_id": spec.template_id, "node_type": spec.node_type, "required": spec.required})

    ordering_matches: List[Dict[str, Any]] = []
    ordering_failures: List[Dict[str, Any]] = []
    for constraint in template.ordering_constraints:
        before_node = candidate_index.get(constraint.before)
        after_node = candidate_index.get(constraint.after)
        ok = before_node is not None and after_node is not None and _node_sort_key(before_node) <= _node_sort_key(after_node)
        payload = {"before": constraint.before, "after": constraint.after, "matched": bool(ok)}
        (ordering_matches if ok else ordering_failures).append(payload)

    semantic_edge_keys = {(edge.source_id, edge.target_id, edge.kind) for edge in ast.semantic_edges}
    edge_matches: List[Dict[str, Any]] = []
    edge_failures: List[Dict[str, Any]] = []
    candidate_semantic_by_template: set[tuple[str, str, str]] = set()
    canonical_to_template = {getattr(node, "canonical_id", ""): template_id for template_id, node in candidate_index.items()}
    for edge in ast.semantic_edges:
        src = canonical_to_template.get(edge.source_id)
        dst = canonical_to_template.get(edge.target_id)
        if src and dst:
            candidate_semantic_by_template.add((src, dst, edge.kind))
    for requirement in template.semantic_edges:
        ok = (requirement.source, requirement.target, requirement.kind) in candidate_semantic_by_template
        payload = {"source": requirement.source, "target": requirement.target, "kind": requirement.kind, "required": requirement.required, "matched": bool(ok)}
        if ok or not requirement.required:
            edge_matches.append(payload)
        else:
            edge_failures.append(payload)

    total_checks = len(template.nodes) + len(template.ordering_constraints) + len([edge for edge in template.semantic_edges if edge.required])
    passed_checks = len(matched_nodes) + len(ordering_matches) + len([edge for edge in edge_matches if edge.get("required")])
    score = 1.0 if total_checks == 0 else max(0.0, min(1.0, float(passed_checks) / float(total_checks)))
    required_missing = len([item for item in missing_nodes if item.get("required")])
    mismatch_count = int(required_missing + len(failed_nodes) + len(ordering_failures) + len(edge_failures))
    return {
        "template_topology": template.topology,
        "candidate_topology": ast.sequence_metadata.topology,
        "topology_match": template.topology is None or str(ast.sequence_metadata.topology).lower() == str(template.topology).lower(),
        "template_top_level_type": template.top_level_type,
        "candidate_top_level_type": str(ordered_nodes[0].node_type).lower() if ordered_nodes else None,
        "node_matches": matched_nodes,
        "node_failures": failed_nodes,
        "missing_nodes": missing_nodes,
        "extra_nodes": extra_nodes,
        "ordering_matches": ordering_matches,
        "ordering_failures": ordering_failures,
        "semantic_edge_matches": edge_matches,
        "semantic_edge_failures": edge_failures,
        "summary": {
            "matched_node_count": len(matched_nodes),
            "failed_node_count": len(failed_nodes),
            "missing_node_count": len(missing_nodes),
            "extra_node_count": len(extra_nodes),
            "ordering_failure_count": len(ordering_failures),
            "semantic_edge_failure_count": len(edge_failures),
            "mismatch_count": mismatch_count,
            "score": float(score),
        },
    }
