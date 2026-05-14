from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .bio_ast import (
    BioAST,
    CargoModuleNode,
    ExpressionCassetteNode,
    GeneNode,
    OperonNode,
    OperatorNode,
    PlasmidNode,
    PromoterNode,
    ProteinProductNode,
    RBSNode,
    RelationshipEdge,
    ReplicationModuleNode,
    SelectionModuleNode,
    TerminatorNode,
    TranscriptNode,
    TranscriptUnitNode,
    MobilityModuleNode,
)

SHINE_DALGARNO = ("AGGAGG", "GGAG", "AGGA")


def infer_regulatory_features(ast: BioAST, *, sequence: str = "", annotations: Optional[Mapping[str, Mapping[str, Any]]] = None) -> BioAST:
    annotations = annotations or {}
    node_by_id = ast.node_by_id
    nodes: List[Any] = list(ast.nodes)
    edges: List[RelationshipEdge] = list(ast.semantic_edges)
    used_ids = {node.canonical_id for node in ast.nodes}

    for gene in [node for node in ast.nodes if isinstance(node, GeneNode)]:
        if gene.start is None or gene.end is None:
            continue
        strand = gene.strand or "+"
        promoter_start = max(0, int(gene.start) - 45)
        promoter_end = max(promoter_start, int(gene.start) - 15)
        promoter_id = f"promoter:{gene.canonical_id}"
        if promoter_id not in used_ids:
            nodes.append(
                PromoterNode(
                    canonical_id=promoter_id,
                    parent_id=gene.parent_id,
                    start=promoter_start,
                    end=promoter_end,
                    strand=strand,
                    metadata={"inference_method": "rule", "rule_name": "upstream_promoter_window", "confidence": 0.7},
                )
            )
            used_ids.add(promoter_id)
        edges.append(
            RelationshipEdge(
                source_id=promoter_id,
                target_id=gene.canonical_id,
                kind="promoter_of",
                metadata={"inference_method": "rule", "confidence": 0.7, "distance_bp": int(gene.start) - promoter_end, "rule_name": "promoter_to_first_orf"},
            )
        )

        operator_id = f"operator:{gene.canonical_id}"
        if operator_id not in used_ids:
            nodes.append(
                OperatorNode(
                    canonical_id=operator_id,
                    parent_id=gene.parent_id,
                    start=max(0, promoter_end - 12),
                    end=max(0, promoter_end - 2),
                    strand=strand,
                    metadata={"inference_method": "rule", "rule_name": "proximal_operator", "confidence": 0.45},
                )
            )
            used_ids.add(operator_id)
        edges.append(
            RelationshipEdge(
                source_id=operator_id,
                target_id=gene.canonical_id,
                kind="operator_of",
                metadata={"inference_method": "rule", "confidence": 0.45, "rule_name": "operator_near_promoter"},
            )
        )

        rbs_start = max(0, int(gene.start) - 12)
        rbs_end = max(rbs_start, int(gene.start) - 4)
        motif_hit = _scan_for_motif(sequence, rbs_start, rbs_end)
        rbs_conf = 0.85 if motif_hit else 0.35
        rbs_id = f"rbs:{gene.canonical_id}"
        if rbs_id not in used_ids:
            nodes.append(
                RBSNode(
                    canonical_id=rbs_id,
                    parent_id=gene.parent_id,
                    start=rbs_start,
                    end=rbs_end,
                    strand=strand,
                    metadata={"inference_method": "rule", "confidence": rbs_conf, "motif_hit": motif_hit},
                )
            )
            used_ids.add(rbs_id)
        edges.append(
            RelationshipEdge(
                source_id=rbs_id,
                target_id=gene.canonical_id,
                kind="rbs_for",
                metadata={"inference_method": "rule", "confidence": rbs_conf, "motif_hit": motif_hit, "rule_name": "shine_dalgarno_upstream"},
            )
        )

        term_id = f"terminator:{gene.canonical_id}"
        if term_id not in used_ids:
            nodes.append(
                TerminatorNode(
                    canonical_id=term_id,
                    parent_id=gene.parent_id,
                    start=int(gene.end) + 4,
                    end=int(gene.end) + 24,
                    strand=strand,
                    metadata={"inference_method": "rule", "rule_name": "downstream_terminator_window", "confidence": 0.5},
                )
            )
            used_ids.add(term_id)
        edges.append(
            RelationshipEdge(
                source_id=term_id,
                target_id=gene.canonical_id,
                kind="terminates",
                metadata={"inference_method": "rule", "confidence": 0.5, "distance_bp": 4, "rule_name": "post_cds_terminator"},
            )
        )

    return BioAST(nodes=tuple(nodes), sequence_metadata=ast.sequence_metadata, relationships=_dedupe(edges))


def infer_transcript_units(ast: BioAST) -> BioAST:
    genes = sorted([n for n in ast.nodes if isinstance(n, GeneNode) and n.start is not None], key=lambda n: (int(n.start), int(n.end or n.start)))
    nodes: List[Any] = list(ast.nodes)
    edges: List[RelationshipEdge] = list(ast.semantic_edges)
    seen = {node.canonical_id for node in ast.nodes}

    for idx, gene in enumerate(genes, start=1):
        tu_id = f"transcript_unit:{idx}:{gene.canonical_id}"
        if tu_id in seen:
            continue
        members = [gene]
        for nxt in genes:
            if nxt.canonical_id == gene.canonical_id or nxt.start is None or nxt.strand != gene.strand:
                continue
            gap = int(nxt.start) - int(members[-1].end or nxt.start)
            if 0 <= gap <= 80:
                members.append(nxt)
            elif gap > 80:
                break
        tu_start = min(int(m.start or 0) for m in members)
        tu_end = max(int(m.end or m.start or 0) for m in members)
        nodes.append(
            TranscriptUnitNode(
                canonical_id=tu_id,
                parent_id=gene.parent_id,
                start=tu_start,
                end=tu_end,
                strand=gene.strand,
                metadata={"inference_method": "rule", "confidence": 0.6, "rule_name": "same_strand_local_cluster"},
            )
        )
        seen.add(tu_id)

        transcript_id = f"transcript:{idx}:{gene.canonical_id}"
        nodes.append(
            TranscriptNode(
                canonical_id=transcript_id,
                parent_id=gene.parent_id,
                start=tu_start,
                end=tu_end,
                strand=gene.strand,
                metadata={"inference_method": "rule", "confidence": 0.6, "rule_name": "tu_materialization"},
            )
        )
        seen.add(transcript_id)
        edges.append(RelationshipEdge(source_id=tu_id, target_id=transcript_id, kind="produces_transcript", metadata={"inference_method": "rule", "confidence": 0.6}))

        for member in members:
            edges.append(RelationshipEdge(source_id=member.canonical_id, target_id=tu_id, kind="part_of_transcript_unit", metadata={"inference_method": "rule", "confidence": 0.6}))
            edges.append(RelationshipEdge(source_id=gene.canonical_id, target_id=member.canonical_id, kind="same_strand_as", metadata={"inference_method": "rule", "confidence": 1.0}))
            if member.canonical_id != gene.canonical_id and member.start is not None:
                edges.append(
                    RelationshipEdge(
                        source_id=gene.canonical_id,
                        target_id=member.canonical_id,
                        kind="downstream_of" if int(member.start) > int(gene.start or 0) else "upstream_of",
                        metadata={"inference_method": "rule", "distance_bp": abs(int(member.start) - int(gene.start or 0)), "confidence": 0.8},
                    )
                )

    return BioAST(nodes=tuple(nodes), sequence_metadata=ast.sequence_metadata, relationships=_dedupe(edges))


def infer_operons(ast: BioAST) -> BioAST:
    nodes: List[Any] = list(ast.nodes)
    edges: List[RelationshipEdge] = list(ast.semantic_edges)
    tu_nodes = [n for n in ast.nodes if isinstance(n, TranscriptUnitNode)]
    used = {n.canonical_id for n in ast.nodes}

    for idx, tu in enumerate(tu_nodes, start=1):
        members = [e.source_id for e in edges if e.kind == "part_of_transcript_unit" and e.target_id == tu.canonical_id]
        if len(members) < 2:
            continue
        op_id = f"operon:{idx}:{tu.canonical_id}"
        if op_id in used:
            continue
        nodes.append(
            OperonNode(
                canonical_id=op_id,
                parent_id=tu.parent_id,
                start=tu.start,
                end=tu.end,
                strand=tu.strand,
                metadata={"inference_method": "rule", "confidence": 0.65, "rule_name": "multi_gene_transcript_unit"},
            )
        )
        used.add(op_id)
        for member in members:
            edges.append(RelationshipEdge(source_id=member, target_id=op_id, kind="part_of_operon", metadata={"inference_method": "rule", "confidence": 0.65}))

    return BioAST(nodes=tuple(nodes), sequence_metadata=ast.sequence_metadata, relationships=_dedupe(edges))


def infer_modules(ast: BioAST) -> BioAST:
    nodes: List[Any] = list(ast.nodes)
    edges: List[RelationshipEdge] = list(ast.semantic_edges)
    used = {node.canonical_id for node in ast.nodes}
    host = next((node for node in ast.nodes if isinstance(node, PlasmidNode)), None)
    parent_id = host.canonical_id if host else next((node.canonical_id for node in ast.nodes if node.parent_id is None), None)

    module_specs = [
        ("replication_module:1", ReplicationModuleNode, ("rep", "ori", "trf")),
        ("selection_module:1", SelectionModuleNode, ("res", "kan", "amp", "tet", "cat")),
        ("cargo_module:1", CargoModuleNode, ("reporter", "cargo", "gfp", "rfp", "cas")),
        ("mobility_module:1", MobilityModuleNode, ("mob", "tra", "oriT", "conj")),
    ]

    genes = [n for n in ast.nodes if isinstance(n, GeneNode)]
    for module_id, module_cls, keywords in module_specs:
        matched = [gene for gene in genes if _gene_matches(gene, keywords)]
        if not matched:
            continue
        if module_id not in used:
            nodes.append(
                module_cls(
                    canonical_id=module_id,
                    parent_id=parent_id,
                    start=min(int(g.start or 0) for g in matched),
                    end=max(int(g.end or g.start or 0) for g in matched),
                    metadata={"inference_method": "rule", "confidence": 0.7, "evidence": [g.gene_id for g in matched], "rule_name": "keyword_module_assignment"},
                )
            )
            used.add(module_id)
        for gene in matched:
            edges.append(RelationshipEdge(source_id=gene.canonical_id, target_id=module_id, kind="part_of_module", metadata={"inference_method": "rule", "confidence": 0.7}))

    # Optional top-level expression cassettes and products.
    for gene in genes:
        cassette_id = f"expression_cassette:{gene.canonical_id}"
        if cassette_id not in used:
            nodes.append(
                ExpressionCassetteNode(
                    canonical_id=cassette_id,
                    parent_id=gene.parent_id,
                    start=gene.start,
                    end=gene.end,
                    strand=gene.strand,
                    metadata={"inference_method": "rule", "confidence": 0.5, "rule_name": "single_gene_cassette"},
                )
            )
            used.add(cassette_id)
        edges.append(RelationshipEdge(source_id=gene.canonical_id, target_id=cassette_id, kind="part_of_module", metadata={"inference_method": "rule", "confidence": 0.5}))

        product_id = f"protein_product:{gene.canonical_id}"
        if product_id not in used:
            nodes.append(
                ProteinProductNode(
                    canonical_id=product_id,
                    parent_id=gene.parent_id,
                    start=gene.start,
                    end=gene.end,
                    strand=gene.strand,
                    metadata={"inference_method": "rule", "confidence": 0.55, "rule_name": "gene_to_protein_projection"},
                )
            )
            used.add(product_id)
        edges.append(RelationshipEdge(source_id=gene.canonical_id, target_id=product_id, kind="encodes", metadata={"inference_method": "rule", "confidence": 0.55}))
        edges.append(RelationshipEdge(source_id=gene.canonical_id, target_id=product_id, kind="produces_protein", metadata={"inference_method": "rule", "confidence": 0.55}))

    return BioAST(nodes=tuple(nodes), sequence_metadata=ast.sequence_metadata, relationships=_dedupe(edges))


def build_bio_regulatory_graph(ast: BioAST, *, sequence: str = "", annotations: Optional[Mapping[str, Mapping[str, Any]]] = None) -> BioAST:
    reg = infer_regulatory_features(ast, sequence=sequence, annotations=annotations)
    tu = infer_transcript_units(reg)
    operons = infer_operons(tu)
    modules = infer_modules(operons)
    return modules


def _scan_for_motif(sequence: str, start: int, end: int) -> Optional[str]:
    if not sequence:
        return None
    window = sequence[max(0, start) : max(0, end) + 1].upper()
    for motif in SHINE_DALGARNO:
        if motif in window:
            return motif
    return None


def _dedupe(edges: Sequence[RelationshipEdge]) -> Tuple[RelationshipEdge, ...]:
    out: List[RelationshipEdge] = []
    seen = set()
    for edge in edges:
        key = (edge.source_id, edge.target_id, edge.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return tuple(out)


def _gene_matches(gene: GeneNode, keywords: Sequence[str]) -> bool:
    haystack = f"{gene.gene_id} {gene.canonical_id} {gene.metadata}".lower()
    return any(keyword.lower() in haystack for keyword in keywords)
