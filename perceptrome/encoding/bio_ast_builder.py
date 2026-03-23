from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from perceptrome.bio_ast import (
    ASTNode,
    BioAST,
    SequenceMetadata,
    sequence_checksum,
    CDSNode,
    DomainNode,
    GeneNode,
    GenomeNode,
    KmerNode,
    MicrofeatureNode,
    ORFNode,
    PlasmidNode,
    RegionNode,
    ResidueNode,
    SMENode,
    VirusNode,
)
from perceptrome.encoding.bio_ast_edges import derive_semantic_edges
from perceptrome.encoding.encode import encode_sequence_one_hot
from perceptrome.encoding.genbank_features import CDSFeature
from perceptrome.encoding.orf import translate_orf
from perceptrome.encoding.parse import reverse_complement


NODE_TYPE_TO_INT = {
    "genome": 0,
    "plasmid": 1,
    "virus": 2,
    "gene": 3,
    "orf": 4,
    "cds": 5,
    "region": 6,
    "domain": 7,
    "sme": 8,
    "residue": 9,
    "kmer": 10,
    "microfeature": 11,
}


@dataclass(frozen=True)
class BuiltBioAST:
    ast: BioAST
    sequence: str

    def to_serialized_paths(self, include_metadata: bool = True) -> List[Dict[str, Any]]:
        children: Dict[str, List[str]] = {}
        node_by_id: Dict[str, ASTNode] = {}
        roots: List[str] = []
        for node in self.ast.nodes:
            node_by_id[node.canonical_id] = node
            children.setdefault(node.canonical_id, [])
            if node.parent_id:
                children.setdefault(node.parent_id, []).append(node.canonical_id)
            else:
                roots.append(node.canonical_id)

        paths: List[Dict[str, Any]] = []

        def walk(node_id: str, stack: List[str]) -> None:
            node = node_by_id[node_id]
            next_stack = stack + [f"{node.node_type}:{node.canonical_id}"]
            node_children = children.get(node_id, [])
            if not node_children:
                payload: Dict[str, Any] = {
                    "path": list(next_stack),
                    "types": [part.split(":", 1)[0] for part in next_stack],
                }
                if include_metadata:
                    payload["leaf_metadata"] = dict(node.metadata)
                paths.append(payload)
                return
            for child in node_children:
                walk(child, next_stack)

        for root_id in roots:
            walk(root_id, [])
        return paths

    def to_tree_message_passing_tensors(self) -> Dict[str, np.ndarray]:
        node_by_id = {node.canonical_id: node for node in self.ast.nodes}
        ordered_nodes = list(self.ast.nodes)
        idx = {node.canonical_id: i for i, node in enumerate(ordered_nodes)}

        edge_pairs: List[Tuple[int, int]] = []
        for node in ordered_nodes:
            if node.parent_id and node.parent_id in idx:
                edge_pairs.append((idx[node.parent_id], idx[node.canonical_id]))

        if edge_pairs:
            edge_index = np.array(edge_pairs, dtype="int64").T
        else:
            edge_index = np.zeros((2, 0), dtype="int64")

        node_type_ids = np.array([NODE_TYPE_TO_INT.get(node.node_type, -1) for node in ordered_nodes], dtype="int64")
        coords = np.array(
            [[node.start if node.start is not None else -1, node.end if node.end is not None else -1] for node in ordered_nodes],
            dtype="int64",
        )
        strand = np.array([1 if node.strand == "+" else (-1 if node.strand == "-" else 0) for node in ordered_nodes], dtype="int64")

        return {
            "node_type_ids": node_type_ids,
            "coords": coords,
            "strand": strand,
            "edge_index": edge_index,
        }

    def to_local_windows(self, window_size: int = 128, stride: int = 64) -> np.ndarray:
        windows: List[np.ndarray] = []
        for node in self.ast.nodes:
            if node.node_type not in {"sme", "domain", "region", "cds"}:
                continue
            if node.start is None or node.end is None:
                continue
            start = max(1, int(node.start))
            end = min(len(self.sequence), int(node.end))
            if end < start:
                continue
            subseq = self.sequence[start - 1 : end]
            windows.append(encode_sequence_one_hot(subseq, window_size=window_size, stride=stride))

        if not windows:
            return encode_sequence_one_hot(self.sequence, window_size=window_size, stride=stride)
        return np.concatenate(windows, axis=0)


class BioASTBuilder:
    def build(
        self,
        *,
        sequence: str,
        cds_features: Optional[Sequence[CDSFeature]] = None,
        feature_annotations: Optional[Mapping[str, Mapping[str, Any]]] = None,
        top_level_type: str = "genome",
        accession: str = "unknown",
        source_format: str = "unknown",
        molecule_type: str = "DNA",
        topology: str | None = None,
    ) -> BuiltBioAST:
        seq = (sequence or "").upper()
        if not seq:
            raise ValueError("sequence cannot be empty")

        top = self._build_top_node(top_level_type=top_level_type, accession=accession, seq_len=len(seq))
        nodes: List[ASTNode] = [top]

        features = list(cds_features) if cds_features else self._fallback_orf_features(seq)
        features.sort(key=lambda item: (item.start, item.end))

        top_child_ids: List[str] = []
        for idx, feature in enumerate(features, start=1):
            gene_id = f"gene:{accession}:{idx}"
            orf_id = f"orf:{accession}:{idx}"
            cds_id = f"cds:{accession}:{idx}"
            strand = "+" if int(feature.strand) >= 0 else "-"

            gene = GeneNode(
                canonical_id=gene_id,
                gene_id=feature.gene_or_locus_tag or gene_id,
                parent_id=top.canonical_id,
                child_ids=(orf_id,),
                start=int(feature.start),
                end=int(feature.end),
                strand=strand,
                metadata={"product": feature.product},
            )
            orf = ORFNode(
                canonical_id=orf_id,
                parent_id=gene_id,
                child_ids=(cds_id,),
                start=int(feature.start),
                end=int(feature.end),
                strand=strand,
                frame=0,
            )
            cds = CDSNode(
                canonical_id=cds_id,
                parent_id=orf_id,
                start=int(feature.start),
                end=int(feature.end),
                strand=strand,
                frame=0,
            )
            nodes.extend([gene, orf, cds])
            top_child_ids.append(gene_id)

            aa = self._translate_feature(seq, feature)
            ann = dict((feature_annotations or {}).get(feature.gene_or_locus_tag, {}))
            derived_nodes, updated_cds = self._derive_regions_domains_and_smes(accession, idx, cds, aa, ann)
            nodes[-1] = updated_cds
            nodes.extend(derived_nodes)

        nodes[0] = self._replace_child_ids(top, tuple(top_child_ids))
        base_ast = BioAST(
            nodes=tuple(nodes),
            sequence_metadata=SequenceMetadata(
                accession=str(accession),
                length=len(seq),
                topology=str(topology or ("circular" if str(top_level_type).lower() == "plasmid" else "linear")),
                molecule_type=str(molecule_type),
                source_format=str(source_format),
                checksum=sequence_checksum(seq),
                metadata={"top_level_type": str(top_level_type).lower()},
            ),
        )
        ast = BioAST(
            nodes=base_ast.nodes,
            sequence_metadata=base_ast.sequence_metadata,
            relationships=derive_semantic_edges(base_ast, feature_annotations=feature_annotations),
        )
        return BuiltBioAST(ast=ast, sequence=seq)

    def _build_top_node(self, *, top_level_type: str, accession: str, seq_len: int) -> ASTNode:
        canonical = f"{top_level_type}:{accession}"
        lower = (top_level_type or "genome").lower()
        if lower == "plasmid":
            return PlasmidNode(canonical_id=canonical, start=1, end=seq_len)
        if lower == "virus":
            return VirusNode(canonical_id=canonical, start=1, end=seq_len)
        return GenomeNode(canonical_id=canonical, start=1, end=seq_len)

    def _replace_child_ids(self, node: ASTNode, child_ids: Tuple[str, ...]) -> ASTNode:
        payload = node.to_dict()
        payload["child_ids"] = list(child_ids)
        return type(node)(**type(node)._base_kwargs(payload))

    def _translate_feature(self, sequence: str, feature: CDSFeature) -> str:
        start = max(1, int(feature.start))
        end = min(len(sequence), int(feature.end))
        dna = sequence[start - 1 : end]
        if int(feature.strand) < 0:
            dna = reverse_complement(dna)
        trim = len(dna) % 3
        if trim:
            dna = dna[:-trim]
        if not dna:
            return ""
        return translate_orf(dna)

    def _derive_regions_domains_and_smes(
        self,
        accession: str,
        idx: int,
        cds: CDSNode,
        aa: str,
        ann: Mapping[str, Any],
    ) -> Tuple[List[ASTNode], CDSNode]:
        out: List[ASTNode] = []

        region_specs = ann.get("regions") if isinstance(ann.get("regions"), list) else []
        domain_specs = ann.get("domains") if isinstance(ann.get("domains"), list) else []
        if not region_specs:
            region_specs = self._fallback_segments(cds.start, cds.end, max_parts=2)
        if not domain_specs:
            domain_specs = self._fallback_segments(cds.start, cds.end, max_parts=3)

        region_ids: List[str] = []
        for r_idx, (r_start, r_end) in enumerate(region_specs, start=1):
            region_id = f"region:{accession}:{idx}:{r_idx}"
            region = RegionNode(canonical_id=region_id, parent_id=cds.canonical_id, start=r_start, end=r_end)
            out.append(region)
            region_ids.append(region_id)

        domain_ids: List[str] = []
        for d_idx, (d_start, d_end) in enumerate(domain_specs, start=1):
            parent_id = region_ids[min(d_idx - 1, len(region_ids) - 1)] if region_ids else cds.canonical_id
            domain_id = f"domain:{accession}:{idx}:{d_idx}"
            domain = DomainNode(canonical_id=domain_id, parent_id=parent_id, start=d_start, end=d_end)
            out.append(domain)
            domain_ids.append(domain_id)

            sme_id = f"sme:{accession}:{idx}:{d_idx}"
            sme = SMENode(
                canonical_id=sme_id,
                parent_id=domain_id,
                start=d_start,
                end=d_end,
                metadata={"aa_context": aa[max(0, d_idx - 1) * 5 : max(0, d_idx - 1) * 5 + 12]},
            )
            out.append(sme)

            residue = ResidueNode(canonical_id=f"residue:{accession}:{idx}:{d_idx}", parent_id=sme_id, start=d_start, end=d_start)
            kmer = KmerNode(canonical_id=f"kmer:{accession}:{idx}:{d_idx}", parent_id=sme_id, start=d_start, end=min(d_end, d_start + 2))
            hydro = self._hydrophobic_fraction(aa)
            micro = MicrofeatureNode(
                canonical_id=f"microfeature:{accession}:{idx}:{d_idx}",
                parent_id=sme_id,
                start=d_start,
                end=d_end,
                metadata={"hydrophobic_fraction": hydro},
            )
            out.extend([residue, kmer, micro])

        updated_cds = CDSNode(
            canonical_id=cds.canonical_id,
            parent_id=cds.parent_id,
            child_ids=tuple(region_ids if region_ids else domain_ids),
            start=cds.start,
            end=cds.end,
            strand=cds.strand,
            frame=cds.frame,
            metadata=dict(cds.metadata),
        )
        return out, updated_cds

    def _fallback_segments(self, start: Optional[int], end: Optional[int], max_parts: int) -> List[Tuple[int, int]]:
        if start is None or end is None or end < start:
            return []
        span = end - start + 1
        if span <= 1:
            return [(start, end)]
        parts = min(max_parts, max(1, span // 30 + 1))
        chunk = max(1, span // parts)
        out: List[Tuple[int, int]] = []
        cur = start
        for i in range(parts):
            nxt = end if i == parts - 1 else min(end, cur + chunk - 1)
            out.append((cur, nxt))
            cur = nxt + 1
            if cur > end:
                break
        return out

    def _fallback_orf_features(self, seq: str) -> List[CDSFeature]:
        seq = seq.upper()
        feats: List[CDSFeature] = []
        stop_codons = {"TAA", "TAG", "TGA"}
        min_bp = 90
        for frame in (0, 1, 2):
            i = frame
            while i + 2 < len(seq):
                codon = seq[i : i + 3]
                if codon != "ATG":
                    i += 3
                    continue
                j = i + 3
                while j + 2 < len(seq):
                    c2 = seq[j : j + 3]
                    if c2 in stop_codons:
                        span = (j + 3) - i
                        if span >= min_bp:
                            feats.append(
                                CDSFeature(
                                    start=i + 1,
                                    end=j + 3,
                                    strand=1,
                                    gene_or_locus_tag=f"ORF_{len(feats)+1}",
                                    product="orf_prediction",
                                    protein_length=max(0, span // 3 - 1),
                                    translation_source="inferred",
                                )
                            )
                        i = j + 3
                        break
                    j += 3
                else:
                    i += 3
        if not feats:
            feats.append(
                CDSFeature(
                    start=1,
                    end=len(seq),
                    strand=1,
                    gene_or_locus_tag="ORF_1",
                    product="fallback_full_length_orf",
                    protein_length=max(0, len(seq) // 3),
                    translation_source="inferred",
                )
            )
        return feats

    def _hydrophobic_fraction(self, aa: str) -> float:
        if not aa:
            return 0.0
        hydro = set("AVLIMFWY")
        return float(sum(1 for c in aa if c in hydro) / len(aa))
