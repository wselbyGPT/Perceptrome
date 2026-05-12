from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union


SCHEMA_VERSION = 4
STABLE_NODE_KINDS = frozenset(
    {
        "genome",
        "plasmid",
        "virus",
        "gene",
        "orf",
        "cds",
        "region",
        "domain",
        "sme",
        "residue",
        "kmer",
        "microfeature",
        "promoter",
        "operator",
        "rbs",
        "terminator",
        "transcript_unit",
        "operon",
        "expression_cassette",
        "replication_module",
        "selection_module",
        "cargo_module",
        "mobility_module",
        "transcript",
        "protein_product",
    }
)
CONTAINMENT_EDGE_KIND = "contains"
SEMANTIC_EDGE_KINDS = frozenset(
    {
        "regulates",
        "overlaps",
        "adjacent_to",
        "homologous_to",
        "promoter_of",
        "operator_of",
        "rbs_for",
        "terminates",
        "part_of_transcript_unit",
        "part_of_operon",
        "part_of_module",
        "encodes",
        "produces_transcript",
        "produces_protein",
        "upstream_of",
        "downstream_of",
        "same_strand_as",
        "opposite_strand_of",
    }
)
REGULATORY_EDGE_KINDS = frozenset(
    {
        "regulates",
        "promoter_of",
        "operator_of",
        "rbs_for",
        "terminates",
        "part_of_transcript_unit",
        "part_of_operon",
        "part_of_module",
        "encodes",
        "produces_transcript",
        "produces_protein",
    }
)
STABLE_EDGE_KINDS = frozenset({CONTAINMENT_EDGE_KIND, *SEMANTIC_EDGE_KINDS})
LEGACY_EDGE_KIND_ALIASES = {
    "parent_child": CONTAINMENT_EDGE_KIND,
    "depends_on": "regulates",
    "supports": "encodes",
}
SECONDARY_TAG_VOCAB = frozenset({"H", "E", "C", "T", "G", "I"})
MOTIF_FAMILY_VOCAB = frozenset({"STRUCTURAL", "REGULATORY", "CATALYTIC", "INTERACTION", "SIGNALING", "OTHER"})
MOTIF_SUBTYPE_VOCAB = frozenset({"BINDING_LOOP", "ACTIVE_SITE", "LOW_COMPLEXITY", "TRANSMEMBRANE", "COILED_COIL", "OTHER"})


def _normalize_optional_float(value: Any, *, field_name: str, minimum: Optional[float] = None, maximum: Optional[float] = None) -> Optional[float]:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric or null") from exc
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return normalized


def _normalize_optional_token(value: Any, *, field_name: str, vocab: Iterable[str]) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().upper()
    if not token:
        return None
    vocab_set = set(vocab)
    if token not in vocab_set:
        raise ValueError(f"{field_name} must be one of {sorted(vocab_set)}")
    return token


def validate_secondary_tag(value: Any) -> Optional[str]:
    return _normalize_optional_token(value, field_name="secondary_tag", vocab=SECONDARY_TAG_VOCAB)


def validate_motif_family(value: Any) -> Optional[str]:
    return _normalize_optional_token(value, field_name="motif_family", vocab=MOTIF_FAMILY_VOCAB)


def validate_motif_subtype(value: Any) -> Optional[str]:
    return _normalize_optional_token(value, field_name="motif_subtype", vocab=MOTIF_SUBTYPE_VOCAB)


def validate_node_kind(value: Any) -> str:
    kind = str(value or "").strip().lower()
    if kind not in STABLE_NODE_KINDS:
        raise ValueError(f"node_type must be one of {sorted(STABLE_NODE_KINDS)}")
    return kind


def validate_edge_kind(value: Any) -> str:
    kind = LEGACY_EDGE_KIND_ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())
    if kind not in STABLE_EDGE_KINDS:
        raise ValueError(f"edge kind must be one of {sorted(STABLE_EDGE_KINDS)}")
    return kind


@dataclass(frozen=True)
class SequenceMetadata:
    accession: str = ""
    length: int = 0
    topology: Optional[str] = None
    molecule_type: Optional[str] = None
    source_format: Optional[str] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError("sequence length must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accession": self.accession,
            "length": self.length,
            "topology": self.topology,
            "molecule_type": self.molecule_type,
            "source_format": self.source_format,
            "checksum": self.checksum,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "SequenceMetadata":
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            accession=str(payload.get("accession", "")),
            length=int(payload.get("length", 0) or 0),
            topology=str(payload["topology"]) if payload.get("topology") is not None else None,
            molecule_type=str(payload["molecule_type"]) if payload.get("molecule_type") is not None else None,
            source_format=str(payload["source_format"]) if payload.get("source_format") is not None else None,
            checksum=str(payload["checksum"]) if payload.get("checksum") is not None else None,
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
        )


@dataclass(frozen=True)
class EnergeticEvolutionaryPayload:
    folding_energy_estimate: Optional[float] = None
    phi_bin: Optional[float] = None
    psi_bin: Optional[float] = None
    conservation_score: Optional[float] = None
    prion_likelihood: Optional[float] = None
    variant_sensitivity: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "folding_energy_estimate", _normalize_optional_float(self.folding_energy_estimate, field_name="folding_energy_estimate"))
        object.__setattr__(self, "phi_bin", _normalize_optional_float(self.phi_bin, field_name="phi_bin", minimum=-180.0, maximum=180.0))
        object.__setattr__(self, "psi_bin", _normalize_optional_float(self.psi_bin, field_name="psi_bin", minimum=-180.0, maximum=180.0))
        object.__setattr__(self, "conservation_score", _normalize_optional_float(self.conservation_score, field_name="conservation_score", minimum=0.0, maximum=1.0))
        object.__setattr__(self, "prion_likelihood", _normalize_optional_float(self.prion_likelihood, field_name="prion_likelihood", minimum=0.0, maximum=1.0))
        object.__setattr__(self, "variant_sensitivity", _normalize_optional_float(self.variant_sensitivity, field_name="variant_sensitivity", minimum=0.0, maximum=1.0))

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {k: getattr(self, k) for k in ("folding_energy_estimate", "phi_bin", "psi_bin", "conservation_score", "prion_likelihood", "variant_sensitivity")}

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "EnergeticEvolutionaryPayload":
        return cls(**dict(payload or {}))


@dataclass(frozen=True)
class AttributeNode:
    key: str
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "value": self.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttributeNode":
        return cls(key=str(payload.get("key", "")), value=payload.get("value"))


TNode = TypeVar("TNode", bound="HierarchicalNode")


@dataclass(frozen=True)
class HierarchicalNode:
    canonical_id: str = ""
    parent_id: Optional[str] = None
    child_ids: Tuple[str, ...] = ()
    start: Optional[int] = None
    end: Optional[int] = None
    strand: Optional[str] = None
    frame: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_type: ClassVar[str] = "node"

    def __post_init__(self) -> None:
        if self.node_type != "node":
            validate_node_kind(self.node_type)
        if not self.canonical_id:
            object.__setattr__(self, "canonical_id", self._infer_canonical_id())

    def _infer_canonical_id(self) -> str:
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type,
            "canonical_id": self.canonical_id,
            "parent_id": self.parent_id,
            "child_ids": list(self.child_ids),
            "start": self.start,
            "end": self.end,
            "strand": self.strand,
            "frame": self.frame,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def _base_kwargs(cls: Type[TNode], payload: Mapping[str, Any]) -> Dict[str, Any]:
        child_ids_raw = payload.get("child_ids")
        return {
            "canonical_id": str(payload.get("canonical_id", "")),
            "parent_id": str(payload["parent_id"]) if payload.get("parent_id") is not None else None,
            "child_ids": tuple(str(v) for v in child_ids_raw) if isinstance(child_ids_raw, list) else (),
            "start": int(payload["start"]) if isinstance(payload.get("start"), int) else None,
            "end": int(payload["end"]) if isinstance(payload.get("end"), int) else None,
            "strand": str(payload["strand"]) if payload.get("strand") is not None else None,
            "frame": int(payload["frame"]) if isinstance(payload.get("frame"), int) else None,
            "metadata": dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
        }


@dataclass(frozen=True)
class GenomeNode(HierarchicalNode):
    node_type: ClassVar[str] = "genome"


@dataclass(frozen=True)
class PlasmidNode(HierarchicalNode):
    node_type: ClassVar[str] = "plasmid"


@dataclass(frozen=True)
class VirusNode(HierarchicalNode):
    node_type: ClassVar[str] = "virus"


@dataclass(frozen=True)
class GeneNode(HierarchicalNode):
    gene_id: str = ""
    dtype: Optional[str] = None
    value: Any = None
    attributes: Tuple[AttributeNode, ...] = ()
    node_type: ClassVar[str] = "gene"

    def _infer_canonical_id(self) -> str:
        return self.gene_id

    def __post_init__(self) -> None:
        if not self.gene_id and self.canonical_id:
            object.__setattr__(self, "gene_id", self.canonical_id)
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload.update({"gene_id": self.gene_id, "dtype": self.dtype, "value": self.value, "attributes": [a.to_dict() for a in self.attributes]})
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneNode":
        attrs = payload.get("attributes")
        return cls(
            **cls._base_kwargs(payload),
            gene_id=str(payload.get("gene_id", "")),
            dtype=payload.get("dtype"),
            value=payload.get("value"),
            attributes=tuple(AttributeNode.from_dict(item) for item in attrs if isinstance(item, Mapping)) if isinstance(attrs, list) else (),
        )

    def get_attribute(self, key: str, default: Any = None) -> Any:
        for attr in self.attributes:
            if attr.key == key:
                return attr.value
        return default


@dataclass(frozen=True)
class ORFNode(HierarchicalNode):
    node_type: ClassVar[str] = "orf"


@dataclass(frozen=True)
class CDSNode(HierarchicalNode):
    node_type: ClassVar[str] = "cds"


@dataclass(frozen=True)
class DomainNode(HierarchicalNode):
    node_type: ClassVar[str] = "domain"


@dataclass(frozen=True)
class RegionNode(HierarchicalNode):
    node_type: ClassVar[str] = "region"


@dataclass(frozen=True)
class SMENode(HierarchicalNode):
    secondary_tag: Optional[str] = None
    motif_family: Optional[str] = None
    motif_subtype: Optional[str] = None
    energetic_evolutionary: EnergeticEvolutionaryPayload = field(default_factory=EnergeticEvolutionaryPayload)
    node_type: ClassVar[str] = "sme"

    def __post_init__(self) -> None:
        object.__setattr__(self, "secondary_tag", validate_secondary_tag(self.secondary_tag))
        object.__setattr__(self, "motif_family", validate_motif_family(self.motif_family))
        object.__setattr__(self, "motif_subtype", validate_motif_subtype(self.motif_subtype))
        if isinstance(self.energetic_evolutionary, Mapping):
            object.__setattr__(self, "energetic_evolutionary", EnergeticEvolutionaryPayload.from_dict(self.energetic_evolutionary))
        elif self.energetic_evolutionary is None:
            object.__setattr__(self, "energetic_evolutionary", EnergeticEvolutionaryPayload())
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "secondary_tag": self.secondary_tag,
                "motif_family": self.motif_family,
                "motif_subtype": self.motif_subtype,
                "energetic_evolutionary": self.energetic_evolutionary.to_dict(),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SMENode":
        return cls(
            **cls._base_kwargs(payload),
            secondary_tag=payload.get("secondary_tag"),
            motif_family=payload.get("motif_family"),
            motif_subtype=payload.get("motif_subtype"),
            energetic_evolutionary=EnergeticEvolutionaryPayload.from_dict(payload.get("energetic_evolutionary")),
        )


@dataclass(frozen=True)
class ResidueNode(HierarchicalNode):
    node_type: ClassVar[str] = "residue"


@dataclass(frozen=True)
class KmerNode(HierarchicalNode):
    node_type: ClassVar[str] = "kmer"


@dataclass(frozen=True)
class MicrofeatureNode(HierarchicalNode):
    node_type: ClassVar[str] = "microfeature"


@dataclass(frozen=True)
class PromoterNode(HierarchicalNode):
    node_type: ClassVar[str] = "promoter"


@dataclass(frozen=True)
class OperatorNode(HierarchicalNode):
    node_type: ClassVar[str] = "operator"


@dataclass(frozen=True)
class RBSNode(HierarchicalNode):
    node_type: ClassVar[str] = "rbs"


@dataclass(frozen=True)
class TerminatorNode(HierarchicalNode):
    node_type: ClassVar[str] = "terminator"


@dataclass(frozen=True)
class TranscriptUnitNode(HierarchicalNode):
    node_type: ClassVar[str] = "transcript_unit"


@dataclass(frozen=True)
class OperonNode(HierarchicalNode):
    node_type: ClassVar[str] = "operon"


@dataclass(frozen=True)
class ExpressionCassetteNode(HierarchicalNode):
    node_type: ClassVar[str] = "expression_cassette"


@dataclass(frozen=True)
class ReplicationModuleNode(HierarchicalNode):
    node_type: ClassVar[str] = "replication_module"


@dataclass(frozen=True)
class SelectionModuleNode(HierarchicalNode):
    node_type: ClassVar[str] = "selection_module"


@dataclass(frozen=True)
class CargoModuleNode(HierarchicalNode):
    node_type: ClassVar[str] = "cargo_module"


@dataclass(frozen=True)
class MobilityModuleNode(HierarchicalNode):
    node_type: ClassVar[str] = "mobility_module"


@dataclass(frozen=True)
class TranscriptNode(HierarchicalNode):
    node_type: ClassVar[str] = "transcript"


@dataclass(frozen=True)
class ProteinProductNode(HierarchicalNode):
    node_type: ClassVar[str] = "protein_product"


ASTNode = Union[
    GenomeNode,
    PlasmidNode,
    VirusNode,
    GeneNode,
    ORFNode,
    CDSNode,
    DomainNode,
    RegionNode,
    SMENode,
    ResidueNode,
    KmerNode,
    MicrofeatureNode,
    PromoterNode,
    OperatorNode,
    RBSNode,
    TerminatorNode,
    TranscriptUnitNode,
    OperonNode,
    ExpressionCassetteNode,
    ReplicationModuleNode,
    SelectionModuleNode,
    CargoModuleNode,
    MobilityModuleNode,
    TranscriptNode,
    ProteinProductNode,
]
_NODE_CLASSES = (
    GenomeNode,
    PlasmidNode,
    VirusNode,
    GeneNode,
    ORFNode,
    CDSNode,
    DomainNode,
    RegionNode,
    SMENode,
    ResidueNode,
    KmerNode,
    MicrofeatureNode,
    PromoterNode,
    OperatorNode,
    RBSNode,
    TerminatorNode,
    TranscriptUnitNode,
    OperonNode,
    ExpressionCassetteNode,
    ReplicationModuleNode,
    SelectionModuleNode,
    CargoModuleNode,
    MobilityModuleNode,
    TranscriptNode,
    ProteinProductNode,
)
_NODE_CLASS_BY_TYPE: Dict[str, Type[HierarchicalNode]] = {cls.node_type: cls for cls in _NODE_CLASSES}


def _node_from_dict(payload: Mapping[str, Any]) -> Optional[ASTNode]:
    node_cls = _NODE_CLASS_BY_TYPE.get(str(payload.get("node_type", "")).lower())
    if node_cls is None:
        return GeneNode.from_dict(payload) if "gene_id" in payload else None
    if node_cls is GeneNode:
        return GeneNode.from_dict(payload)
    if node_cls is SMENode:
        return SMENode.from_dict(payload)
    return node_cls(**node_cls._base_kwargs(payload))


def _normalize_window_positions(window: Optional[Iterable[Union[int, Tuple[int, int]]]]) -> Tuple[Tuple[int, int], ...]:
    if window is None:
        return ()
    normalized: List[Tuple[int, int]] = []
    for item in window:
        if isinstance(item, int):
            start, end = item, item
        elif isinstance(item, tuple) and len(item) == 2 and all(isinstance(v, int) for v in item):
            start, end = item
        else:
            raise ValueError("Window values must be ints or (start, end) integer tuples")
        if start > end:
            raise ValueError("Window start must be <= end")
        normalized.append((start, end))
    return tuple(normalized)


def build_sme_node(
    *,
    parent: Union[DomainNode, RegionNode],
    sme_id: str,
    residue_window: Optional[Iterable[Union[int, Tuple[int, int]]]] = None,
    kmer_window: Optional[Iterable[Union[int, Tuple[int, int]]]] = None,
    secondary_tag: Optional[str] = None,
    motif_family: Optional[str] = None,
    motif_subtype: Optional[str] = None,
    energetic_evolutionary: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[Union[DomainNode, RegionNode], SMENode, Tuple[Union[ResidueNode, KmerNode], ...]]:
    residue_ranges = _normalize_window_positions(residue_window)
    kmer_ranges = _normalize_window_positions(kmer_window)
    child_nodes: List[Union[ResidueNode, KmerNode]] = []
    child_ids: List[str] = []
    for index, (start, end) in enumerate(residue_ranges, start=1):
        cid = f"residue:{sme_id}:{index}"
        child_ids.append(cid)
        child_nodes.append(ResidueNode(canonical_id=cid, parent_id=sme_id, start=start, end=end))
    for index, (start, end) in enumerate(kmer_ranges, start=1):
        cid = f"kmer:{sme_id}:{index}"
        child_ids.append(cid)
        child_nodes.append(KmerNode(canonical_id=cid, parent_id=sme_id, start=start, end=end))
    sme_node = SMENode(
        canonical_id=sme_id,
        parent_id=parent.canonical_id,
        child_ids=tuple(child_ids),
        secondary_tag=secondary_tag,
        motif_family=motif_family,
        motif_subtype=motif_subtype,
        energetic_evolutionary=EnergeticEvolutionaryPayload.from_dict(energetic_evolutionary),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )
    return replace(parent, child_ids=parent.child_ids + (sme_id,)), sme_node, tuple(child_nodes)


@dataclass(frozen=True)
class RelationshipEdge:
    source_id: str
    target_id: str
    kind: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", validate_edge_kind(self.kind))

    @property
    def relation(self) -> str:
        return self.kind

    @property
    def source_gene_id(self) -> str:
        return self.source_id

    @property
    def target_gene_id(self) -> str:
        return self.target_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "kind": self.kind,
            "relation": self.kind,
            "metadata": dict(self.metadata),
            "source_gene_id": self.source_id,
            "target_gene_id": self.target_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelationshipEdge":
        return cls(
            source_id=str(payload.get("source_id", payload.get("source_gene_id", ""))),
            target_id=str(payload.get("target_id", payload.get("target_gene_id", ""))),
            kind=str(payload.get("kind", payload.get("relation", ""))),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
        )


def _compute_containment_edges(nodes: Iterable[ASTNode]) -> Tuple[RelationshipEdge, ...]:
    return tuple(RelationshipEdge(source_id=node.parent_id, target_id=node.canonical_id, kind=CONTAINMENT_EDGE_KIND) for node in nodes if node.parent_id)


def validate_bio_ast_document(ast: "BioAST") -> None:
    node_ids = {node.canonical_id for node in ast.nodes}
    for node in ast.nodes:
        validate_node_kind(node.node_type)
        if node.parent_id is not None and node.parent_id not in node_ids:
            raise ValueError(f"parent_id {node.parent_id!r} missing for node {node.canonical_id!r}")
        for child_id in node.child_ids:
            if child_id not in node_ids:
                raise ValueError(f"child_id {child_id!r} missing for node {node.canonical_id!r}")
    for edge in ast.edges:
        validate_edge_kind(edge.kind)
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            raise ValueError(f"edge endpoints must reference known nodes: {edge.source_id!r}->{edge.target_id!r}")
        if edge.kind == CONTAINMENT_EDGE_KIND and ast.node_by_id[edge.target_id].parent_id != edge.source_id:
            raise ValueError("contains edges must match node parent_id")


@dataclass(frozen=True)
class BioAST:
    genes: Tuple[GeneNode, ...] = ()
    nodes: Tuple[ASTNode, ...] = ()
    relationships: Tuple[RelationshipEdge, ...] = ()
    sequence_metadata: SequenceMetadata = field(default_factory=SequenceMetadata)
    edges: Tuple[RelationshipEdge, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        nodes = self.nodes or tuple(self.genes)
        genes = self.genes or tuple(node for node in nodes if isinstance(node, GeneNode))
        semantic_edges = tuple(edge for edge in (self.relationships or self.edges) if edge.kind != CONTAINMENT_EDGE_KIND)
        containment_edges = _compute_containment_edges(nodes)
        object.__setattr__(self, "nodes", tuple(nodes))
        object.__setattr__(self, "genes", tuple(genes))
        object.__setattr__(self, "relationships", semantic_edges)
        object.__setattr__(self, "edges", containment_edges + semantic_edges)
        if self.sequence_metadata.length == 0 and nodes:
            max_end = max((int(node.end) for node in nodes if node.end is not None), default=0)
            object.__setattr__(self, "sequence_metadata", replace(self.sequence_metadata, length=max_end))
        validate_bio_ast_document(self)

    @property
    def hierarchical_nodes(self) -> Tuple[ASTNode, ...]:
        return self.nodes

    @property
    def semantic_edges(self) -> Tuple[RelationshipEdge, ...]:
        return self.relationships

    @property
    def regulatory_edges(self) -> Tuple[RelationshipEdge, ...]:
        return tuple(edge for edge in self.relationships if edge.kind in REGULATORY_EDGE_KINDS)

    @property
    def containment_edges(self) -> Tuple[RelationshipEdge, ...]:
        return tuple(edge for edge in self.edges if edge.kind == CONTAINMENT_EDGE_KIND)

    @property
    def structural_edges(self) -> Tuple[RelationshipEdge, ...]:
        return self.containment_edges

    @property
    def node_by_id(self) -> Dict[str, ASTNode]:
        return {node.canonical_id: node for node in self.nodes}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence_metadata": self.sequence_metadata.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
            "hierarchical_nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "relationships": [e.to_dict() for e in self.relationships],
            "genes": [g.to_dict() for g in self.genes],
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "BioAST":
        if not isinstance(payload, Mapping):
            return cls()
        version = int(payload.get("schema_version", 1))
        if version < SCHEMA_VERSION:
            payload = migrate_bio_ast_payload(payload)
        nodes_payload = payload.get("nodes", payload.get("hierarchical_nodes"))
        nodes = tuple(node for item in nodes_payload if isinstance(item, Mapping) if (node := _node_from_dict(item)) is not None) if isinstance(nodes_payload, list) else ()
        edges_payload = payload.get("edges")
        edges = tuple(RelationshipEdge.from_dict(item) for item in edges_payload if isinstance(item, Mapping)) if isinstance(edges_payload, list) else ()
        genes_payload = payload.get("genes")
        genes = tuple(GeneNode.from_dict(item) for item in genes_payload if isinstance(item, Mapping)) if isinstance(genes_payload, list) else tuple(node for node in nodes if isinstance(node, GeneNode))
        return cls(
            genes=genes,
            nodes=nodes or tuple(genes),
            edges=edges,
            sequence_metadata=SequenceMetadata.from_dict(payload.get("sequence_metadata")),
            schema_version=SCHEMA_VERSION,
        )


def migrate_bio_ast_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    version = int(payload.get("schema_version", 1))
    if version >= SCHEMA_VERSION:
        return dict(payload)
    nodes_payload = payload.get("nodes")
    if not isinstance(nodes_payload, list):
        genes_payload = payload.get("genes")
        nodes_payload = [dict(item) for item in genes_payload if isinstance(item, Mapping)] if isinstance(genes_payload, list) else []
    relationships = payload.get("relationships")
    semantic_edges = [RelationshipEdge.from_dict(item).to_dict() for item in relationships if isinstance(item, Mapping)] if isinstance(relationships, list) else []
    node_objs = [node for item in nodes_payload if isinstance(item, Mapping) if (node := _node_from_dict(item)) is not None]
    seq_meta = SequenceMetadata.from_dict(payload.get("sequence_metadata"))
    if not seq_meta.length:
        seq_meta = replace(seq_meta, length=max((int(node.end) for node in node_objs if node.end is not None), default=0))
    semantic_edge_objs = tuple(RelationshipEdge.from_dict(edge) for edge in semantic_edges)
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence_metadata": seq_meta.to_dict(),
        "nodes": [node.to_dict() for node in node_objs],
        "edges": [edge.to_dict() for edge in (_compute_containment_edges(node_objs) + semantic_edge_objs)],
        "relationships": [edge.to_dict() for edge in semantic_edge_objs],
        "genes": [node.to_dict() for node in node_objs if isinstance(node, GeneNode)],
    }


def ast_from_flat_genes_payload(payload: Optional[Mapping[str, Any]]) -> BioAST:
    if not isinstance(payload, Mapping):
        return BioAST()
    if "schema_version" in payload or "nodes" in payload:
        return BioAST.from_dict(payload)
    raw_genes = payload.get("genes") if "genes" in payload else payload
    if not isinstance(raw_genes, Mapping):
        return BioAST()
    genes = tuple(GeneNode(gene_id=str(gid), canonical_id=str(gid), value=value) for gid, value in raw_genes.items())
    return BioAST(genes=genes, nodes=genes)


def ast_to_flat_genes_payload(ast: BioAST) -> Dict[str, Any]:
    return {"genes": {gene.gene_id: gene.value for gene in ast.genes}}


def registry_ast_from_definitions(definitions: Iterable[Mapping[str, Any]]) -> BioAST:
    nodes: List[GeneNode] = []
    edges: List[RelationshipEdge] = []
    for definition in definitions:
        gene_id = str(definition["id"])
        nodes.append(
            GeneNode(
                gene_id=gene_id,
                canonical_id=gene_id,
                dtype=str(definition["dtype"]),
                value=definition.get("default"),
                metadata={
                    "mutation_rate": definition["mutation_rate"],
                    "novelty_weight": definition["novelty_weight"],
                    "min_value": definition.get("min_value"),
                    "max_value": definition.get("max_value"),
                    "choices": definition.get("choices"),
                },
            )
        )
        if definition.get("depends_on"):
            edges.append(RelationshipEdge(source_id=gene_id, target_id=str(definition.get("depends_on")), kind="regulates"))
    return BioAST(genes=tuple(nodes), nodes=tuple(nodes), relationships=tuple(edges))


def definition_map_from_registry_ast(ast: BioAST) -> Dict[str, Dict[str, Any]]:
    deps = {edge.source_id: edge.target_id for edge in ast.relationships if edge.kind == "regulates"}
    return {
        node.gene_id: {
            "id": node.gene_id,
            "dtype": node.dtype,
            "mutation_rate": float(node.metadata.get("mutation_rate", 0.0)),
            "novelty_weight": float(node.metadata.get("novelty_weight", 0.0)),
            "default": node.value,
            "min_value": node.metadata.get("min_value"),
            "max_value": node.metadata.get("max_value"),
            "choices": node.metadata.get("choices"),
            "depends_on": deps.get(node.gene_id),
        }
        for node in ast.genes
    }


def sequence_checksum(sequence: str) -> str:
    return hashlib.sha256((sequence or "").encode("utf-8")).hexdigest()
