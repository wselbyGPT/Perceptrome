from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union


SCHEMA_VERSION = 2

SECONDARY_TAG_VOCAB = frozenset({"H", "E", "C", "T", "G", "I"})
MOTIF_FAMILY_VOCAB = frozenset(
    {
        "STRUCTURAL",
        "REGULATORY",
        "CATALYTIC",
        "INTERACTION",
        "SIGNALING",
        "OTHER",
    }
)
MOTIF_SUBTYPE_VOCAB = frozenset(
    {
        "BINDING_LOOP",
        "ACTIVE_SITE",
        "LOW_COMPLEXITY",
        "TRANSMEMBRANE",
        "COILED_COIL",
        "OTHER",
    }
)


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


@dataclass(frozen=True)
class EnergeticEvolutionaryPayload:
    folding_energy_estimate: Optional[float] = None
    phi_bin: Optional[float] = None
    psi_bin: Optional[float] = None
    conservation_score: Optional[float] = None
    prion_likelihood: Optional[float] = None
    variant_sensitivity: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "folding_energy_estimate",
            _normalize_optional_float(self.folding_energy_estimate, field_name="folding_energy_estimate"),
        )
        object.__setattr__(self, "phi_bin", _normalize_optional_float(self.phi_bin, field_name="phi_bin", minimum=-180.0, maximum=180.0))
        object.__setattr__(self, "psi_bin", _normalize_optional_float(self.psi_bin, field_name="psi_bin", minimum=-180.0, maximum=180.0))
        object.__setattr__(
            self,
            "conservation_score",
            _normalize_optional_float(self.conservation_score, field_name="conservation_score", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "prion_likelihood",
            _normalize_optional_float(self.prion_likelihood, field_name="prion_likelihood", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "variant_sensitivity",
            _normalize_optional_float(self.variant_sensitivity, field_name="variant_sensitivity", minimum=0.0, maximum=1.0),
        )

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "folding_energy_estimate": self.folding_energy_estimate,
            "phi_bin": self.phi_bin,
            "psi_bin": self.psi_bin,
            "conservation_score": self.conservation_score,
            "prion_likelihood": self.prion_likelihood,
            "variant_sensitivity": self.variant_sensitivity,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "EnergeticEvolutionaryPayload":
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            folding_energy_estimate=payload.get("folding_energy_estimate"),
            phi_bin=payload.get("phi_bin"),
            psi_bin=payload.get("psi_bin"),
            conservation_score=payload.get("conservation_score"),
            prion_likelihood=payload.get("prion_likelihood"),
            variant_sensitivity=payload.get("variant_sensitivity"),
        )


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
        child_ids = (
            tuple(str(child_id) for child_id in child_ids_raw)
            if isinstance(child_ids_raw, list)
            else ()
        )
        metadata = payload.get("metadata")
        return {
            "canonical_id": str(payload.get("canonical_id", "")),
            "parent_id": str(payload["parent_id"]) if payload.get("parent_id") is not None else None,
            "child_ids": child_ids,
            "start": int(payload["start"]) if isinstance(payload.get("start"), int) else None,
            "end": int(payload["end"]) if isinstance(payload.get("end"), int) else None,
            "strand": str(payload["strand"]) if payload.get("strand") is not None else None,
            "frame": int(payload["frame"]) if isinstance(payload.get("frame"), int) else None,
            "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
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
        payload.update(
            {
                "gene_id": self.gene_id,
                "dtype": self.dtype,
                "value": self.value,
                "attributes": [attr.to_dict() for attr in self.attributes],
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneNode":
        attrs = payload.get("attributes")
        return cls(
            **cls._base_kwargs(payload),
            gene_id=str(payload.get("gene_id", "")),
            dtype=payload.get("dtype"),
            value=payload.get("value"),
            attributes=tuple(
                AttributeNode.from_dict(item)
                for item in attrs
                if isinstance(item, Mapping)
            )
            if isinstance(attrs, list)
            else (),
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
]


_NODE_CLASS_BY_TYPE: Dict[str, Type[HierarchicalNode]] = {
    GenomeNode.node_type: GenomeNode,
    PlasmidNode.node_type: PlasmidNode,
    VirusNode.node_type: VirusNode,
    GeneNode.node_type: GeneNode,
    ORFNode.node_type: ORFNode,
    CDSNode.node_type: CDSNode,
    DomainNode.node_type: DomainNode,
    RegionNode.node_type: RegionNode,
    SMENode.node_type: SMENode,
    ResidueNode.node_type: ResidueNode,
    KmerNode.node_type: KmerNode,
    MicrofeatureNode.node_type: MicrofeatureNode,
}


def _node_from_dict(payload: Mapping[str, Any]) -> Optional[ASTNode]:
    node_type = str(payload.get("node_type", "")).lower()
    node_cls = _NODE_CLASS_BY_TYPE.get(node_type)
    if node_cls is None:
        if "gene_id" in payload:
            return GeneNode.from_dict(payload)
        return None

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
        elif isinstance(item, tuple) and len(item) == 2 and all(isinstance(value, int) for value in item):
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
    parent_id = parent.canonical_id
    residue_ranges = _normalize_window_positions(residue_window)
    kmer_ranges = _normalize_window_positions(kmer_window)

    child_nodes: List[Union[ResidueNode, KmerNode]] = []
    child_ids: List[str] = []
    for index, (start, end) in enumerate(residue_ranges, start=1):
        canonical_id = f"residue:{sme_id}:{index}"
        child_ids.append(canonical_id)
        child_nodes.append(ResidueNode(canonical_id=canonical_id, parent_id=sme_id, start=start, end=end))
    for index, (start, end) in enumerate(kmer_ranges, start=1):
        canonical_id = f"kmer:{sme_id}:{index}"
        child_ids.append(canonical_id)
        child_nodes.append(KmerNode(canonical_id=canonical_id, parent_id=sme_id, start=start, end=end))

    sme_node = SMENode(
        canonical_id=sme_id,
        parent_id=parent_id,
        child_ids=tuple(child_ids),
        secondary_tag=secondary_tag,
        motif_family=motif_family,
        motif_subtype=motif_subtype,
        energetic_evolutionary=EnergeticEvolutionaryPayload.from_dict(energetic_evolutionary),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )
    updated_parent = replace(parent, child_ids=parent.child_ids + (sme_id,))
    return updated_parent, sme_node, tuple(child_nodes)


@dataclass(frozen=True)
class RelationshipEdge:
    source_gene_id: str
    target_gene_id: str
    relation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_gene_id": self.source_gene_id,
            "target_gene_id": self.target_gene_id,
            "relation": self.relation,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelationshipEdge":
        return cls(
            source_gene_id=str(payload.get("source_gene_id", "")),
            target_gene_id=str(payload.get("target_gene_id", "")),
            relation=str(payload.get("relation", "")),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
        )


@dataclass(frozen=True)
class BioAST:
    genes: Tuple[GeneNode, ...] = ()
    nodes: Tuple[ASTNode, ...] = ()
    relationships: Tuple[RelationshipEdge, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.nodes and self.genes:
            object.__setattr__(self, "nodes", tuple(self.genes))
        elif self.nodes and not self.genes:
            object.__setattr__(
                self,
                "genes",
                tuple(node for node in self.nodes if isinstance(node, GeneNode)),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "genes": [gene.to_dict() for gene in self.genes],
            "relationships": [edge.to_dict() for edge in self.relationships],
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "BioAST":
        if not isinstance(payload, Mapping):
            return cls()

        schema_version = int(payload.get("schema_version", 1))
        rel_payload = payload.get("relationships")
        relationships = (
            tuple(RelationshipEdge.from_dict(item) for item in rel_payload if isinstance(item, Mapping))
            if isinstance(rel_payload, list)
            else ()
        )

        if schema_version < SCHEMA_VERSION:
            genes_payload = payload.get("genes")
            genes = (
                tuple(GeneNode.from_dict(item) for item in genes_payload if isinstance(item, Mapping))
                if isinstance(genes_payload, list)
                else ()
            )
            return cls(genes=genes, nodes=tuple(genes), relationships=relationships, schema_version=SCHEMA_VERSION)

        nodes_payload = payload.get("nodes")
        nodes = (
            tuple(node for item in nodes_payload if isinstance(item, Mapping) if (node := _node_from_dict(item)) is not None)
            if isinstance(nodes_payload, list)
            else ()
        )

        genes_payload = payload.get("genes")
        genes = (
            tuple(GeneNode.from_dict(item) for item in genes_payload if isinstance(item, Mapping))
            if isinstance(genes_payload, list)
            else tuple(node for node in nodes if isinstance(node, GeneNode))
        )

        return cls(genes=genes, nodes=nodes or tuple(genes), relationships=relationships, schema_version=schema_version)


def ast_from_flat_genes_payload(payload: Optional[Mapping[str, Any]]) -> BioAST:
    if not isinstance(payload, Mapping):
        return BioAST()

    if "schema_version" in payload or "nodes" in payload:
        return BioAST.from_dict(payload)

    raw_genes = payload.get("genes") if "genes" in payload else payload
    if not isinstance(raw_genes, Mapping):
        return BioAST()

    genes = tuple(
        GeneNode(gene_id=str(gene_id), canonical_id=str(gene_id), value=value)
        for gene_id, value in raw_genes.items()
    )
    return BioAST(genes=genes, nodes=genes)


def ast_to_flat_genes_payload(ast: BioAST) -> Dict[str, Any]:
    return {"genes": {gene.gene_id: gene.value for gene in ast.genes}}


def registry_ast_from_definitions(definitions: Iterable[Mapping[str, Any]]) -> BioAST:
    nodes: List[GeneNode] = []
    edges: List[RelationshipEdge] = []

    for definition in definitions:
        gene_id = str(definition["id"])
        metadata = {
            "mutation_rate": definition["mutation_rate"],
            "novelty_weight": definition["novelty_weight"],
            "min_value": definition.get("min_value"),
            "max_value": definition.get("max_value"),
            "choices": definition.get("choices"),
        }
        nodes.append(
            GeneNode(
                gene_id=gene_id,
                canonical_id=gene_id,
                dtype=str(definition["dtype"]),
                value=definition.get("default"),
                metadata=metadata,
            )
        )
        depends_on = definition.get("depends_on")
        if depends_on:
            edges.append(
                RelationshipEdge(
                    source_gene_id=gene_id,
                    target_gene_id=str(depends_on),
                    relation="depends_on",
                )
            )

    return BioAST(genes=tuple(nodes), nodes=tuple(nodes), relationships=tuple(edges))


def definition_map_from_registry_ast(ast: BioAST) -> Dict[str, Dict[str, Any]]:
    deps: Dict[str, str] = {
        edge.source_gene_id: edge.target_gene_id for edge in ast.relationships if edge.relation == "depends_on"
    }
    result: Dict[str, Dict[str, Any]] = {}
    for node in ast.genes:
        result[node.gene_id] = {
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
    return result
