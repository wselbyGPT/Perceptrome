from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union


SCHEMA_VERSION = 2


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
    node_type: ClassVar[str] = "sme"


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
    return node_cls(**node_cls._base_kwargs(payload))


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
