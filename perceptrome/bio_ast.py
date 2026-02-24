from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class AttributeNode:
    key: str
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "value": self.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttributeNode":
        return cls(key=str(payload.get("key", "")), value=payload.get("value"))


@dataclass(frozen=True)
class GeneNode:
    gene_id: str
    dtype: Optional[str] = None
    value: Any = None
    attributes: Tuple[AttributeNode, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene_id": self.gene_id,
            "dtype": self.dtype,
            "value": self.value,
            "attributes": [attr.to_dict() for attr in self.attributes],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneNode":
        attrs = payload.get("attributes")
        return cls(
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
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
        )

    def get_attribute(self, key: str, default: Any = None) -> Any:
        for attr in self.attributes:
            if attr.key == key:
                return attr.value
        return default


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
    relationships: Tuple[RelationshipEdge, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genes": [gene.to_dict() for gene in self.genes],
            "relationships": [edge.to_dict() for edge in self.relationships],
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "BioAST":
        if not isinstance(payload, Mapping):
            return cls()
        genes_payload = payload.get("genes")
        rel_payload = payload.get("relationships")
        genes = (
            tuple(GeneNode.from_dict(item) for item in genes_payload if isinstance(item, Mapping))
            if isinstance(genes_payload, list)
            else ()
        )
        relationships = (
            tuple(RelationshipEdge.from_dict(item) for item in rel_payload if isinstance(item, Mapping))
            if isinstance(rel_payload, list)
            else ()
        )
        return cls(genes=genes, relationships=relationships)


def ast_from_flat_genes_payload(payload: Optional[Mapping[str, Any]]) -> BioAST:
    if not isinstance(payload, Mapping):
        return BioAST()

    raw_genes = payload.get("genes") if "genes" in payload else payload
    if not isinstance(raw_genes, Mapping):
        return BioAST()

    return BioAST(genes=tuple(GeneNode(gene_id=str(gene_id), value=value) for gene_id, value in raw_genes.items()))


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

    return BioAST(genes=tuple(nodes), relationships=tuple(edges))


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
