from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from perceptrome.genome_ast import RelationshipEdge


@dataclass(frozen=True)
class GeneDefinition:
    id: str
    dtype: str
    mutation_rate: float
    novelty_weight: float
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    outgoing_relationships: List[RelationshipEdge] = None
    incoming_relationships: List[RelationshipEdge] = None

    def __post_init__(self) -> None:
        if self.outgoing_relationships is None:
            self.outgoing_relationships = []
        if self.incoming_relationships is None:
            self.incoming_relationships = []


class GeneRegistry:
    def __init__(self, definitions: Iterable[GeneDefinition]):
        self._definitions: Dict[str, GeneDefinition] = {}
        for definition in definitions:
            if definition.id in self._definitions:
                raise ValueError(f"Duplicate gene id: {definition.id}")
            self._definitions[definition.id] = definition

    @property
    def definitions(self) -> Dict[str, GeneDefinition]:
        return dict(self._definitions)

    def defaults(self) -> Dict[str, Any]:
        return {gene_id: definition.default for gene_id, definition in self._definitions.items()}

    def hydrate(self, genome_values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        hydrated = self.defaults()
        if not isinstance(genome_values, dict):
            return hydrated

        for gene_id, value in genome_values.items():
            definition = self._definitions.get(gene_id)
            if definition is None:
                continue
            hydrated[gene_id] = self._coerce_value(definition, value)
        return hydrated

    def _coerce_value(self, definition: GeneDefinition, value: Any) -> Any:
        dtype = definition.dtype.lower()
        coerced: Any

        if dtype == "int":
            coerced = int(value)
        elif dtype == "float":
            coerced = float(value)
        elif dtype == "bool":
            coerced = bool(value)
        elif dtype == "str":
            coerced = str(value)
        elif dtype == "choice":
            if definition.choices is None:
                raise ValueError(f"Gene '{definition.id}' is choice dtype but has no choices")
            if value not in definition.choices:
                raise ValueError(
                    f"Gene '{definition.id}' got unsupported value '{value}'. "
                    f"Expected one of: {definition.choices}"
                )
            coerced = value
        else:
            raise ValueError(f"Unsupported dtype '{definition.dtype}' for gene '{definition.id}'")

        if definition.min_value is not None and isinstance(coerced, (int, float)) and coerced < definition.min_value:
            raise ValueError(f"Gene '{definition.id}'={coerced} below min_value={definition.min_value}")
        if definition.max_value is not None and isinstance(coerced, (int, float)) and coerced > definition.max_value:
            raise ValueError(f"Gene '{definition.id}'={coerced} above max_value={definition.max_value}")
        return coerced


@dataclass
class Genome:
    genes: Dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]], registry: GeneRegistry) -> "Genome":
        # Backward compatibility: older payloads may be {"gene_id": value} directly.
        raw_genes = payload.get("genes") if isinstance(payload, dict) and "genes" in payload else payload
        return cls(genes=registry.hydrate(raw_genes if isinstance(raw_genes, dict) else None))

    def to_dict(self) -> Dict[str, Any]:
        return {"genes": dict(self.genes)}


DEFAULT_GENE_REGISTRY = GeneRegistry(
    [
        GeneDefinition(
            id="tokenizer",
            dtype="choice",
            choices=["base", "codon", "aa"],
            mutation_rate=0.02,
            novelty_weight=1.0,
            default="base",
        ),
        GeneDefinition(
            id="window_size",
            dtype="int",
            min_value=64,
            max_value=4096,
            mutation_rate=0.15,
            novelty_weight=1.1,
            default=512,
        ),
        GeneDefinition(
            id="stride",
            dtype="int",
            min_value=1,
            max_value=2048,
            mutation_rate=0.12,
            novelty_weight=1.0,
            default=256,
        ),
        GeneDefinition(
            id="learning_rate",
            dtype="float",
            min_value=1e-6,
            max_value=1.0,
            mutation_rate=0.2,
            novelty_weight=1.3,
            default=1e-3,
        ),
        GeneDefinition(
            id="beta_kl",
            dtype="float",
            min_value=0.0,
            max_value=1.0,
            mutation_rate=0.1,
            novelty_weight=1.0,
            default=1e-3,
        ),
        GeneDefinition(
            id="model_type",
            dtype="choice",
            choices=["mlp", "transformer"],
            mutation_rate=0.03,
            novelty_weight=1.2,
            default="mlp",
        ),
        GeneDefinition(
            id="transformer_layers",
            dtype="int",
            min_value=1,
            max_value=16,
            mutation_rate=0.08,
            novelty_weight=1.15,
            default=4,
            incoming_relationships=[
                RelationshipEdge(
                    type="requires",
                    source="transformer_layers",
                    target="model_type",
                    condition={"gene": "model_type", "op": "eq", "value": "transformer"},
                )
            ],
        ),
    ]
)
