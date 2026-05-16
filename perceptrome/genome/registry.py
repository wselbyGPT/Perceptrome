from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from perceptrome.model_catalog import DNA_GENERATIVE_MODEL_TYPES
from perceptrome.bio_ast import (
    ast_from_flat_genes_payload,
    ast_to_flat_genes_payload,
    definition_map_from_registry_ast,
    registry_ast_from_definitions,
)


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
    depends_on: Optional[str] = None


class GeneRegistry:
    def __init__(self, definitions: Iterable[GeneDefinition]):
        serialized = []
        seen: set[str] = set()
        for definition in definitions:
            if definition.id in seen:
                raise ValueError(f"Duplicate gene id: {definition.id}")
            seen.add(definition.id)
            serialized.append(
                {
                    "id": definition.id,
                    "dtype": definition.dtype,
                    "mutation_rate": definition.mutation_rate,
                    "novelty_weight": definition.novelty_weight,
                    "default": definition.default,
                    "min_value": definition.min_value,
                    "max_value": definition.max_value,
                    "choices": definition.choices,
                    "depends_on": definition.depends_on,
                }
            )
        self._ast = registry_ast_from_definitions(serialized)

    @property
    def definitions(self) -> Dict[str, GeneDefinition]:
        definition_map = definition_map_from_registry_ast(self._ast)
        return {gene_id: GeneDefinition(**payload) for gene_id, payload in definition_map.items()}

    def defaults(self) -> Dict[str, Any]:
        return {gene_id: definition.default for gene_id, definition in self.definitions.items()}

    def hydrate(self, genome_values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        definitions = self.definitions
        hydrated = {gene_id: definition.default for gene_id, definition in definitions.items()}
        if not isinstance(genome_values, dict):
            return hydrated

        for gene_id, value in genome_values.items():
            definition = definitions.get(gene_id)
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
        ast = ast_from_flat_genes_payload(payload)
        return cls(genes=registry.hydrate({gene.gene_id: gene.value for gene in ast.genes}))

    def to_dict(self) -> Dict[str, Any]:
        ast = ast_from_flat_genes_payload({"genes": dict(self.genes)})
        return ast_to_flat_genes_payload(ast)


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
            choices=list(DNA_GENERATIVE_MODEL_TYPES),
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
            depends_on="model_type",
        ),
    ]
)
