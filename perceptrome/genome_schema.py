from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


CURRENT_GENOME_SCHEMA_VERSION = 3

AST_NODE_TYPE_GENE = "gene"
AST_EDGE_TYPE_DEPENDS_ON = "depends_on"
AST_EDGE_TYPE_CONFLICTS_WITH = "conflicts_with"


def _default_ast_edges(genes: Mapping[str, Any]) -> List[Dict[str, str]]:
    edges: List[Dict[str, str]] = []
    model_type = "model_type"
    for dependent in ("transformer_d_model", "transformer_nhead", "transformer_layers", "transformer_dropout"):
        if dependent in genes and model_type in genes:
            edges.append({"source": dependent, "target": model_type, "type": AST_EDGE_TYPE_DEPENDS_ON})
    return edges


def _default_genes(
    *,
    tokenizer: str,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
    loss_type: str,
    model_type: str,
    transformer_d_model: int,
    transformer_nhead: int,
    transformer_layers: int,
    transformer_dropout: float,
    learning_rate: float,
    beta_kl: float,
) -> Dict[str, Any]:
    return {
        "tokenizer": str(tokenizer).lower(),
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
        "hidden_dim": int(hidden_dim),
        "loss_type": str(loss_type).lower(),
        "model_type": str(model_type).lower(),
        "transformer_d_model": int(transformer_d_model),
        "transformer_nhead": int(transformer_nhead),
        "transformer_layers": int(transformer_layers),
        "transformer_dropout": float(transformer_dropout),
        "learning_rate": float(learning_rate),
        "beta_kl": float(beta_kl),
    }


def _normalize_gene_id(gene_id: str) -> str:
    deprecated_map = {
        "latent_dim": "hidden_dim",
        "lr": "learning_rate",
        "dropout": "transformer_dropout",
        "dropout_pct": "transformer_dropout",
    }
    return deprecated_map.get(gene_id, gene_id)


def _transform_old_ranges(genes: Dict[str, Any]) -> None:
    # Older snapshots used percentages for dropout.
    if "transformer_dropout" in genes:
        d = float(genes["transformer_dropout"])
        if d > 1.0:
            genes["transformer_dropout"] = d / 100.0

    # Older snapshots could store normalized LR in [0,1] as learning_rate_norm.
    if "learning_rate_norm" in genes and "learning_rate" not in genes:
        n = float(genes["learning_rate_norm"])
        n = max(0.0, min(1.0, n))
        min_lr = 1e-5
        max_lr = 1e-2
        genes["learning_rate"] = min_lr * ((max_lr / min_lr) ** n)


def _nodes_from_genes(genes: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [{"id": gene_id, "kind": AST_NODE_TYPE_GENE, "value": value} for gene_id, value in genes.items()]


def _genes_from_nodes(nodes: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    genes: Dict[str, Any] = {}
    for node in nodes:
        gene_id = str(node.get("id", "")).strip()
        if not gene_id:
            continue
        genes[_normalize_gene_id(gene_id)] = node.get("value")
    return genes


def _edge_tuple(edge: Mapping[str, Any]) -> tuple[str, str, str]:
    source = str(edge.get("source", "")).strip()
    target = str(edge.get("target", "")).strip()
    edge_type = str(edge.get("type", "")).strip().lower()
    return source, target, edge_type


def _normalize_edges(payload: Mapping[str, Any], genes: Mapping[str, Any]) -> List[Dict[str, str]]:
    edge_map: Dict[tuple[str, str, str], Dict[str, str]] = {}
    known_genes = set(genes.keys())

    for raw_edge in payload.get("edges", []):
        if not isinstance(raw_edge, Mapping):
            continue
        source, target, edge_type = _edge_tuple(raw_edge)
        if source in known_genes and target in known_genes and edge_type:
            edge_map[(source, target, edge_type)] = {"source": source, "target": target, "type": edge_type}

    for pair in payload.get("dependencies", []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        source, target = pair
        src = _normalize_gene_id(str(source))
        tgt = _normalize_gene_id(str(target))
        if src in known_genes and tgt in known_genes:
            edge_map[(src, tgt, AST_EDGE_TYPE_DEPENDS_ON)] = {
                "source": src,
                "target": tgt,
                "type": AST_EDGE_TYPE_DEPENDS_ON,
            }

    for pair in payload.get("conflicts", []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left, right = pair
        src = _normalize_gene_id(str(left))
        tgt = _normalize_gene_id(str(right))
        if src in known_genes and tgt in known_genes:
            edge_map[(src, tgt, AST_EDGE_TYPE_CONFLICTS_WITH)] = {
                "source": src,
                "target": tgt,
                "type": AST_EDGE_TYPE_CONFLICTS_WITH,
            }
            edge_map[(tgt, src, AST_EDGE_TYPE_CONFLICTS_WITH)] = {
                "source": tgt,
                "target": src,
                "type": AST_EDGE_TYPE_CONFLICTS_WITH,
            }

    for edge in _default_ast_edges(genes):
        edge_map[(edge["source"], edge["target"], edge["type"])] = edge

    return list(edge_map.values())


def extract_genes_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    schema_version = int(payload.get("schema_version", 0)) if isinstance(payload, Mapping) else 0
    if schema_version >= 3 and isinstance(payload.get("nodes"), list):
        return _genes_from_nodes(payload.get("nodes", []))

    raw_genes = payload.get("genes", payload) if isinstance(payload, Mapping) else {}
    genes: Dict[str, Any] = {}
    if isinstance(raw_genes, Mapping):
        for k, v in raw_genes.items():
            genes[_normalize_gene_id(str(k))] = v
    return genes


def downgrade_genome_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    genes = extract_genes_from_payload(payload)
    return {"schema_version": 2, "genes": genes}


def migrate_genome_payload(
    payload: Mapping[str, Any],
    *,
    tokenizer: str,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
    loss_type: str,
    model_type: str,
    transformer_d_model: int,
    transformer_nhead: int,
    transformer_layers: int,
    transformer_dropout: float,
    learning_rate: float,
    beta_kl: float,
) -> Dict[str, Any]:
    defaults = _default_genes(
        tokenizer=tokenizer,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        loss_type=loss_type,
        model_type=model_type,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        learning_rate=learning_rate,
        beta_kl=beta_kl,
    )

    # Legacy checkpoints stored genes directly in meta without a genome object.
    schema_version = int(payload.get("schema_version", 0)) if isinstance(payload, Mapping) else 0
    genes = extract_genes_from_payload(payload)

    if schema_version < CURRENT_GENOME_SCHEMA_VERSION:
        _transform_old_ranges(genes)

    for gene_id, default_value in defaults.items():
        genes.setdefault(gene_id, default_value)

    genes["tokenizer"] = str(genes["tokenizer"]).lower()
    genes["loss_type"] = str(genes["loss_type"]).lower()
    genes["model_type"] = str(genes["model_type"]).lower()

    constraints = payload.get("constraints") if isinstance(payload, Mapping) else None

    migrated = {
        "schema_version": CURRENT_GENOME_SCHEMA_VERSION,
        "nodes": _nodes_from_genes(genes),
        "edges": _normalize_edges(payload, genes),
    }
    if constraints is not None:
        migrated["constraints"] = constraints
    return migrated
