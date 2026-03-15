from __future__ import annotations

from typing import Any, Dict, Mapping


CURRENT_GENOME_SCHEMA_VERSION = 2


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
    **extra_genes: Any,
) -> Dict[str, Any]:
    genes = {
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
    genes.update(extra_genes)
    return genes


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
    **extra_genes: Any,
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
        **extra_genes,
    )

    # Legacy checkpoints stored genes directly in meta without a genome object.
    schema_version = int(payload.get("schema_version", 0))
    raw_genes = payload.get("genes", payload)

    genes: Dict[str, Any] = {}
    if isinstance(raw_genes, Mapping):
        for k, v in raw_genes.items():
            genes[_normalize_gene_id(str(k))] = v

    if schema_version < CURRENT_GENOME_SCHEMA_VERSION:
        _transform_old_ranges(genes)

    for gene_id, default_value in defaults.items():
        genes.setdefault(gene_id, default_value)

    genes["tokenizer"] = str(genes["tokenizer"]).lower()
    genes["loss_type"] = str(genes["loss_type"]).lower()
    genes["model_type"] = str(genes["model_type"]).lower()

    return {
        "schema_version": CURRENT_GENOME_SCHEMA_VERSION,
        "genes": genes,
    }

