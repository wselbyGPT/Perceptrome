"""Shared model catalog for genomic DNA modeling surfaces.

This module is intentionally dependency-free so it can be imported by the CLI,
web server, and config/registry code without pulling in PyTorch.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


DNA_GENERATIVE_MODEL_TYPES: Tuple[str, ...] = (
    "mlp",
    "transformer",
    "ssm",
    "conv",
    "recurrent",
    "wavenet",
    "mamba",
    "attention_pool",
    "bytenet",
    "tree",
    "hybrid",
    "hierarchical",
)

MODEL_TYPE_ALIASES: Dict[str, str] = {
    "rnn": "recurrent",
    "gru": "recurrent",
    "lstm": "recurrent",
    "bigru": "recurrent",
    "bilstm": "recurrent",
    "cnn": "conv",
    "resnet": "conv",
    "residual_conv": "conv",
    "moe": "mlp",
    "gnn": "mlp",
    "tcn": "wavenet",
    "dilated": "wavenet",
    "causal_conv": "wavenet",
    "ssm_mamba": "mamba",
    "selective_ssm": "mamba",
    "s4": "mamba",
    "attn_pool": "attention_pool",
    "perceiver": "attention_pool",
    "cross_attn": "attention_pool",
    "hier": "hierarchical",
}

DNA_MODEL_TYPE_CHOICES: Tuple[str, ...] = tuple(
    dict.fromkeys((*DNA_GENERATIVE_MODEL_TYPES, *MODEL_TYPE_ALIASES.keys()))
)

TRAINING_MODEL_OVERRIDE_KEYS: Tuple[str, ...] = (
    "hidden_dim",
    "transformer_d_model",
    "transformer_nhead",
    "transformer_layers",
    "transformer_dropout",
    "ast_tree_layers",
    "ast_motif_kernel_size",
    "ast_motif_channels",
    "hierarchical_latent_dim",
    "ast_node_type_vocab_size",
    "hierarchical_ablation_mode",
    "learning_rate",
    "beta_kl",
    "kl_warmup_steps",
    "max_grad_norm",
)


def normalize_model_type(model_type: str) -> str:
    mt = str(model_type or "mlp").lower()
    return MODEL_TYPE_ALIASES.get(mt, mt)


def apply_training_model_overrides(cfg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Fold web/API model params into a loaded config dictionary."""
    training = cfg.get("training")
    if not isinstance(training, dict):
        training = {}
        cfg["training"] = training

    model_type = params.get("model_type")
    if model_type is None:
        legacy_family = str(params.get("model_family") or "").strip().lower()
        if legacy_family in {"baseline", "vae"}:
            model_type = "mlp"
        elif legacy_family:
            model_type = legacy_family
    if model_type is not None:
        normalized = normalize_model_type(str(model_type))
        if normalized not in DNA_GENERATIVE_MODEL_TYPES:
            raise ValueError(f"Unsupported genomic DNA model_type: {model_type}")
        training["model_type"] = normalized

    for key in TRAINING_MODEL_OVERRIDE_KEYS:
        if key in params and params[key] is not None:
            training[key] = params[key]
    return cfg
