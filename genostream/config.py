import json, os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

DEFAULT_CONFIG: Dict[str, Any] = {
    "ncbi": {"email": "you@example.com", "api_key": None, "max_retries": 3, "backoff_seconds": 2.0},
    "training": {
        # genome defaults
        "steps_per_plasmid": 50,
        "batch_size": 16,
        "window_size": 512,
        "stride": 256,
        "max_stream_epochs": 100,
        "shuffle_catalog": True,
        "hidden_dim": 512,
        "learning_rate": 1e-3,
        "beta_kl": 1e-3,
        "kl_warmup_steps": 10000,
        "max_grad_norm": 5.0,

        # tokenizer
        "tokenizer": "base",      # "base" | "codon" | "aa"
        "frame_offset": 0,        # 0|1|2 (codon mode only)

        # proteome defaults (aa mode)
        "protein_window_aa": 256,
        "protein_stride_aa": 128,
        "min_orf_aa": 90,
        # proteome denoising / grammar
        # (applied only when tokenizer=aa)
        "aa_mask_prob": 0.10,
        "aa_span_mask_prob": 0.05,
        "aa_span_mask_len": 12,

        # proteome sampling balance + filters
        "max_windows_per_protein": 4,
        "protein_len_min": None,
        "protein_len_max": None,
        "translation_only": False,

        # curriculum (length/quality/balance) — aa+genbank only
        "curriculum_enabled": True,
        # total_steps boundaries (from state/progress.json)
        "curriculum_steps": [0, 2000, 8000],
        "curriculum_phases": [
            {"translation_only": True,  "protein_len_min": 100, "protein_len_max": 400, "max_windows_per_protein": 2, "aa_mask_prob": 0.08, "aa_span_mask_prob": 0.03, "aa_span_mask_len": 10},
            {"translation_only": True,  "protein_len_min": 60,  "protein_len_max": 800, "max_windows_per_protein": 4, "aa_mask_prob": 0.10, "aa_span_mask_prob": 0.05, "aa_span_mask_len": 12},
            {"translation_only": False, "protein_len_min": 60,  "protein_len_max": None,"max_windows_per_protein": 6, "aa_mask_prob": 0.12, "aa_span_mask_prob": 0.06, "aa_span_mask_len": 14},
        ],
    },
    "io": {
        "cache_fasta_dir": "cache/fasta",
        "cache_genbank_dir": "cache/genbank",
        "cache_encoded_dir": "cache/encoded",
        "model_dir": "model",
        "checkpoints_dir": "model/checkpoints",
        "logs_dir": "logs",
        "state_file": "state/progress.json",
    },
}

    protein_strict_cds_only: bool
    protein_require_translation: bool
    protein_reject_partial: bool
    protein_require_start_m: bool
    protein_x_free: bool
    protein_min_aa: int
    protein_max_aa: int

@dataclass
class NCBIConfig:
    email: str
    api_key: Optional[str]
    max_retries: int
    backoff_seconds: float

    protein_strict_cds_only: bool
    protein_require_translation: bool
    protein_reject_partial: bool
    protein_require_start_m: bool
    protein_x_free: bool
    protein_min_aa: int
    protein_max_aa: int

@dataclass
class TrainingConfig:
    steps_per_plasmid: int
    batch_size: int
    window_size: int
    stride: int
    max_stream_epochs: int
    shuffle_catalog: bool
    hidden_dim: int
    learning_rate: float
    beta_kl: float
    kl_warmup_steps: int
    max_grad_norm: float

    tokenizer: str
    frame_offset: int

    protein_window_aa: int
    protein_stride_aa: int
    min_orf_aa: int

    # denoising / inpainting (aa only)
    aa_mask_prob: float
    aa_span_mask_prob: float
    aa_span_mask_len: int

    # balance + filters (aa only)
    max_windows_per_protein: int
    protein_len_min: Optional[int]
    protein_len_max: Optional[int]
    translation_only: bool

    # curriculum (aa+genbank only)
    curriculum_enabled: bool
    curriculum_steps: list
    curriculum_phases: list

@dataclass
class IOConfig:
    cache_fasta_dir: str
    cache_genbank_dir: str
    cache_encoded_dir: str
    model_dir: str
    checkpoints_dir: str
    logs_dir: str
    state_file: str

def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_full_config(path: str) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if yaml is not None and path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} did not contain a dict.")
        deep_update(cfg, data)
    elif path and not os.path.exists(path):
        print(f"[config] {path} not found; using built-in defaults.")
    elif path and yaml is None and path:
        print("[config] PyYAML not installed; ignoring config and using defaults.")
    return cfg

def extract_configs(cfg: Dict[str, Any]) -> Tuple[NCBIConfig, TrainingConfig, IOConfig]:
    n = cfg["ncbi"]; t = cfg["training"]; io = cfg["io"]
    return (
        NCBIConfig(
            email=n.get("email", "you@example.com"),
            api_key=n.get("api_key"),
            max_retries=int(n.get("max_retries", 3)),
            backoff_seconds=float(n.get("backoff_seconds", 2.0)),
        ),
        TrainingConfig(
            steps_per_plasmid=int(t.get("steps_per_plasmid", 50)),
            batch_size=int(t.get("batch_size", 16)),
            window_size=int(t.get("window_size", 512)),
            stride=int(t.get("stride", 256)),
            max_stream_epochs=int(t.get("max_stream_epochs", 100)),
            shuffle_catalog=bool(t.get("shuffle_catalog", True)),
            hidden_dim=int(t.get("hidden_dim", 512)),
            learning_rate=float(t.get("learning_rate", 1e-3)),
            beta_kl=float(t.get("beta_kl", 1e-3)),
            kl_warmup_steps=int(t.get("kl_warmup_steps", 10000)),
            max_grad_norm=float(t.get("max_grad_norm", 5.0)),

            tokenizer=str(t.get("tokenizer", "base")),
            frame_offset=int(t.get("frame_offset", 0)),

            protein_window_aa=int(t.get("protein_window_aa", 256)),
            protein_stride_aa=int(t.get("protein_stride_aa", 128)),
            min_orf_aa=int(t.get("min_orf_aa", 90)),
            aa_mask_prob=float(t.get("aa_mask_prob", 0.10)),
            aa_span_mask_prob=float(t.get("aa_span_mask_prob", 0.05)),
            aa_span_mask_len=int(t.get("aa_span_mask_len", 12)),

            max_windows_per_protein=int(t.get("max_windows_per_protein", 4)),
            protein_len_min=(None if t.get("protein_len_min") is None else int(t.get("protein_len_min"))),
            protein_len_max=(None if t.get("protein_len_max") is None else int(t.get("protein_len_max"))),
            translation_only=bool(t.get("translation_only", False)),

            curriculum_enabled=bool(t.get("curriculum_enabled", True)),
            curriculum_steps=list(t.get("curriculum_steps", [0, 2000, 8000])),
            curriculum_phases=list(t.get("curriculum_phases", [])),

        ),
        IOConfig(
            cache_fasta_dir=io.get("cache_fasta_dir", "cache/fasta"),
            cache_genbank_dir=io.get("cache_genbank_dir", "cache/genbank"),
            cache_encoded_dir=io.get("cache_encoded_dir", "cache/encoded"),
            model_dir=io.get("model_dir", "model"),
            checkpoints_dir=io.get("checkpoints_dir", "model/checkpoints"),
            logs_dir=io.get("logs_dir", "logs"),
            state_file=io.get("state_file", "state/progress.json"),
        ),
    )
