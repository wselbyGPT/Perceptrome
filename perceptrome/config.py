import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "ncbi": {
        "email": "you@example.com",
        "api_key": None,
        "max_retries": 3,
        "backoff_seconds": 2.0,
    },
    "uniprot": {
        "base_url": "https://rest.uniprot.org",
        "default_query": "reviewed:true AND fragment:false",
        "include_isoforms": False,
        "request_timeout": 60,
        "retries": 3,
        "backoff_seconds": 2.0,
        "records_per_shard": 50000,
        "gzip_output": False,
    },
    "training": {
        # genome defaults
        "steps_per_plasmid": 50,
        "batch_size": 16,
        "window_size": 512,
        "stride": 256,
        "max_stream_epochs": 100,
        "shuffle_catalog": True,
        "hidden_dim": 512,
        "model_type": "mlp",  # "mlp" | "transformer" | "ssm" | "tree" | "hybrid" | "hierarchical" | "conv" | "recurrent" | "wavenet" | "mamba" | "attention_pool" | "bytenet"
        "transformer_d_model": 256,
        "transformer_nhead": 8,
        "transformer_layers": 4,
        "transformer_dropout": 0.1,
        "ast_tree_layers": 4,
        "ast_motif_kernel_size": 7,
        "ast_motif_channels": 64,
        "hierarchical_latent_dim": 256,
        "ast_node_type_vocab_size": 64,
        "hierarchical_ablation_mode": "hierarchical",  # "hierarchical" | "cnn_only" | "ast_only"
        "stage_a_steps": 0,
        "stage_b_steps": 0,
        "stage_c_steps": 0,
        "stage_d_steps": 0,
        "critic_loss_weight": 0.2,
        "learning_rate": 1e-3,
        "beta_kl": 1e-3,
        "kl_warmup_steps": 10000,
        "max_grad_norm": 5.0,
        "tensorboard_log_every": 10,
        "tensorboard_run_id": None,

        # tokenizer
        "tokenizer": "base",   # "base" | "codon" | "aa"
        "frame_offset": 0,     # 0|1|2 (codon mode only)

        # proteome defaults (aa mode)
        "protein_window_aa": 256,
        "protein_stride_aa": 128,
        "min_orf_aa": 90,

        # GenBank CDS-only grounded filters (aa mode + source=genbank)
        "strict_cds": False,              # no ORF fallback; error if nothing passes
        "require_translation": False,      # keep only CDS with /translation
        "x_free": False,                  # reject proteins containing X
        "require_start_m": False,         # require protein begins with M
        "reject_partial_cds": False,       # reject CDS locations with < or >
        "max_protein_aa": None,            # None = no max

        # (optional) denoising / inpainting knobs (safe if unused)
        "aa_mask_prob": 0.10,
        "aa_span_mask_prob": 0.05,
        "aa_span_mask_len": 12,

        # (optional) balance + filters (safe if unused)
        "max_windows_per_protein": 4,
        "protein_len_min": None,
        "protein_len_max": None,
        "translation_only": False,

        # CVAE class conditioning (0 = unconditional, backward compatible)
        "cvae_num_classes": 0,
        "cvae_cond_dim": 64,

        # (optional) curriculum (safe if unused)
        "curriculum_enabled": True,
        "curriculum_steps": [0, 2000, 8000],
        "curriculum_phases": [
            {
                "translation_only": True,
                "protein_len_min": 100,
                "protein_len_max": 400,
                "max_windows_per_protein": 2,
                "aa_mask_prob": 0.08,
                "aa_span_mask_prob": 0.03,
                "aa_span_mask_len": 10,
            },
            {
                "translation_only": True,
                "protein_len_min": 60,
                "protein_len_max": 800,
                "max_windows_per_protein": 4,
                "aa_mask_prob": 0.10,
                "aa_span_mask_prob": 0.05,
                "aa_span_mask_len": 12,
            },
            {
                "translation_only": False,
                "protein_len_min": 60,
                "protein_len_max": None,
                "max_windows_per_protein": 6,
                "aa_mask_prob": 0.12,
                "aa_span_mask_prob": 0.06,
                "aa_span_mask_len": 14,
            },
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


@dataclass
class NCBIConfig:
    email: str
    api_key: Optional[str]
    max_retries: int
    backoff_seconds: float


@dataclass
class UniProtConfig:
    base_url: str
    default_query: str
    include_isoforms: bool
    request_timeout: int
    retries: int
    backoff_seconds: float
    records_per_shard: int
    gzip_output: bool


@dataclass
class TrainingConfig:
    steps_per_plasmid: int
    batch_size: int
    window_size: int
    stride: int
    max_stream_epochs: int
    shuffle_catalog: bool
    hidden_dim: int
    model_type: str
    transformer_d_model: int
    transformer_nhead: int
    transformer_layers: int
    transformer_dropout: float
    ast_tree_layers: int
    ast_motif_kernel_size: int
    ast_motif_channels: int
    hierarchical_latent_dim: int
    ast_node_type_vocab_size: int
    hierarchical_ablation_mode: str
    stage_a_steps: int
    stage_b_steps: int
    stage_c_steps: int
    stage_d_steps: int
    critic_loss_weight: float
    learning_rate: float
    beta_kl: float
    kl_warmup_steps: int
    max_grad_norm: float
    tensorboard_log_every: int
    tensorboard_run_id: Optional[str]

    tokenizer: str
    frame_offset: int

    protein_window_aa: int
    protein_stride_aa: int
    min_orf_aa: int

    # grounded GenBank/CDS flags
    strict_cds: bool
    require_translation: bool
    x_free: bool
    require_start_m: bool
    reject_partial_cds: bool
    max_protein_aa: Optional[int]

    # optional denoising / grammar
    aa_mask_prob: float
    aa_span_mask_prob: float
    aa_span_mask_len: int

    # optional balance / filters
    max_windows_per_protein: int
    protein_len_min: Optional[int]
    protein_len_max: Optional[int]
    translation_only: bool

    # CVAE class conditioning
    cvae_num_classes: int
    cvae_cond_dim: int

    # optional curriculum
    curriculum_enabled: bool
    curriculum_steps: List[Any]
    curriculum_phases: List[Any]

    # ---- compatibility aliases (if older code expects these names) ----
    @property
    def protein_strict_cds_only(self) -> bool:
        return self.strict_cds

    @property
    def protein_require_translation(self) -> bool:
        return self.require_translation

    @property
    def protein_reject_partial(self) -> bool:
        return self.reject_partial_cds

    @property
    def protein_require_start_m(self) -> bool:
        return self.require_start_m

    @property
    def protein_x_free(self) -> bool:
        return self.x_free

    @property
    def protein_min_aa(self) -> int:
        return self.min_orf_aa

    @property
    def protein_max_aa(self) -> Optional[int]:
        return self.max_protein_aa


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
    """Recursively update dicts; lists are replaced (not merged)."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_full_config(path: str) -> Dict[str, Any]:
    # deep copy via JSON to keep it simple + deterministic
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


def _opt_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if v == "":
        return None
    return int(v)


def extract_configs(cfg: Dict[str, Any]) -> Tuple[NCBIConfig, TrainingConfig, IOConfig]:
    n = cfg.get("ncbi", {}) or {}
    t = cfg.get("training", {}) or {}
    io = cfg.get("io", {}) or {}

    ncbi_cfg = NCBIConfig(
        email=str(n.get("email", "you@example.com")),
        api_key=(None if n.get("api_key") in (None, "") else str(n.get("api_key"))),
        max_retries=int(n.get("max_retries", 3)),
        backoff_seconds=float(n.get("backoff_seconds", 2.0)),
    )

    train_cfg = TrainingConfig(
        steps_per_plasmid=int(t.get("steps_per_plasmid", 50)),
        batch_size=int(t.get("batch_size", 16)),
        window_size=int(t.get("window_size", 512)),
        stride=int(t.get("stride", 256)),
        max_stream_epochs=int(t.get("max_stream_epochs", 100)),
        shuffle_catalog=bool(t.get("shuffle_catalog", True)),
        hidden_dim=int(t.get("hidden_dim", 512)),
        model_type=str(t.get("model_type", "mlp")),
        transformer_d_model=int(t.get("transformer_d_model", 256)),
        transformer_nhead=int(t.get("transformer_nhead", 8)),
        transformer_layers=int(t.get("transformer_layers", 4)),
        transformer_dropout=float(t.get("transformer_dropout", 0.1)),
        ast_tree_layers=int(t.get("ast_tree_layers", 4)),
        ast_motif_kernel_size=int(t.get("ast_motif_kernel_size", 7)),
        ast_motif_channels=int(t.get("ast_motif_channels", 64)),
        hierarchical_latent_dim=int(t.get("hierarchical_latent_dim", t.get("hidden_dim", 512))),
        ast_node_type_vocab_size=int(t.get("ast_node_type_vocab_size", 64)),
        hierarchical_ablation_mode=str(t.get("hierarchical_ablation_mode", "hierarchical")),
        stage_a_steps=int(t.get("stage_a_steps", 0)),
        stage_b_steps=int(t.get("stage_b_steps", 0)),
        stage_c_steps=int(t.get("stage_c_steps", 0)),
        stage_d_steps=int(t.get("stage_d_steps", 0)),
        critic_loss_weight=float(t.get("critic_loss_weight", 0.2)),
        learning_rate=float(t.get("learning_rate", 1e-3)),
        beta_kl=float(t.get("beta_kl", 1e-3)),
        kl_warmup_steps=int(t.get("kl_warmup_steps", 10000)),
        max_grad_norm=float(t.get("max_grad_norm", 5.0)),
        tensorboard_log_every=int(t.get("tensorboard_log_every", 10)),
        tensorboard_run_id=(None if t.get("tensorboard_run_id", None) in (None, "") else str(t.get("tensorboard_run_id"))),

        tokenizer=str(t.get("tokenizer", "base")),
        frame_offset=int(t.get("frame_offset", 0)),

        protein_window_aa=int(t.get("protein_window_aa", 256)),
        protein_stride_aa=int(t.get("protein_stride_aa", 128)),
        min_orf_aa=int(t.get("min_orf_aa", 90)),

        strict_cds=bool(t.get("strict_cds", False)),
        require_translation=bool(t.get("require_translation", False)),
        x_free=bool(t.get("x_free", False)),
        require_start_m=bool(t.get("require_start_m", False)),
        reject_partial_cds=bool(t.get("reject_partial_cds", False)),
        max_protein_aa=_opt_int(t.get("max_protein_aa", None)),

        aa_mask_prob=float(t.get("aa_mask_prob", 0.10)),
        aa_span_mask_prob=float(t.get("aa_span_mask_prob", 0.05)),
        aa_span_mask_len=int(t.get("aa_span_mask_len", 12)),

        max_windows_per_protein=int(t.get("max_windows_per_protein", 4)),
        protein_len_min=_opt_int(t.get("protein_len_min", None)),
        protein_len_max=_opt_int(t.get("protein_len_max", None)),
        translation_only=bool(t.get("translation_only", False)),

        cvae_num_classes=int(t.get("cvae_num_classes", 0)),
        cvae_cond_dim=int(t.get("cvae_cond_dim", 64)),

        curriculum_enabled=bool(t.get("curriculum_enabled", True)),
        curriculum_steps=list(t.get("curriculum_steps", [0, 2000, 8000])),
        curriculum_phases=list(t.get("curriculum_phases", [])),
    )

    io_cfg = IOConfig(
        cache_fasta_dir=str(io.get("cache_fasta_dir", "cache/fasta")),
        cache_genbank_dir=str(io.get("cache_genbank_dir", "cache/genbank")),
        cache_encoded_dir=str(io.get("cache_encoded_dir", "cache/encoded")),
        model_dir=str(io.get("model_dir", "model")),
        checkpoints_dir=str(io.get("checkpoints_dir", "model/checkpoints")),
        logs_dir=str(io.get("logs_dir", "logs")),
        state_file=str(io.get("state_file", "state/progress.json")),
    )

    return ncbi_cfg, train_cfg, io_cfg


def extract_uniprot_config(cfg: Dict[str, Any]) -> UniProtConfig:
    u = cfg.get("uniprot", {}) or {}
    return UniProtConfig(
        base_url=str(u.get("base_url", "https://rest.uniprot.org")).rstrip("/"),
        default_query=str(u.get("default_query", "reviewed:true AND fragment:false")),
        include_isoforms=bool(u.get("include_isoforms", False)),
        request_timeout=int(u.get("request_timeout", 60)),
        retries=int(u.get("retries", 3)),
        backoff_seconds=float(u.get("backoff_seconds", 2.0)),
        records_per_shard=int(u.get("records_per_shard", 50000)),
        gzip_output=bool(u.get("gzip_output", False)),
    )
