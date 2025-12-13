#!/usr/bin/env bash
set -euo pipefail

# --- genostream/config.py ---
cat > genostream/config.py <<'PY'
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
        # NEW:
        "tokenizer": "base",      # "base" | "codon"
        "frame_offset": 0,        # 0|1|2 (used for codon tokenization)
    },
    "io": {
        "cache_fasta_dir": "cache/fasta",
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

@dataclass
class IOConfig:
    cache_fasta_dir: str
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
        ),
        IOConfig(
            cache_fasta_dir=io.get("cache_fasta_dir", "cache/fasta"),
            cache_encoded_dir=io.get("cache_encoded_dir", "cache/encoded"),
            model_dir=io.get("model_dir", "model"),
            checkpoints_dir=io.get("checkpoints_dir", "model/checkpoints"),
            logs_dir=io.get("logs_dir", "logs"),
            state_file=io.get("state_file", "state/progress.json"),
        ),
    )
PY

# --- genostream/io_utils.py (add encoded_cache_path) ---
python3 - <<'PY'
import pathlib, re

p = pathlib.Path("genostream/io_utils.py")
txt = p.read_text(encoding="utf-8")

if "def encoded_cache_path" in txt:
    print("[upgrade] io_utils.py already has encoded_cache_path")
    raise SystemExit(0)

# Append helper at end
txt = txt.rstrip() + """

def encoded_cache_path(io_cfg: IOConfig, accession: str, tokenizer: str, window_size_bp: int, stride_bp: int, frame_offset: int) -> str:
    \"""
    Encoded cache file path that avoids mixing tokenizers / window params.

    Examples:
      ABC.base.w512.s256.npy
      ABC.codon.w510.s255.f0.npy
    \"""
    tok = tokenizer.lower()
    tag = f"{tok}.w{window_size_bp}.s{stride_bp}"
    if tok == "codon":
        tag += f".f{int(frame_offset)}"
    fname = f"{accession}.{tag}.npy"
    import os
    return os.path.join(io_cfg.cache_encoded_dir, fname)
"""
p.write_text(txt, encoding="utf-8")
print("[upgrade] patched genostream/io_utils.py")
PY

# --- genostream/encoding.py (codon tokenizer + gc calc + meta helper) ---
cat > genostream/encoding.py <<'PY'
import logging, os
from typing import List, Optional, Tuple

import numpy as np
from .config import IOConfig

# -----------------------------
# Codon tokenizer constants
# -----------------------------
_CODON_BASES = "ACGT"
CODONS: List[str] = [a + b + c for a in _CODON_BASES for b in _CODON_BASES for c in _CODON_BASES]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS)}
UNK_IDX = len(CODONS)  # 64
IDX_TO_CODON: List[str] = CODONS + ["NNN"]
CODON_VOCAB_SIZE = len(IDX_TO_CODON)  # 65

# GC info per token
_gc_frac = []
_gc_count = []
for c in CODONS:
    gc = sum(1 for ch in c if ch in ("G", "C"))
    _gc_count.append(float(gc))
    _gc_frac.append(float(gc) / 3.0)
# UNK token defaults
_gc_count.append(0.0)
_gc_frac.append(0.0)
GC_COUNT_PER_TOKEN = np.array(_gc_count, dtype=np.float32)     # (65,)
GC_FRAC_PER_TOKEN = np.array(_gc_frac, dtype=np.float32)       # (65,)

def tokenizer_meta(tokenizer: str, window_size_bp: int) -> Tuple[int, int]:
    """
    Returns (seq_len_units, vocab_size).

    base: units = window_size_bp, vocab = 4
    codon: units = window_size_bp//3, vocab = 65 (64 codons + UNK)
    """
    tok = tokenizer.lower()
    if tok == "base":
        return window_size_bp, 4
    if tok == "codon":
        if window_size_bp % 3 != 0:
            raise ValueError(f"codon tokenizer requires window_size divisible by 3 (got {window_size_bp})")
        return window_size_bp // 3, CODON_VOCAB_SIZE
    raise ValueError(f"Unknown tokenizer: {tokenizer}")

def parse_fasta_sequence(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA not found: {path}")
    seq_lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    seq = "".join(seq_lines).upper()
    if not seq:
        raise ValueError(f"No sequence found in {path}")
    return seq

def encode_sequence_one_hot(seq: str, window_size: int, stride: int) -> np.ndarray:
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    length = len(seq)

    if length < window_size:
        arr = np.zeros((window_size, 4), dtype=np.float32)
        for i, base in enumerate(seq):
            idx = base_to_idx.get(base)
            if idx is not None:
                arr[i, idx] = 1.0
        return arr[None, ...]

    windows: List[np.ndarray] = []
    for start in range(0, length - window_size + 1, stride):
        window_seq = seq[start : start + window_size]
        arr = np.zeros((window_size, 4), dtype=np.float32)
        for i, base in enumerate(window_seq):
            idx = base_to_idx.get(base)
            if idx is not None:
                arr[i, idx] = 1.0
        windows.append(arr)

    if not windows:
        arr = np.zeros((window_size, 4), dtype=np.float32)
        for i, base in enumerate(seq[:window_size]):
            idx = base_to_idx.get(base)
            if idx is not None:
                arr[i, idx] = 1.0
        windows.append(arr)

    return np.stack(windows, axis=0)

def encode_sequence_codons(seq: str, window_size_bp: int, stride_bp: int, frame_offset: int = 0) -> np.ndarray:
    """
    Encode sequence into codon one-hot windows.

    Output shape: (num_windows, window_codons, 65)
    """
    if frame_offset not in (0, 1, 2):
        raise ValueError("frame_offset must be 0, 1, or 2")
    if window_size_bp % 3 != 0:
        raise ValueError(f"codon tokenizer requires window_size divisible by 3 (got {window_size_bp})")
    if stride_bp % 3 != 0:
        raise ValueError(f"codon tokenizer requires stride divisible by 3 (got {stride_bp})")

    # Trim usable length to full codons from frame_offset
    seq = seq.upper()
    if len(seq) <= frame_offset + 2:
        # Too short; return a single empty-ish window
        window_codons = window_size_bp // 3
        out = np.zeros((1, window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        return out

    usable = len(seq) - ((len(seq) - frame_offset) % 3)
    codon_count = max(0, (usable - frame_offset) // 3)
    window_codons = window_size_bp // 3
    stride_codons = stride_bp // 3

    def codon_at(i: int) -> int:
        # i is codon index (0..codon_count-1)
        start = frame_offset + 3 * i
        c = seq[start : start + 3]
        return CODON_TO_IDX.get(c, UNK_IDX)

    if codon_count <= 0:
        out = np.zeros((1, window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        return out

    windows: List[np.ndarray] = []

    if codon_count < window_codons:
        arr = np.zeros((window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        for j in range(codon_count):
            idx = codon_at(j)
            arr[j, idx] = 1.0
        windows.append(arr)
        return np.stack(windows, axis=0)

    for start_c in range(0, codon_count - window_codons + 1, stride_codons):
        arr = np.zeros((window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        for j in range(window_codons):
            idx = codon_at(start_c + j)
            arr[j, idx] = 1.0
        windows.append(arr)

    if not windows:
        arr = np.zeros((window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        for j in range(window_codons):
            idx = codon_at(j)
            arr[j, idx] = 1.0
        windows.append(arr)

    return np.stack(windows, axis=0)

def encode_accession(
    accession: str,
    io_cfg: IOConfig,
    window_size: int,
    stride: int,
    tokenizer: str = "base",
    frame_offset: int = 0,
    save_to_disk: bool = True,
    out_path: Optional[str] = None,
) -> np.ndarray:
    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA for {accession} not found at {fasta_path}")

    seq = parse_fasta_sequence(fasta_path)

    tok = tokenizer.lower()
    if tok == "base":
        encoded = encode_sequence_one_hot(seq, window_size, stride)
        logging.info(
            f"{accession}: encoded(BASE) len={len(seq)} -> windows={encoded.shape[0]} "
            f"window_size={window_size} stride={stride} shape={encoded.shape}"
        )
    elif tok == "codon":
        encoded = encode_sequence_codons(seq, window_size_bp=window_size, stride_bp=stride, frame_offset=frame_offset)
        logging.info(
            f"{accession}: encoded(CODON) len={len(seq)} -> windows={encoded.shape[0]} "
            f"window_bp={window_size} stride_bp={stride} frame={frame_offset} shape={encoded.shape}"
        )
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    if save_to_disk:
        os.makedirs(io_cfg.cache_encoded_dir, exist_ok=True)
        if out_path is None:
            out_path = os.path.join(io_cfg.cache_encoded_dir, f"{accession}.npy")
        np.save(out_path, encoded)
        logging.info(f"{accession}: saved encoded tensor to {out_path}")

    return encoded

def compute_gc_from_encoded(encoded: np.ndarray, tokenizer: str = "base") -> np.ndarray:
    """
    Returns GC fraction per window.

    base encoded: (N, Wbp, 4)
    codon encoded: (N, Wcodons, 65)
    """
    tok = tokenizer.lower()
    if tok == "base":
        if encoded.ndim != 3 or encoded.shape[2] != 4:
            raise ValueError("base encoded must have shape (num_windows, window_size, 4)")
        gc_counts = encoded[:, :, 1] + encoded[:, :, 2]  # (N, W)
        window_size = encoded.shape[1]
        return (gc_counts.sum(axis=1) / float(window_size)).astype(np.float32)

    if tok == "codon":
        if encoded.ndim != 3 or encoded.shape[2] != CODON_VOCAB_SIZE:
            raise ValueError("codon encoded must have shape (num_windows, window_codons, 65)")
        # expected GC fraction per codon position, then mean across codons
        # (N, W, V) * (V,) -> (N, W)
        gc_pos = (encoded * GC_FRAC_PER_TOKEN[None, None, :]).sum(axis=2)
        return gc_pos.mean(axis=1).astype(np.float32)

    raise ValueError(f"Unknown tokenizer: {tokenizer}")
PY

# --- genostream/model.py (generalized input_dim + checkpoint meta includes tokenizer/vocab/seq_len) ---
cat > genostream/model.py <<'PY'
import logging, os
from typing import Dict, Tuple

try:
    import torch
    from torch import nn, optim
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    optim = None  # type: ignore

from .config import IOConfig

class PlasmidVAE(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for PlasmidVAE.")
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        h = self.act(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h = self.act(self.fc2(z))
        return self.out_act(self.fc_out(h))

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def get_device() -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_or_init_model(
    io_cfg: IOConfig,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
    learning_rate: float,
    device: "torch.device",
    tokenizer: str,
) -> Tuple[PlasmidVAE, "optim.Optimizer", int, str]:
    """
    seq_len: number of positions (bp or codons)
    vocab_size: 4 for base, 65 for codon
    """
    if torch is None or nn is None or optim is None:
        raise RuntimeError("PyTorch is required.")

    input_dim = int(seq_len) * int(vocab_size)
    ckpt_path = os.path.join(io_cfg.checkpoints_dir, "latest.pt")

    model = PlasmidVAE(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    global_step = 0

    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location=device)
        meta: Dict[str, object] = data.get("meta", {})
        ck_tok = str(meta.get("tokenizer", "base")).lower()
        ck_seq = int(meta.get("seq_len", seq_len))
        ck_vocab = int(meta.get("vocab_size", vocab_size))
        ck_hidden = int(meta.get("hidden_dim", hidden_dim))

        if ck_tok != tokenizer.lower():
            raise ValueError(f"Checkpoint tokenizer={ck_tok} but requested tokenizer={tokenizer}. Delete {ckpt_path} or match settings.")
        if ck_seq != seq_len:
            raise ValueError(f"Checkpoint seq_len={ck_seq} but requested seq_len={seq_len}. Delete {ckpt_path} or match settings.")
        if ck_vocab != vocab_size:
            raise ValueError(f"Checkpoint vocab_size={ck_vocab} but requested vocab_size={vocab_size}. Delete {ckpt_path} or match settings.")
        if ck_hidden != hidden_dim:
            raise ValueError(f"Checkpoint hidden_dim={ck_hidden} but requested hidden_dim={hidden_dim}. Delete {ckpt_path} or match settings.")

        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optim"])
        global_step = int(meta.get("global_step", 0))

        logging.info(f"Loaded checkpoint {ckpt_path} (tokenizer={ck_tok}, seq_len={ck_seq}, vocab={ck_vocab}, hidden={ck_hidden}, step={global_step})")
    else:
        logging.info(f"Initializing new VAE (tokenizer={tokenizer}, seq_len={seq_len}, vocab={vocab_size}, input_dim={input_dim}, hidden={hidden_dim}, lr={learning_rate})")

    return model, optimizer, global_step, ckpt_path

def save_checkpoint(
    ckpt_path: str,
    model: PlasmidVAE,
    optimizer: "optim.Optimizer",
    global_step: int,
    tokenizer: str,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
) -> None:
    if torch is None:
        return
    payload = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "meta": {
            "global_step": int(global_step),
            "tokenizer": str(tokenizer).lower(),
            "seq_len": int(seq_len),
            "vocab_size": int(vocab_size),
            "hidden_dim": int(hidden_dim),
        },
    }
    tmp = ckpt_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, ckpt_path)
    logging.info(f"Saved checkpoint step={global_step} -> {ckpt_path}")

def vae_loss(
    recon: "torch.Tensor",
    x: "torch.Tensor",
    mu: "torch.Tensor",
    logvar: "torch.Tensor",
    beta_kl: float,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch required.")
    mse = nn.MSELoss(reduction="mean")(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = mse + float(beta_kl) * kl
    return total, mse, kl
PY

# --- genostream/training.py (pass tokenizer/seq_len/vocab into model loader + checkpoint) ---
cat > genostream/training.py <<'PY'
import logging, os
from typing import Dict, Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

from .config import IOConfig, TrainingConfig
from .model import get_device, load_or_init_model, save_checkpoint, vae_loss
from .encoding import tokenizer_meta

def train_on_encoded(
    accession: str,
    encoded: np.ndarray,
    steps: int,
    batch_size: int,
    state: Dict[str, Any],
    io_cfg: IOConfig,
    train_cfg: TrainingConfig,
    tokenizer: str,
    window_size_bp: int,
) -> float:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    device = get_device()

    seq_len, vocab_size = tokenizer_meta(tokenizer, window_size_bp)
    hidden_dim = train_cfg.hidden_dim

    model, optimizer, global_step, ckpt_path = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tokenizer,
    )

    windows_tensor = torch.from_numpy(encoded)  # (N, L, V)
    dataset = TensorDataset(windows_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if len(dataloader) == 0:
        logging.warning(f"{accession}: no windows to train on (shape={encoded.shape})")
        return 0.0

    logging.info(
        f"{accession}: train tokenizer={tokenizer} windows={encoded.shape[0]} "
        f"L={encoded.shape[1]} V={encoded.shape[2]} steps={steps} batch={batch_size}"
    )

    step_count = 0
    last_total = 0.0
    it = iter(dataloader)

    while step_count < steps:
        try:
            (batch,) = next(it)
        except StopIteration:
            it = iter(dataloader)
            (batch,) = next(it)

        batch = batch.to(device)         # (B, L, V)
        x_flat = batch.view(batch.size(0), -1)

        if train_cfg.kl_warmup_steps > 0:
            warm = min(1.0, (global_step + 1) / float(train_cfg.kl_warmup_steps))
            beta = train_cfg.beta_kl * warm
        else:
            beta = train_cfg.beta_kl

        optimizer.zero_grad(set_to_none=True)
        recon_flat, mu, logvar = model(x_flat)
        total, recon, kl = vae_loss(recon_flat, x_flat, mu, logvar, beta)
        total.backward()

        if train_cfg.max_grad_norm and train_cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

        optimizer.step()

        step_count += 1
        global_step += 1
        last_total = float(total.item())

        if step_count % 10 == 0 or step_count == steps:
            logging.info(
                f"{accession}: step {step_count}/{steps} total={total.item():.6f} recon={recon.item():.6f} kl={kl.item():.6f} beta={beta:.3g}"
            )

    state["total_steps"] = int(state.get("total_steps", 0)) + step_count

    save_checkpoint(
        ckpt_path=ckpt_path,
        model=model,
        optimizer=optimizer,
        global_step=global_step,
        tokenizer=tokenizer,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    )

    return last_total

def cleanup_accession_files(accession: str, io_cfg: IOConfig, encoded_path: str) -> None:
    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
    for path in (fasta_path, encoded_path):
        if os.path.exists(path):
            try:
                os.remove(path)
                logging.info(f"{accession}: deleted {path}")
            except OSError as e:
                logging.warning(f"{accession}: failed to delete {path}: {e}")

def compute_window_errors(
    accession: str,
    encoded: np.ndarray,
    io_cfg: IOConfig,
    train_cfg: TrainingConfig,
    tokenizer: str,
    window_size_bp: int,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    device = get_device()

    seq_len, vocab_size = tokenizer_meta(tokenizer, window_size_bp)
    hidden_dim = train_cfg.hidden_dim

    model, optimizer, global_step, ckpt_path = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tokenizer,
    )

    model.eval()
    with torch.no_grad():
        wt = torch.from_numpy(encoded).to(device)
        if wt.numel() == 0:
            return np.zeros((0,), dtype=np.float32)
        N = wt.size(0)
        x_flat = wt.view(N, -1)
        mu, logvar = model.encode(x_flat)
        recon_flat = model.decode(mu)
        recon = recon_flat.view_as(wt)
        mse = (recon - wt).pow(2).mean(dim=(1, 2))
        return mse.cpu().numpy().astype(np.float32)
PY

# --- genostream/generate.py (codon-aware generation) ---
cat > genostream/generate.py <<'PY'
import logging, os, random
from typing import List, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .config import TrainingConfig, IOConfig
from .model import get_device, load_or_init_model
from .encoding import tokenizer_meta, IDX_TO_CODON, CODON_VOCAB_SIZE, GC_COUNT_PER_TOKEN

def generate_plasmid_sequence(
    train_cfg: TrainingConfig,
    io_cfg: IOConfig,
    length_bp: int,
    num_windows: Optional[int],
    window_size_bp: int,
    seed: Optional[int],
    latent_scale: float,
    temperature: float,
    gc_bias: float,
    name: str,
    output_path: str,
    tokenizer: str,
) -> str:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")

    tok = tokenizer.lower()
    if tok == "codon":
        # Keep output length a multiple of 3
        if length_bp % 3 != 0:
            length_bp = (length_bp // 3) * 3

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = get_device()
    seq_len, vocab_size = tokenizer_meta(tok, window_size_bp)
    hidden_dim = train_cfg.hidden_dim

    model, optimizer, global_step, ckpt_path = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tok,
    )
    model.eval()

    if num_windows is not None:
        n_windows = int(num_windows)
        target_bp = n_windows * window_size_bp
    else:
        n_windows = (length_bp + window_size_bp - 1) // window_size_bp
        target_bp = length_bp

    temperature = max(1e-3, float(temperature))
    latent_scale = float(latent_scale)
    gc_bias = float(gc_bias)

    logging.info(
        f"[generate] tokenizer={tok} windows={n_windows} window_bp={window_size_bp} "
        f"seq_len={seq_len} vocab={vocab_size} hidden={hidden_dim} temp={temperature} "
        f"latent_scale={latent_scale} gc_bias={gc_bias}"
    )

    seq_chars: List[str] = []

    def sample_index(weights: np.ndarray) -> int:
        w = weights.astype(np.float64)
        w = np.clip(w, 1e-9, None)

        # temperature: sharpen/soften
        if temperature != 1.0:
            w = w ** (1.0 / temperature)

        # gc bias:
        if gc_bias != 1.0:
            if tok == "base":
                # indices: A,C,G,T -> boost C/G
                w[1] *= gc_bias
                w[2] *= gc_bias
            else:
                # codon: boost by GC count (0..3)
                w *= (gc_bias ** GC_COUNT_PER_TOKEN[: w.shape[0]])

        s = w.sum()
        if s <= 0:
            return int(np.random.randint(0, w.shape[0]))
        w /= s
        return int(np.random.choice(w.shape[0], p=w))

    with torch.no_grad():
        for _ in range(n_windows):
            z = torch.randn(1, hidden_dim, device=device) * latent_scale
            recon_flat = model.decode(z)   # (1, seq_len*vocab)
            recon = recon_flat.view(seq_len, vocab_size).cpu().numpy()

            if tok == "base":
                idx_to_base = ["A", "C", "G", "T"]
                for j in range(seq_len):
                    idx = sample_index(recon[j])
                    seq_chars.append(idx_to_base[idx])
            else:
                for j in range(seq_len):
                    idx = sample_index(recon[j])
                    seq_chars.append(IDX_TO_CODON[idx])

    seq = "".join(seq_chars) if tok == "base" else "".join(seq_chars)  # codons already strings
    if tok == "codon":
        # seq is codon concatenation
        if num_windows is None and target_bp is not None:
            seq = seq[:target_bp]
    else:
        if num_windows is None and target_bp is not None:
            seq = seq[:target_bp]

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

    return seq
PY

# --- genostream/cli.py (add --tokenizer/--frame-offset + use encoded_cache_path + pass tokenizer meta to training/generation) ---
cat > genostream/cli.py <<'PY'
#!/usr/bin/env python3
import argparse, logging, os
from typing import Any

import numpy as np

from .config import extract_configs, load_full_config
from .encoding import compute_gc_from_encoded, encode_accession
from .generate import generate_plasmid_sequence
from .io_utils import ensure_dirs, load_state, read_catalog, save_state, setup_logging, encoded_cache_path
from .ncbi_fetch import fetch_fasta
from .training import cleanup_accession_files, compute_window_errors, train_on_encoded

try:
    import curses
except ImportError:
    curses = None  # type: ignore

from .scope import run_scope_ui, run_scope_stream_ui, ScopeStreamContext

def _get_tok(args, train_cfg):
    return (getattr(args, "tokenizer", None) or train_cfg.tokenizer).lower()

def _get_frame(args, train_cfg):
    return int(getattr(args, "frame_offset", None) if getattr(args, "frame_offset", None) is not None else train_cfg.frame_offset)

def _validate_tok_params(tokenizer: str, window_size: int, stride: int, frame: int):
    if tokenizer == "codon":
        if window_size % 3 != 0:
            raise ValueError(f"--tokenizer codon requires --window-size divisible by 3 (got {window_size})")
        if stride % 3 != 0:
            raise ValueError(f"--tokenizer codon requires --stride divisible by 3 (got {stride})")
        if frame not in (0,1,2):
            raise ValueError("--frame-offset must be 0,1,2")

def cmd_init(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, _, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    state = {"current_index": 0, "total_steps": 0, "plasmid_visit_counts": {}, "epoch": 0, "last_checkpoint": None}
    save_state(io_cfg.state_file, state)
    print(f"Initialized project. State file at: {io_cfg.state_file}")
    return 0

def cmd_catalog_show(args: argparse.Namespace) -> int:
    accessions = read_catalog(args.path)
    print(f"Catalog: {args.path}\n  {len(accessions)} accessions")
    for acc in accessions[:10]:
        print(f"    {acc}")
    if len(accessions) > 10:
        print(f"    ... (+{len(accessions)-10} more)")
    return 0

def cmd_fetch_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg); setup_logging(io_cfg.logs_dir)
    fetch_fasta(args.accession, io_cfg, ncbi_cfg, force=args.force)
    return 0

def cmd_encode_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg); setup_logging(io_cfg.logs_dir)

    window_size = args.window_size or train_cfg.window_size
    stride = args.stride or train_cfg.stride
    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    _validate_tok_params(tok, window_size, stride, frame)

    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{args.accession}.fasta")
    if not os.path.exists(fasta_path):
        print(f"FASTA for {args.accession} not found at {fasta_path}. Run fetch-one first.")
        return 1

    out_path = encoded_cache_path(io_cfg, args.accession, tok, window_size, stride, frame)
    encoded = encode_accession(args.accession, io_cfg, window_size, stride, tokenizer=tok, frame_offset=frame, save_to_disk=True, out_path=out_path)
    print(f"{args.accession}: encoded tokenizer={tok} -> shape={encoded.shape} saved={out_path}")
    return 0

def cmd_train_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg); setup_logging(io_cfg.logs_dir)
    state = load_state(io_cfg.state_file)

    window_size = args.window_size or train_cfg.window_size
    stride = args.stride or train_cfg.stride
    batch_size = args.batch_size or train_cfg.batch_size
    steps = args.steps or train_cfg.steps_per_plasmid
    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    _validate_tok_params(tok, window_size, stride, frame)

    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{args.accession}.fasta")
    if not os.path.exists(fasta_path):
        logging.info(f"{args.accession}: FASTA not found; fetching.")
        fetch_fasta(args.accession, io_cfg, ncbi_cfg, force=False)

    enc_path = encoded_cache_path(io_cfg, args.accession, tok, window_size, stride, frame)
    if os.path.exists(enc_path) and not args.reencode:
        encoded = np.load(enc_path)
        logging.info(f"{args.accession}: using cached encoded at {enc_path} shape={encoded.shape}")
    else:
        encoded = encode_accession(args.accession, io_cfg, window_size, stride, tokenizer=tok, frame_offset=frame, save_to_disk=True, out_path=enc_path)

    last_total = train_on_encoded(
        args.accession, encoded,
        steps=steps, batch_size=batch_size,
        state=state, io_cfg=io_cfg, train_cfg=train_cfg,
        tokenizer=tok, window_size_bp=window_size,
    )

    pvc = state["plasmid_visit_counts"]
    pvc[args.accession] = pvc.get(args.accession, 0) + 1
    save_state(io_cfg.state_file, state)

    print(f"{args.accession}: train-one tokenizer={tok} steps={steps} batch={batch_size} last_total={last_total:.6f}")
    return 0

def cmd_scope_one(args: argparse.Namespace) -> int:
    if curses is None:
        raise RuntimeError("curses not available")
    cfg = load_full_config(args.config)
    _, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg); setup_logging(io_cfg.logs_dir)

    window_size = args.window_size or train_cfg.window_size
    stride = args.stride or train_cfg.stride
    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    _validate_tok_params(tok, window_size, stride, frame)

    enc_path = encoded_cache_path(io_cfg, args.accession, tok, window_size, stride, frame)
    if os.path.exists(enc_path) and not args.reencode:
        encoded = np.load(enc_path)
    else:
        encoded = encode_accession(args.accession, io_cfg, window_size, stride, tokenizer=tok, frame_offset=frame, save_to_disk=True, out_path=enc_path)

    errors = compute_window_errors(args.accession, encoded, io_cfg=io_cfg, train_cfg=train_cfg, tokenizer=tok, window_size_bp=window_size)
    gc_values = compute_gc_from_encoded(encoded, tokenizer=tok)

    curses.wrapper(
        run_scope_ui,
        accession=args.accession,
        errors=errors,
        gc_values=gc_values,
        window_size=window_size,
        stride=stride,
        fps=args.fps,
    )
    return 0

def cmd_scope_stream(args: argparse.Namespace) -> int:
    if curses is None:
        raise RuntimeError("curses not available")
    cfg = load_full_config(args.config)
    _, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg); setup_logging(io_cfg.logs_dir)

    window_size = args.window_size or train_cfg.window_size
    stride = args.stride or train_cfg.stride
    steps = args.steps or train_cfg.steps_per_plasmid
    batch_size = args.batch_size or train_cfg.batch_size
    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    _validate_tok_params(tok, window_size, stride, frame)

    enc_path = encoded_cache_path(io_cfg, args.accession, tok, window_size, stride, frame)
    if os.path.exists(enc_path) and not args.reencode:
        encoded = np.load(enc_path)
    else:
        encoded = encode_accession(args.accession, io_cfg, window_size, stride, tokenizer=tok, frame_offset=frame, save_to_disk=True, out_path=enc_path)

    gc_values = compute_gc_from_encoded(encoded, tokenizer=tok)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .model import get_device, load_or_init_model
    from .encoding import tokenizer_meta

    device = get_device()
    seq_len, vocab_size = tokenizer_meta(tok, window_size)
    hidden_dim = train_cfg.hidden_dim

    model, optimizer, global_step, ckpt_path = load_or_init_model(
        io_cfg=io_cfg, seq_len=seq_len, vocab_size=vocab_size,
        hidden_dim=hidden_dim, learning_rate=train_cfg.learning_rate,
        device=device, tokenizer=tok
    )

    windows_tensor = torch.from_numpy(encoded)
    dataset = TensorDataset(windows_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    ctx = ScopeStreamContext(
        model=model, optimizer=optimizer, device=device,
        dataloader=dataloader, dataloader_iter=iter(dataloader),
        global_step=global_step, last_total=0.0,
        steps_target=steps, steps_done=0,
        beta_kl=train_cfg.beta_kl, kl_warmup_steps=train_cfg.kl_warmup_steps,
        max_grad_norm=train_cfg.max_grad_norm,
    )

    curses.wrapper(
        run_scope_stream_ui,
        accession=args.accession,
        windows_tensor=windows_tensor,
        gc_values=gc_values,
        window_size=window_size,
        stride=stride,
        fps=args.fps,
        update_every=args.update_every,
        ctx=ctx,
    )
    return 0

def cmd_stream(args: argparse.Namespace) -> int:
    import random
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg); setup_logging(io_cfg.logs_dir)

    accessions = read_catalog(args.catalog)
    state = load_state(io_cfg.state_file)

    window_size = args.window_size or train_cfg.window_size
    stride = args.stride or train_cfg.stride
    batch_size = args.batch_size or train_cfg.batch_size
    steps_per_plasmid = args.steps_per_plasmid or train_cfg.steps_per_plasmid
    max_epochs = args.max_epochs or train_cfg.max_stream_epochs
    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    _validate_tok_params(tok, window_size, stride, frame)

    epoch = int(state.get("epoch", 0))

    while epoch < max_epochs:
        indices = list(range(len(accessions)))
        if train_cfg.shuffle_catalog:
            random.shuffle(indices)

        for i, idx in enumerate(indices):
            acc = accessions[idx]

            fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{acc}.fasta")
            if not os.path.exists(fasta_path):
                fetch_fasta(acc, io_cfg, ncbi_cfg, force=False)

            enc_path = encoded_cache_path(io_cfg, acc, tok, window_size, stride, frame)
            if os.path.exists(enc_path):
                encoded = np.load(enc_path)
            else:
                encoded = encode_accession(acc, io_cfg, window_size, stride, tokenizer=tok, frame_offset=frame, save_to_disk=True, out_path=enc_path)

            _ = train_on_encoded(
                acc, encoded,
                steps=steps_per_plasmid, batch_size=batch_size,
                state=state, io_cfg=io_cfg, train_cfg=train_cfg,
                tokenizer=tok, window_size_bp=window_size,
            )

            pvc = state["plasmid_visit_counts"]
            pvc[acc] = pvc.get(acc, 0) + 1
            state["current_index"] = idx
            state["epoch"] = epoch
            save_state(io_cfg.state_file, state)

            if args.delete_cache:
                cleanup_accession_files(acc, io_cfg, enc_path)

        epoch += 1

    print("[stream] Training complete.")
    return 0

def cmd_generate_plasmid(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg); setup_logging(io_cfg.logs_dir)

    tok = _get_tok(args, train_cfg)
    window_size = args.window_size or train_cfg.window_size
    stride = train_cfg.stride
    frame = _get_frame(args, train_cfg)
    _validate_tok_params(tok, window_size, stride, frame)

    seq = generate_plasmid_sequence(
        train_cfg=train_cfg,
        io_cfg=io_cfg,
        length_bp=args.length_bp,
        num_windows=args.num_windows,
        window_size_bp=window_size,
        seed=args.seed,
        latent_scale=args.latent_scale,
        temperature=args.temperature,
        gc_bias=args.gc_bias,
        name=args.name,
        output_path=args.output,
        tokenizer=tok,
    )
    print(f"[generate-plasmid] tokenizer={tok} wrote {len(seq)} bp -> {args.output}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stream_train.py", description="Genostream streaming VAE trainer + scope.")
    p.add_argument("--config", default="stream_config.yaml", help="YAML config path (default: stream_config.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init"); s.set_defaults(func=cmd_init)

    s = sub.add_parser("catalog-show"); s.add_argument("path"); s.set_defaults(func=cmd_catalog_show)

    s = sub.add_parser("fetch-one")
    s.add_argument("accession"); s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_fetch_one)

    def add_tok_args(sp):
        sp.add_argument("--tokenizer", choices=["base","codon"], default=None, help="Override tokenizer (default from config)")
        sp.add_argument("--frame-offset", type=int, choices=[0,1,2], default=None, help="Codon frame offset (default from config)")

    s = sub.add_parser("encode-one")
    s.add_argument("accession")
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    add_tok_args(s)
    s.set_defaults(func=cmd_encode_one)

    s = sub.add_parser("train-one")
    s.add_argument("accession")
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--reencode", action="store_true")
    add_tok_args(s)
    s.set_defaults(func=cmd_train_one)

    s = sub.add_parser("scope-one")
    s.add_argument("accession")
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--fps", type=float, default=12.0)
    s.add_argument("--reencode", action="store_true")
    add_tok_args(s)
    s.set_defaults(func=cmd_scope_one)

    s = sub.add_parser("scope-stream")
    s.add_argument("accession")
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--fps", type=float, default=12.0)
    s.add_argument("--update-every", type=int, default=5)
    s.add_argument("--reencode", action="store_true")
    add_tok_args(s)
    s.set_defaults(func=cmd_scope_stream)

    s = sub.add_parser("stream")
    s.add_argument("--catalog", required=True)
    s.add_argument("--max-epochs", type=int, default=None)
    s.add_argument("--steps-per-plasmid", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--delete-cache", action="store_true")
    add_tok_args(s)
    s.set_defaults(func=cmd_stream)

    s = sub.add_parser("generate-plasmid")
    s.add_argument("--length-bp", type=int, default=10000)
    s.add_argument("--num-windows", type=int, default=None)
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--name", default="genostream_plasmid_1")
    s.add_argument("--output", default="generated/novel_plasmid.fasta")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--latent-scale", type=float, default=1.0)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--gc-bias", type=float, default=1.0)
    add_tok_args(s)
    s.set_defaults(func=cmd_generate_plasmid)

    return p

def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
PY

echo "[upgrade] DONE. Codon tokenizer support installed."
