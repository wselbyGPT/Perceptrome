#!/usr/bin/env bash
set -euo pipefail

echo "[upgrade] Installing AA/proteome tokenizer (ORF -> protein -> AA windows) ..."

# -----------------------------
# genostream/config.py
# -----------------------------
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

    protein_window_aa: int
    protein_stride_aa: int
    min_orf_aa: int

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

            protein_window_aa=int(t.get("protein_window_aa", 256)),
            protein_stride_aa=int(t.get("protein_stride_aa", 128)),
            min_orf_aa=int(t.get("min_orf_aa", 90)),
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

# -----------------------------
# genostream/io_utils.py
# (update encoded_cache_path signature; keep backward compatible)
# -----------------------------
python3 - <<'PY'
import pathlib, re
p = pathlib.Path("genostream/io_utils.py")
txt = p.read_text(encoding="utf-8")

# Replace existing encoded_cache_path (if any) with upgraded version
pattern = r"def encoded_cache_path\(.*?\n(?:.*\n)*?\n"
if "def encoded_cache_path" in txt:
    # crude but effective: cut old function block by finding next def after it
    start = txt.find("def encoded_cache_path")
    rest = txt[start:]
    m = re.search(r"\n(?=def [a-zA-Z_]+\()", rest)
    end = start + (m.start() if m else len(rest))
    txt = txt[:start] + txt[end:]

# Append upgraded function (won't affect others)
txt = txt.rstrip() + """

def encoded_cache_path(
    io_cfg: IOConfig,
    accession: str,
    tokenizer: str,
    window_size: int,
    stride: int,
    frame_offset: int,
    min_orf_aa: int | None = None,
) -> str:
    \"""
    Encoded cache file path that avoids mixing tokenizers / window params.

    Examples:
      ABC.base.w512.s256.npy
      ABC.codon.w510.s255.f0.npy
      ABC.aa.w256.s128.min90.npy
    \"""
    tok = tokenizer.lower()
    tag = f"{tok}.w{int(window_size)}.s{int(stride)}"
    if tok == "codon":
        tag += f".f{int(frame_offset)}"
    if tok == "aa" and min_orf_aa is not None:
        tag += f".min{int(min_orf_aa)}"
    import os
    fname = f"{accession}.{tag}.npy"
    return os.path.join(io_cfg.cache_encoded_dir, fname)
"""
p.write_text(txt, encoding="utf-8")
print("[upgrade] patched genostream/io_utils.py")
PY

# -----------------------------
# genostream/encoding.py
# (add AA tokenizer + ORF finder + translation)
# -----------------------------
cat > genostream/encoding.py <<'PY'
import logging, os
from typing import List, Optional, Tuple

import numpy as np
from .config import IOConfig

# -----------------------------
# CODON tokenizer constants
# -----------------------------
_CODON_BASES = "ACGT"
CODONS: List[str] = [a + b + c for a in _CODON_BASES for b in _CODON_BASES for c in _CODON_BASES]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS)}
UNK_IDX = len(CODONS)  # 64
IDX_TO_CODON: List[str] = CODONS + ["NNN"]
CODON_VOCAB_SIZE = len(IDX_TO_CODON)  # 65

_gc_frac = []
_gc_count = []
for c in CODONS:
    gc = sum(1 for ch in c if ch in ("G", "C"))
    _gc_count.append(float(gc))
    _gc_frac.append(float(gc) / 3.0)
_gc_count.append(0.0)
_gc_frac.append(0.0)
GC_COUNT_PER_TOKEN = np.array(_gc_count, dtype=np.float32)     # (65,)
GC_FRAC_PER_TOKEN = np.array(_gc_frac, dtype=np.float32)       # (65,)

# -----------------------------
# AA tokenizer constants
# -----------------------------
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # 20
AA_UNK = "X"
AA_VOCAB = AA_ALPHABET + AA_UNK       # 21
AA_TO_IDX = {a: i for i, a in enumerate(AA_VOCAB)}
IDX_TO_AA = list(AA_VOCAB)
AA_VOCAB_SIZE = len(AA_VOCAB)

HYDROPHOBIC = set(list("AILMFWVYC"))  # simple set
HYDRO_IDX = np.array([1.0 if aa in HYDROPHOBIC else 0.0 for aa in IDX_TO_AA], dtype=np.float32)

# Standard genetic code (DNA codons)
CODON_TO_AA = {
    # Phenylalanine
    "TTT":"F","TTC":"F",
    # Leucine
    "TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    # Isoleucine
    "ATT":"I","ATC":"I","ATA":"I",
    # Methionine (start)
    "ATG":"M",
    # Valine
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    # Serine
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","AGT":"S","AGC":"S",
    # Proline
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    # Threonine
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    # Alanine
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    # Tyrosine
    "TAT":"Y","TAC":"Y",
    # Histidine
    "CAT":"H","CAC":"H",
    # Glutamine
    "CAA":"Q","CAG":"Q",
    # Asparagine
    "AAT":"N","AAC":"N",
    # Lysine
    "AAA":"K","AAG":"K",
    # Aspartic acid
    "GAT":"D","GAC":"D",
    # Glutamic acid
    "GAA":"E","GAG":"E",
    # Cysteine
    "TGT":"C","TGC":"C",
    # Tryptophan
    "TGG":"W",
    # Arginine
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
    # Glycine
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}
STOP_CODONS = {"TAA","TAG","TGA"}
START_CODON = "ATG"

def tokenizer_meta(tokenizer: str, window_size: int) -> Tuple[int, int]:
    """
    Returns (seq_len_units, vocab_size)

    base: units = window_size (bp), vocab = 4
    codon: units = window_size//3 (codons), vocab = 65
    aa: units = window_size (amino acids), vocab = 21
    """
    tok = tokenizer.lower()
    if tok == "base":
        return window_size, 4
    if tok == "codon":
        if window_size % 3 != 0:
            raise ValueError(f"codon tokenizer requires window_size divisible by 3 (got {window_size})")
        return window_size // 3, CODON_VOCAB_SIZE
    if tok == "aa":
        if window_size <= 0:
            raise ValueError("aa tokenizer requires window_size > 0 (amino acids)")
        return window_size, AA_VOCAB_SIZE
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

def reverse_complement(seq: str) -> str:
    comp = {"A":"T","C":"G","G":"C","T":"A","N":"N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))

# -----------------------------
# Base encoding
# -----------------------------
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

# -----------------------------
# Codon encoding
# -----------------------------
def encode_sequence_codons(seq: str, window_size_bp: int, stride_bp: int, frame_offset: int = 0) -> np.ndarray:
    if frame_offset not in (0, 1, 2):
        raise ValueError("frame_offset must be 0, 1, or 2")
    if window_size_bp % 3 != 0:
        raise ValueError(f"codon tokenizer requires window_size divisible by 3 (got {window_size_bp})")
    if stride_bp % 3 != 0:
        raise ValueError(f"codon tokenizer requires stride divisible by 3 (got {stride_bp})")

    seq = seq.upper()
    if len(seq) <= frame_offset + 2:
        window_codons = window_size_bp // 3
        out = np.zeros((1, window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        return out

    usable = len(seq) - ((len(seq) - frame_offset) % 3)
    codon_count = max(0, (usable - frame_offset) // 3)
    window_codons = window_size_bp // 3
    stride_codons = stride_bp // 3

    def codon_at(i: int) -> int:
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

# -----------------------------
# Proteome (ORF -> AA) extraction + encoding
# -----------------------------
def translate_orf(dna: str) -> str:
    """
    dna length must be multiple of 3; stops are not included here.
    """
    dna = dna.upper()
    aas: List[str] = []
    for i in range(0, len(dna) - 2, 3):
        cod = dna[i:i+3]
        if cod in STOP_CODONS:
            break
        aa = CODON_TO_AA.get(cod, "X")
        aas.append(aa)
    return "".join(aas)

def find_orfs_proteins(seq: str, min_orf_aa: int = 90) -> List[str]:
    """
    Very simple ORF finder:
      - scans 3 frames on forward and reverse-complement
      - start codon ATG
      - stops at TAA/TAG/TGA
      - returns translated proteins length >= min_orf_aa
    """
    seq = seq.upper()
    proteins: List[str] = []

    def scan_strand(s: str):
        L = len(s)
        for frame in (0, 1, 2):
            i = frame
            while i + 2 < L:
                cod = s[i:i+3]
                if cod == START_CODON:
                    # find stop
                    j = i
                    while j + 2 < L:
                        cod2 = s[j:j+3]
                        if cod2 in STOP_CODONS:
                            orf_dna = s[i:j]  # exclude stop codon
                            if len(orf_dna) >= min_orf_aa * 3:
                                prot = translate_orf(orf_dna)
                                if len(prot) >= min_orf_aa:
                                    proteins.append(prot)
                            i = j + 3
                            break
                        j += 3
                    else:
                        # no stop found
                        i += 3
                else:
                    i += 3

    scan_strand(seq)
    scan_strand(reverse_complement(seq))
    return proteins

def encode_proteins_aa_windows(proteins: List[str], window_aa: int, stride_aa: int) -> np.ndarray:
    """
    Returns: (num_windows, window_aa, 21)
    Pads short proteins to one window.
    """
    if window_aa <= 0 or stride_aa <= 0:
        raise ValueError("window_aa and stride_aa must be > 0")
    windows: List[np.ndarray] = []

    for prot in proteins:
        prot = prot.strip().upper()
        if not prot:
            continue

        if len(prot) < window_aa:
            arr = np.zeros((window_aa, AA_VOCAB_SIZE), dtype=np.float32)
            for i, aa in enumerate(prot):
                idx = AA_TO_IDX.get(aa, AA_TO_IDX[AA_UNK])
                arr[i, idx] = 1.0
            windows.append(arr)
            continue

        for start in range(0, len(prot) - window_aa + 1, stride_aa):
            chunk = prot[start:start+window_aa]
            arr = np.zeros((window_aa, AA_VOCAB_SIZE), dtype=np.float32)
            for i, aa in enumerate(chunk):
                idx = AA_TO_IDX.get(aa, AA_TO_IDX[AA_UNK])
                arr[i, idx] = 1.0
            windows.append(arr)

    if not windows:
        # return a single all-zero window to keep pipeline alive
        return np.zeros((1, window_aa, AA_VOCAB_SIZE), dtype=np.float32)

    return np.stack(windows, axis=0)

def encode_accession(
    accession: str,
    io_cfg: IOConfig,
    window_size: int,
    stride: int,
    tokenizer: str = "base",
    frame_offset: int = 0,
    min_orf_aa: int = 90,
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
        logging.info(f"{accession}: encoded(BASE) len={len(seq)} -> windows={encoded.shape[0]} shape={encoded.shape}")

    elif tok == "codon":
        encoded = encode_sequence_codons(seq, window_size_bp=window_size, stride_bp=stride, frame_offset=frame_offset)
        logging.info(f"{accession}: encoded(CODON) len={len(seq)} -> windows={encoded.shape[0]} shape={encoded.shape}")

    elif tok == "aa":
        proteins = find_orfs_proteins(seq, min_orf_aa=min_orf_aa)
        encoded = encode_proteins_aa_windows(proteins, window_aa=window_size, stride_aa=stride)
        logging.info(
            f"{accession}: encoded(AA) genome_len={len(seq)} proteins={len(proteins)} "
            f"-> windows={encoded.shape[0]} window_aa={window_size} stride_aa={stride} shape={encoded.shape}"
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
    Returns per-window metric:
      - base/codon: GC fraction
      - aa: hydrophobic fraction (simple proxy), returned in same slot
    """
    tok = tokenizer.lower()

    if tok == "base":
        if encoded.ndim != 3 or encoded.shape[2] != 4:
            raise ValueError("base encoded must have shape (num_windows, window_size, 4)")
        gc_counts = encoded[:, :, 1] + encoded[:, :, 2]
        window_size = encoded.shape[1]
        return (gc_counts.sum(axis=1) / float(window_size)).astype(np.float32)

    if tok == "codon":
        if encoded.ndim != 3 or encoded.shape[2] != CODON_VOCAB_SIZE:
            raise ValueError("codon encoded must have shape (num_windows, window_codons, 65)")
        gc_pos = (encoded * GC_FRAC_PER_TOKEN[None, None, :]).sum(axis=2)
        return gc_pos.mean(axis=1).astype(np.float32)

    if tok == "aa":
        if encoded.ndim != 3 or encoded.shape[2] != AA_VOCAB_SIZE:
            raise ValueError("aa encoded must have shape (num_windows, window_aa, 21)")
        # expected hydrophobic indicator per position, then mean
        hydro_pos = (encoded * HYDRO_IDX[None, None, :]).sum(axis=2)  # (N, W)
        return hydro_pos.mean(axis=1).astype(np.float32)

    raise ValueError(f"Unknown tokenizer: {tokenizer}")
PY

# -----------------------------
# genostream/cli.py
# (add tokenizer=aa, min-orf-aa, and aa-default window/stride)
# -----------------------------
cat > genostream/cli.py <<'PY'
#!/usr/bin/env python3
import argparse, logging, os
from typing import Any

import numpy as np

from .config import extract_configs, load_full_config
from .encoding import compute_gc_from_encoded, encode_accession
from .generate import generate_plasmid_sequence, generate_protein_sequence
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

def _get_min_orf(args, train_cfg):
    v = getattr(args, "min_orf_aa", None)
    return int(v if v is not None else train_cfg.min_orf_aa)

def _pick_window_stride(args, train_cfg, tok: str):
    if tok == "aa":
        w = args.window_size if args.window_size is not None else train_cfg.protein_window_aa
        s = args.stride if args.stride is not None else train_cfg.protein_stride_aa
    else:
        w = args.window_size if args.window_size is not None else train_cfg.window_size
        s = args.stride if args.stride is not None else train_cfg.stride
    return int(w), int(s)

def _validate_tok_params(tokenizer: str, window_size: int, stride: int, frame: int):
    if tokenizer == "codon":
        if window_size % 3 != 0:
            raise ValueError(f"--tokenizer codon requires window_size divisible by 3 (got {window_size})")
        if stride % 3 != 0:
            raise ValueError(f"--tokenizer codon requires stride divisible by 3 (got {stride})")
        if frame not in (0,1,2):
            raise ValueError("--frame-offset must be 0,1,2")
    elif tokenizer == "aa":
        if window_size <= 0 or stride <= 0:
            raise ValueError("--tokenizer aa requires positive --window-size/--stride (amino acids)")
    else:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size/stride must be positive")

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

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{args.accession}.fasta")
    if not os.path.exists(fasta_path):
        print(f"FASTA for {args.accession} not found at {fasta_path}. Run fetch-one first.")
        return 1

    out_path = encoded_cache_path(io_cfg, args.accession, tok, window_size, stride, frame, min_orf_aa=(min_orf if tok=="aa" else None))
    encoded = encode_accession(
        args.accession, io_cfg, window_size, stride,
        tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
        save_to_disk=True, out_path=out_path
    )
    print(f"{args.accession}: encoded tokenizer={tok} -> shape={encoded.shape} saved={out_path}")
    return 0

def cmd_train_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg); setup_logging(io_cfg.logs_dir)
    state = load_state(io_cfg.state_file)

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    batch_size = args.batch_size or train_cfg.batch_size
    steps = args.steps or train_cfg.steps_per_plasmid

    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{args.accession}.fasta")
    if not os.path.exists(fasta_path):
        logging.info(f"{args.accession}: FASTA not found; fetching.")
        fetch_fasta(args.accession, io_cfg, ncbi_cfg, force=False)

    enc_path = encoded_cache_path(io_cfg, args.accession, tok, window_size, stride, frame, min_orf_aa=(min_orf if tok=="aa" else None))
    if os.path.exists(enc_path) and not args.reencode:
        encoded = np.load(enc_path)
        logging.info(f"{args.accession}: using cached encoded at {enc_path} shape={encoded.shape}")
    else:
        encoded = encode_accession(
            args.accession, io_cfg, window_size, stride,
            tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
            save_to_disk=True, out_path=enc_path
        )

    last_total = train_on_encoded(
        args.accession, encoded,
        steps=steps, batch_size=batch_size,
        state=state, io_cfg=io_cfg, train_cfg=train_cfg,
        tokenizer=tok, window_size_bp=window_size,  # "units" for aa, ok
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

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    enc_path = encoded_cache_path(io_cfg, args.accession, tok, window_size, stride, frame, min_orf_aa=(min_orf if tok=="aa" else None))
    if os.path.exists(enc_path) and not args.reencode:
        encoded = np.load(enc_path)
    else:
        encoded = encode_accession(
            args.accession, io_cfg, window_size, stride,
            tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
            save_to_disk=True, out_path=enc_path
        )

    errors = compute_window_errors(args.accession, encoded, io_cfg=io_cfg, train_cfg=train_cfg, tokenizer=tok, window_size_bp=window_size)
    metric = compute_gc_from_encoded(encoded, tokenizer=tok)  # for aa: hydrophobic fraction

    curses.wrapper(
        run_scope_ui,
        accession=args.accession,
        errors=errors,
        gc_values=metric,
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

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    steps = args.steps or train_cfg.steps_per_plasmid
    batch_size = args.batch_size or train_cfg.batch_size

    enc_path = encoded_cache_path(io_cfg, args.accession, tok, window_size, stride, frame, min_orf_aa=(min_orf if tok=="aa" else None))
    if os.path.exists(enc_path) and not args.reencode:
        encoded = np.load(enc_path)
    else:
        encoded = encode_accession(
            args.accession, io_cfg, window_size, stride,
            tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
            save_to_disk=True, out_path=enc_path
        )

    metric = compute_gc_from_encoded(encoded, tokenizer=tok)

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
        gc_values=metric,
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

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    batch_size = args.batch_size or train_cfg.batch_size
    steps_per_plasmid = args.steps_per_plasmid or train_cfg.steps_per_plasmid
    max_epochs = args.max_epochs or train_cfg.max_stream_epochs

    epoch = int(state.get("epoch", 0))

    while epoch < max_epochs:
        indices = list(range(len(accessions)))
        if train_cfg.shuffle_catalog:
            random.shuffle(indices)

        for idx in indices:
            acc = accessions[idx]

            fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{acc}.fasta")
            if not os.path.exists(fasta_path):
                fetch_fasta(acc, io_cfg, ncbi_cfg, force=False)

            enc_path = encoded_cache_path(io_cfg, acc, tok, window_size, stride, frame, min_orf_aa=(min_orf if tok=="aa" else None))
            if os.path.exists(enc_path):
                encoded = np.load(enc_path)
            else:
                encoded = encode_accession(
                    acc, io_cfg, window_size, stride,
                    tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
                    save_to_disk=True, out_path=enc_path
                )

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
    if tok not in ("base", "codon"):
        raise ValueError("generate-plasmid supports tokenizer base|codon only (use generate-protein for aa).")

    window_size = args.window_size if args.window_size is not None else train_cfg.window_size
    stride = train_cfg.stride
    frame = _get_frame(args, train_cfg)
    _validate_tok_params(tok, int(window_size), int(stride), frame)

    seq = generate_plasmid_sequence(
        train_cfg=train_cfg,
        io_cfg=io_cfg,
        length_bp=args.length_bp,
        num_windows=args.num_windows,
        window_size_bp=int(window_size),
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

def cmd_generate_protein(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg); setup_logging(io_cfg.logs_dir)

    tok = "aa"
    window_aa = args.window_aa if args.window_aa is not None else train_cfg.protein_window_aa

    seq = generate_protein_sequence(
        train_cfg=train_cfg,
        io_cfg=io_cfg,
        length_aa=args.length_aa,
        num_windows=args.num_windows,
        window_aa=int(window_aa),
        seed=args.seed,
        latent_scale=args.latent_scale,
        temperature=args.temperature,
        name=args.name,
        output_path=args.output,
    )
    print(f"[generate-protein] wrote {len(seq)} aa -> {args.output}")
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
        sp.add_argument("--tokenizer", choices=["base","codon","aa"], default=None, help="Override tokenizer (default from config)")
        sp.add_argument("--frame-offset", type=int, choices=[0,1,2], default=None, help="Codon frame offset (default from config)")
        sp.add_argument("--min-orf-aa", type=int, default=None, help="AA tokenizer: minimum ORF length in amino acids (default from config)")

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

    s = sub.add_parser("generate-protein")
    s.add_argument("--length-aa", type=int, default=600)
    s.add_argument("--num-windows", type=int, default=None)
    s.add_argument("--window-aa", type=int, default=None)
    s.add_argument("--name", default="genostream_protein_1")
    s.add_argument("--output", default="generated/novel_protein.faa")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--latent-scale", type=float, default=1.0)
    s.add_argument("--temperature", type=float, default=1.0)
    s.set_defaults(func=cmd_generate_protein)

    return p

def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
PY

# -----------------------------
# genostream/generate.py
# (add protein generator; keep plasmid generator)
# -----------------------------
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
from .encoding import tokenizer_meta, IDX_TO_CODON, CODON_VOCAB_SIZE, GC_COUNT_PER_TOKEN, IDX_TO_AA, AA_VOCAB_SIZE

def _sample_index(weights: np.ndarray, temperature: float) -> int:
    w = weights.astype(np.float64)
    w = np.clip(w, 1e-9, None)
    temperature = max(1e-3, float(temperature))
    if temperature != 1.0:
        w = w ** (1.0 / temperature)
    s = w.sum()
    if s <= 0:
        return int(np.random.randint(0, w.shape[0]))
    w /= s
    return int(np.random.choice(w.shape[0], p=w))

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
    if tok not in ("base", "codon"):
        raise ValueError("generate_plasmid_sequence only supports base|codon")

    if tok == "codon" and length_bp % 3 != 0:
        length_bp = (length_bp // 3) * 3

    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
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

    temperature = float(temperature)
    latent_scale = float(latent_scale)
    gc_bias = float(gc_bias)

    seq_parts: List[str] = []

    with torch.no_grad():
        for _ in range(n_windows):
            z = torch.randn(1, hidden_dim, device=device) * latent_scale
            recon_flat = model.decode(z)   # (1, seq_len*vocab)
            recon = recon_flat.view(seq_len, vocab_size).cpu().numpy()

            if tok == "base":
                idx_to_base = ["A", "C", "G", "T"]
                for j in range(seq_len):
                    w = recon[j].copy()
                    # apply gc bias to C/G
                    if gc_bias != 1.0:
                        w[1] *= gc_bias
                        w[2] *= gc_bias
                    idx = _sample_index(w, temperature)
                    seq_parts.append(idx_to_base[idx])
            else:
                for j in range(seq_len):
                    w = recon[j].copy()
                    if gc_bias != 1.0:
                        w *= (gc_bias ** GC_COUNT_PER_TOKEN[: w.shape[0]])
                    idx = _sample_index(w, temperature)
                    seq_parts.append(IDX_TO_CODON[idx])

    seq = "".join(seq_parts)
    seq = seq[:target_bp]

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

    return seq

def generate_protein_sequence(
    train_cfg: TrainingConfig,
    io_cfg: IOConfig,
    length_aa: int,
    num_windows: Optional[int],
    window_aa: int,
    seed: Optional[int],
    latent_scale: float,
    temperature: float,
    name: str,
    output_path: str,
) -> str:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")

    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = get_device()
    tok = "aa"
    seq_len, vocab_size = tokenizer_meta(tok, window_aa)
    assert vocab_size == AA_VOCAB_SIZE
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
        target_aa = n_windows * window_aa
    else:
        n_windows = (length_aa + window_aa - 1) // window_aa
        target_aa = length_aa

    temperature = float(temperature)
    latent_scale = float(latent_scale)

    aa_chars: List[str] = []

    with torch.no_grad():
        for _ in range(n_windows):
            z = torch.randn(1, hidden_dim, device=device) * latent_scale
            recon_flat = model.decode(z)
            recon = recon_flat.view(seq_len, vocab_size).cpu().numpy()

            for j in range(seq_len):
                idx = _sample_index(recon[j], temperature)
                aa_chars.append(IDX_TO_AA[idx])

    seq = "".join(aa_chars)[:target_aa]

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

    return seq
PY

echo "[upgrade] DONE."
