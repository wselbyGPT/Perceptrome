import json
import logging
import os
from typing import Any, Dict, List

from .config import IOConfig


def read_catalog(path: str) -> List[str]:
    """Read accession IDs from a plain-text catalog (one per line, comments allowed)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalog file not found: {path}")
    accessions: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            acc = line.split()[0]
            accessions.append(acc)
    if not accessions:
        raise ValueError(f"Catalog {path} contained no accessions.")
    return accessions


def ensure_dirs(io_cfg: IOConfig) -> None:
    os.makedirs(io_cfg.cache_fasta_dir, exist_ok=True)
    os.makedirs(getattr(io_cfg, 'cache_genbank_dir', 'cache/genbank'), exist_ok=True)
    os.makedirs(io_cfg.cache_encoded_dir, exist_ok=True)
    os.makedirs(io_cfg.model_dir, exist_ok=True)
    os.makedirs(io_cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(io_cfg.logs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(io_cfg.state_file), exist_ok=True)


def load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "current_index": 0,
            "total_steps": 0,
            "plasmid_visit_counts": {},
            "last_checkpoint": None,
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def setup_logging(logs_dir: str) -> None:
    """Set up training + fetch loggers."""
    os.makedirs(logs_dir, exist_ok=True)
    train_log = os.path.join(logs_dir, "training.log")
    fetch_log = os.path.join(logs_dir, "fetch.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(train_log, mode="a"),
            logging.StreamHandler(),
        ],
    )

    fetch_handler = logging.FileHandler(fetch_log, mode="a")
    fetch_logger = logging.getLogger("fetch")
    fetch_logger.setLevel(logging.INFO)
    fetch_logger.addHandler(fetch_handler)

def encoded_cache_path(
    io_cfg: IOConfig,
    accession: str,
    tokenizer: str,
    window_size: int,
    stride: int,
    frame_offset: int,
    source: str = "fasta",
    min_orf_aa: int | None = None,
    max_windows_per_protein: int | None = None,
    protein_len_min: int | None = None,
    protein_len_max: int | None = None,
    translation_only: bool = False,
    curriculum_tag: str | None = None,
) -> str:
    """
    Encoded cache file path that avoids mixing tokenizers / window params.

    Examples:
      ABC.base.w512.s256.npy
      ABC.codon.w510.s255.f0.npy
      ABC.aa.w256.s128.min90.npy
    """
    tok = tokenizer.lower()
    tag = f"{tok}.w{int(window_size)}.s{int(stride)}"
    src = source.lower()
    if src == "fasta":
        tag += ".srcfa"
    elif src == "genbank":
        tag += ".srcgb"
    else:
        tag += f".src{src}"
    if tok == "codon":
        tag += f".f{int(frame_offset)}"
    if tok == "aa" and min_orf_aa is not None:
        tag += f".min{int(min_orf_aa)}"
    if tok == "aa" and max_windows_per_protein is not None:
        tag += f".wpp{int(max_windows_per_protein)}"
    if tok == "aa" and protein_len_min is not None:
        tag += f".pmin{int(protein_len_min)}"
    if tok == "aa" and protein_len_max is not None:
        tag += f".pmax{int(protein_len_max)}"
    if tok == "aa" and translation_only:
        tag += ".tronly"
    if tok == "aa" and curriculum_tag:
        tag += f".{curriculum_tag}"
    import os
    fname = f"{accession}.{tag}.npy"
    return os.path.join(io_cfg.cache_encoded_dir, fname)
