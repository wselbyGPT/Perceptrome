import json
import logging
import os
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import IOConfig
from .genome import DEFAULT_GENE_REGISTRY, Genome


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


def write_catalog(path: str, accessions: Sequence[str], header: Optional[Sequence[str]] = None) -> None:
    """Write a plain-text accession catalog.

    Args:
        path: Destination path for the catalog file.
        accessions: Accessions to write (one per line).
        header: Optional comment lines written before entries.
    """
    final_accessions = [str(acc).strip() for acc in accessions if str(acc).strip()]
    if not final_accessions:
        raise ValueError("Catalog write aborted: no accessions to write.")

    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if header:
            for line in header:
                txt = str(line).strip()
                if not txt:
                    continue
                f.write(txt if txt.startswith("#") else f"# {txt}")
                f.write("\n")
        for acc in final_accessions:
            f.write(f"{acc}\n")


def select_unique_accessions(
    category_quotas: Iterable[Tuple[str, int]],
    category_candidates: Mapping[str, Sequence[str]],
    seed: Optional[int] = None,
    shuffle_within_category: bool = False,
) -> List[str]:
    """Build a unique accession list across categories in deterministic order.

    Categories are processed in the provided ``category_quotas`` order.
    Duplicate accessions are removed globally across all categories.
    """
    seen: set[str] = set()
    selected: List[str] = []
    rng = random.Random(seed)

    for category, quota in category_quotas:
        q = max(0, int(quota))
        if q == 0:
            continue

        candidates = [str(acc).strip() for acc in category_candidates.get(category, []) if str(acc).strip()]
        if shuffle_within_category:
            rng.shuffle(candidates)

        picked = 0
        for accession in candidates:
            if accession in seen:
                continue
            seen.add(accession)
            selected.append(accession)
            picked += 1
            if picked >= q:
                break

    return selected


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
        return default_state()
    with open(path, "r", encoding="utf-8") as f:
        return normalize_state(json.load(f))


def default_state() -> Dict[str, Any]:
    genome = Genome.from_dict(None, DEFAULT_GENE_REGISTRY)
    return {
        "current_index": 0,
        "total_steps": 0,
        "plasmid_visit_counts": {},
        "epoch": 0,
        "last_checkpoint": None,
        "genome": genome.to_dict(),
    }


def normalize_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    state = default_state()
    if isinstance(raw, dict):
        state.update(raw)

    genome_payload = raw.get("genome") if isinstance(raw, dict) else None
    if genome_payload is None and isinstance(raw, dict):
        legacy_genes = raw.get("genes")
        if isinstance(legacy_genes, dict):
            genome_payload = {"genes": legacy_genes}

    state["genome"] = Genome.from_dict(genome_payload, DEFAULT_GENE_REGISTRY).to_dict()
    return state


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


def load_or_encode_accession(
    accession: str,
    io_cfg: IOConfig,
    *,
    tokenizer: str,
    window_size: int,
    stride: int,
    frame_offset: int,
    source: str,
    cache_kw: Dict[str, Any],
    protein_opts: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Load encoded windows for ``accession`` from the cache, encoding on miss.

    Caller is responsible for ensuring the underlying record (FASTA/GenBank)
    exists on disk first; this helper performs only cache lookup + encoding.
    """
    from .encoding_main import encode_accession

    enc_path = encoded_cache_path(
        io_cfg, accession, tokenizer, window_size, stride, frame_offset,
        source=source, **cache_kw,
    )
    if os.path.exists(enc_path):
        return np.load(enc_path)
    return encode_accession(
        accession, io_cfg, window_size, stride,
        tokenizer=tokenizer,
        frame_offset=frame_offset,
        min_orf_aa=cache_kw.get("min_orf_aa"),
        source=source,
        max_windows_per_protein=cache_kw.get("max_windows_per_protein"),
        protein_len_min=cache_kw.get("protein_len_min"),
        protein_len_max=cache_kw.get("protein_len_max"),
        translation_only=bool(cache_kw.get("translation_only", False)),
        protein_opts=protein_opts or {},
        save_to_disk=True,
        out_path=enc_path,
    )
