import argparse, logging, os
from typing import Any

import numpy as np

from ..config import extract_configs, load_full_config
from ..encoding_main import compute_gc_from_encoded, encode_accession
from ..generate import generate_plasmid_sequence, generate_protein_sequence
from ..io_utils import ensure_dirs, load_state, read_catalog, save_state, setup_logging, encoded_cache_path
from ..ncbi_fetch import fetch_fasta, fetch_genbank
from ..training import cleanup_accession_files, compute_window_errors, train_on_encoded

try:
    import curses
except ImportError:
    curses = None  # type: ignore

from ..scope import run_scope_ui, run_scope_stream_ui, ScopeStreamContext

def _get_tok(args, train_cfg):
    return (getattr(args, "tokenizer", None) or train_cfg.tokenizer).lower()

def _get_frame(args, train_cfg):
    return int(getattr(args, "frame_offset", None) if getattr(args, "frame_offset", None) is not None else train_cfg.frame_offset)

def _get_min_orf(args, train_cfg):
    # AA mode: allow --min-protein-aa as a clearer alias.
    v = getattr(args, "min_protein_aa", None)
    if v is None:
        v = getattr(args, "min_orf_aa", None)
    if v is not None:
        return int(v)
    return int(getattr(train_cfg, "min_orf_aa", 90))


def _get_grounded(args, train_cfg, tok: str, src: str) -> dict:
    """Return GenBank-CDS protein filter knobs (aa+genbank only)."""
    tok = (tok or "").lower()
    src = (src or "").lower()
    if tok != "aa" or src != "genbank":
        return {
            "strict_cds": False,
            "require_translation": False,
            "x_free": False,
            "require_start_m": False,
            "reject_partial_cds": False,
            "max_protein_aa": None,
        }

    def cfg_bool(*names: str, default: bool = False) -> bool:
        for name in names:
            if hasattr(train_cfg, name):
                return bool(getattr(train_cfg, name))
        return default

    def cfg_int(*names: str):
        for name in names:
            if hasattr(train_cfg, name):
                v = getattr(train_cfg, name)
                return None if v is None else int(v)
        return None

    strict_cds = bool(getattr(args, "strict_cds", False)) or cfg_bool("protein_strict_cds_only", "strict_cds", default=False)
    require_translation = bool(getattr(args, "require_translation", False)) or cfg_bool("protein_require_translation", "require_translation", default=False)
    x_free = bool(getattr(args, "x_free", False)) or cfg_bool("protein_x_free", "x_free", default=False)
    require_start_m = bool(getattr(args, "require_start_m", False)) or cfg_bool("protein_require_start_m", "require_start_m", default=False)
    reject_partial_cds = bool(getattr(args, "reject_partial_cds", False)) or cfg_bool("protein_reject_partial_cds", "reject_partial_cds", default=False)

    max_protein_aa = getattr(args, "max_protein_aa", None)
    if max_protein_aa is None:
        max_protein_aa = cfg_int("protein_max_aa", "max_protein_aa")

    return {
        "strict_cds": bool(strict_cds),
        "require_translation": bool(require_translation),
        "x_free": bool(x_free),
        "require_start_m": bool(require_start_m),
        "reject_partial_cds": bool(reject_partial_cds),
        "max_protein_aa": (None if max_protein_aa is None else int(max_protein_aa)),
    }

def _bool_opt(v, default: bool) -> bool:
    return default if v is None else bool(v)


def _get_protein_opts(args, train_cfg):
    """Options used for aa+genbank CDS extraction and filtering."""
    return {
        "protein_strict_cds_only": _bool_opt(getattr(args, "strict_cds", None), getattr(train_cfg, "protein_strict_cds_only", False)),
        "protein_require_translation": _bool_opt(getattr(args, "require_translation", None), getattr(train_cfg, "protein_require_translation", False)),
        "protein_reject_partial": _bool_opt(getattr(args, "reject_partial_cds", None), getattr(train_cfg, "protein_reject_partial", True)),
        "protein_require_start_m": _bool_opt(getattr(args, "require_start_m", None), getattr(train_cfg, "protein_require_start_m", True)),
        "protein_x_free": _bool_opt(getattr(args, "x_free", None), getattr(train_cfg, "protein_x_free", True)),
        "protein_min_aa": int(getattr(args, "min_protein_aa", None) if getattr(args, "min_protein_aa", None) is not None else getattr(train_cfg, "protein_min_aa", getattr(train_cfg, "min_orf_aa", 90))),
        "protein_max_aa": int(getattr(args, "max_protein_aa", None) if getattr(args, "max_protein_aa", None) is not None else getattr(train_cfg, "protein_max_aa", 5000)),
    }


def _get_source(args, tok: str) -> str:
    v = getattr(args, "source", None)
    if v is not None:
        return str(v).lower()
    # Default behavior:
    #   - base/codon: FASTA (original pipeline)
    #   - aa: GenBank (prefer CDS translations when available)
    return "genbank" if tok == "aa" else "fasta"

def _ensure_record(accession: str, src: str, io_cfg, ncbi_cfg, force: bool = False) -> str:
    """Ensure the requested record exists in cache; fetch if missing."""
    src = (src or "fasta").lower()
    if src == "genbank":
        gb_dir = getattr(io_cfg, "cache_genbank_dir", "cache/genbank")
        gb_path = os.path.join(gb_dir, f"{accession}.gb")
        if not os.path.exists(gb_path) or force:
            fetch_genbank(accession, io_cfg, ncbi_cfg, force=force)
        return gb_path
    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
    if not os.path.exists(fasta_path) or force:
        fetch_fasta(accession, io_cfg, ncbi_cfg, force=force)
    return fasta_path

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



def _resolve_proteome_params(args: argparse.Namespace, train_cfg, state, tok: str, src: str) -> dict[str, Any]:
    """Resolve proteome-related knobs (curriculum, balanced sampling, denoising).

    Applied only for tok=="aa" and src=="genbank". CLI flags override config, which overrides curriculum.
    """
    tok = (tok or "").lower()
    src = (src or "").lower()

    # Defaults from config
    pol: dict[str, Any] = {
        "protein_len_min": getattr(train_cfg, "protein_len_min", None),
        "protein_len_max": getattr(train_cfg, "protein_len_max", None),
        "translation_only": bool(getattr(train_cfg, "translation_only", False)),
        "max_windows_per_protein": getattr(train_cfg, "max_windows_per_protein", None),
        "mask_prob": float(getattr(train_cfg, "aa_mask_prob", 0.05)) if tok == "aa" else 0.0,
        "span_mask_prob": float(getattr(train_cfg, "aa_span_mask_prob", 0.0)),
        "span_mask_len": int(getattr(train_cfg, "aa_span_mask_len", 0)),
        "curriculum_tag": None,
    }

    total_steps = int(state.get("total_steps", 0)) if isinstance(state, dict) else 0

    # Curriculum (optional)
    if (
        tok == "aa"
        and src == "genbank"
        and not bool(getattr(args, "no_curriculum", False))
        and bool(getattr(train_cfg, "curriculum_enabled", False))
    ):
        phases = list(getattr(train_cfg, "curriculum_phases", []) or [])
        steps = list(getattr(train_cfg, "curriculum_steps", []) or [])
        if phases:
            # Determine phase index by total_steps
            idx = 0
            if steps:
                for j, s in enumerate(steps):
                    try:
                        if total_steps >= int(s):
                            idx = j
                    except Exception:
                        pass
            idx = max(0, min(idx, len(phases) - 1))
            phase = phases[idx] if idx < len(phases) else {}
            if isinstance(phase, dict):
                for k in ("protein_len_min", "protein_len_max", "translation_only", "max_windows_per_protein", "mask_prob", "span_mask_prob", "span_mask_len"):
                    if k in phase and phase[k] is not None:
                        pol[k] = phase[k]
            pol["curriculum_tag"] = f"cur{idx}"

    # CLI overrides (only override if user explicitly set the flag)
    if getattr(args, "protein_len_min", None) is not None:
        pol["protein_len_min"] = int(getattr(args, "protein_len_min"))
    if getattr(args, "protein_len_max", None) is not None:
        pol["protein_len_max"] = int(getattr(args, "protein_len_max"))
    if getattr(args, "max_windows_per_protein", None) is not None:
        pol["max_windows_per_protein"] = int(getattr(args, "max_windows_per_protein"))

    # translation-only tri-state: --translation-only / --allow-translated
    if getattr(args, "translation_only", None) is True:
        pol["translation_only"] = True
    if getattr(args, "allow_translated", None) is True:
        pol["translation_only"] = False

    if getattr(args, "mask_prob", None) is not None:
        pol["mask_prob"] = float(getattr(args, "mask_prob"))
    if getattr(args, "span_mask_prob", None) is not None:
        pol["span_mask_prob"] = float(getattr(args, "span_mask_prob"))
    if getattr(args, "span_mask_len", None) is not None:
        pol["span_mask_len"] = int(getattr(args, "span_mask_len"))

    return pol

