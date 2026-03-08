import argparse
import datetime
import difflib
import json
import logging
import os
import random
import subprocess
import sys
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np

from perceptrome.cli.common import (
    extract_configs, load_full_config,
    compute_gc_from_encoded, encode_accession,
    generate_plasmid_sequence, generate_protein_sequence,
    ensure_dirs, load_state, read_catalog, save_state, setup_logging, encoded_cache_path, default_state,
    fetch_fasta, fetch_genbank,
    cleanup_accession_files, compute_window_errors, train_on_encoded,
    curses,
    run_scope_ui, run_scope_stream_ui, ScopeStreamContext,
    _get_tok, _get_frame, _get_min_orf, _get_grounded, _get_protein_opts,
    _get_source, _ensure_record,
)
from perceptrome.catalog_schema import parse_catalog_schema
from perceptrome.encoding.bio_ast_builder import BioASTBuilder
from perceptrome.encoding.genbank_features import parse_cds_features_from_genbank
from perceptrome.io_utils import select_unique_accessions, write_catalog
from perceptrome.encoding.parse import parse_fasta_sequence, parse_genbank_dna
from perceptrome.pretrain import PretrainPipelineConfig, run_pretraining
from perceptrome.run_layout import ensure_run_layout, path_in_run, update_run_manifest
from perceptrome.jobs import JobEngine, JobSpec
from perceptrome.jobs.manifest_schema import extract_tokenizer_encoding_config
from perceptrome.jobs.manifest_writer import (
    config_hash,
    iso_now,
    sidecar_manifest_path,
    write_sidecar_run_manifest,
)


# -----------------------------
# Small helpers
# -----------------------------
def _pick_window_stride(args, train_cfg, tok: str) -> Tuple[int, int]:
    # Prefer CLI flags; fall back to config.
    if tok == "aa":
        ws = getattr(args, "window_size", None)
        st = getattr(args, "stride", None)
        if ws is None:
            ws = getattr(train_cfg, "protein_window_aa", None) or getattr(train_cfg, "window_size", None)
        if st is None:
            st = getattr(train_cfg, "protein_stride_aa", None) or getattr(train_cfg, "stride", None)
    else:
        ws = getattr(args, "window_size", None) or getattr(train_cfg, "window_size", None)
        st = getattr(args, "stride", None) or getattr(train_cfg, "stride", None)

    if ws is None or st is None:
        raise ValueError("window_size/stride not set (use --window-size/--stride or set in config)")

    ws = int(ws)
    st = int(st)

    if tok == "codon":
        if ws % 3 != 0:
            raise ValueError(f"codon tokenizer requires --window-size divisible by 3 (got {ws})")
        if st % 3 != 0:
            raise ValueError(f"codon tokenizer requires --stride divisible by 3 (got {st})")

    return ws, st


def _validate_tok_params(tok: str, window_size: int, stride: int, frame_offset: int) -> None:
    tok = (tok or "").lower()
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be > 0")

    if tok == "codon":
        if window_size % 3 != 0:
            raise ValueError(f"codon tokenizer requires window_size divisible by 3 (got {window_size})")
        if stride % 3 != 0:
            raise ValueError(f"codon tokenizer requires stride divisible by 3 (got {stride})")
        if frame_offset not in (0, 1, 2):
            raise ValueError(f"frame_offset must be 0,1,2 for codon tokenizer (got {frame_offset})")
        return

    if tok == "base":
        return

    if tok == "aa":
        # frame_offset irrelevant in aa-mode; ok.
        return

    raise ValueError(f"Unknown tokenizer: {tok}")


def _resolve_proteome_params(args, train_cfg, state=None, tok: str = "base", src: str = "fasta") -> Dict[str, Any]:
    """
    Returns knobs relevant to AA/proteome runs (but safe to call always).
    IMPORTANT: encoded_cache_path() does NOT accept grounded protein keys.
    We keep protein_opts separate for encode_accession().
    """
    tok = (tok or "base").lower()
    src = (src or "fasta").lower()

    pol: Dict[str, Any] = {
        "max_windows_per_protein": getattr(args, "max_windows_per_protein", None),
        "protein_len_min": getattr(args, "protein_len_min", None),
        "protein_len_max": getattr(args, "protein_len_max", None),
        "translation_only": bool(getattr(args, "translation_only", False)),
        # grounded/compat protein knobs bundled here for encode_accession()
        "protein_opts": _get_grounded(args, train_cfg, tok, src),
        # optional tag if you add curriculum later; harmless if None
        "curriculum_tag": getattr(args, "curriculum_tag", None) if hasattr(args, "curriculum_tag") else None,
    }
    return pol


def _cache_kwargs(tok: str, min_orf: int, pol: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only kwargs that encoded_cache_path() is allowed to receive.
    """
    kw: Dict[str, Any] = {
        "min_orf_aa": (min_orf if tok == "aa" else None),
        "max_windows_per_protein": (pol.get("max_windows_per_protein") if tok == "aa" else None),
        "protein_len_min": (pol.get("protein_len_min") if tok == "aa" else None),
        "protein_len_max": (pol.get("protein_len_max") if tok == "aa" else None),
        "translation_only": (bool(pol.get("translation_only", False)) if tok == "aa" else False),
        "curriculum_tag": pol.get("curriculum_tag"),
    }
    return kw


def _apply_cli_training_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    model_type = getattr(args, "model_type", None)
    if model_type is not None:
        cfg.setdefault("training", {})["model_type"] = str(model_type).lower()
    return cfg


def _is_checkpoint_model_mismatch(err: Exception) -> bool:
    msg = str(err)
    return "Checkpoint model_type=" in msg and "requested model_type=" in msg


def _kmer_set(seq: str, k: int) -> set[str]:
    if k <= 0 or len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _jaccard_kmers(a: str, b: str, k: int = 9) -> float:
    ka = _kmer_set(a, k)
    kb = _kmer_set(b, k)
    if not ka and not kb:
        return 1.0
    if not ka or not kb:
        return 0.0
    return float(len(ka & kb) / len(ka | kb))


def _sequence_similarity(a: str, b: str) -> float:
    """Similarity score in [0,1] based on global edit-like matching."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def _gc_fraction(seq: str) -> float:
    seq = (seq or "").upper()
    if not seq:
        return 0.0
    return float((seq.count("G") + seq.count("C")) / len(seq))


def _reference_score(generated_seq: str, ref_seq: str) -> Dict[str, float]:
    seq_sim = _sequence_similarity(generated_seq, ref_seq)
    kmer_sim = _jaccard_kmers(generated_seq, ref_seq, k=9)
    gc_delta = abs(_gc_fraction(generated_seq) - _gc_fraction(ref_seq))
    length_ratio = (
        min(len(generated_seq), len(ref_seq)) / max(len(generated_seq), len(ref_seq))
        if generated_seq and ref_seq
        else 0.0
    )
    score = (0.55 * seq_sim) + (0.30 * kmer_sim) + (0.10 * length_ratio) + (0.05 * (1.0 - gc_delta))
    return {
        "score": float(score),
        "seq_similarity": float(seq_sim),
        "kmer_jaccard": float(kmer_sim),
        "gc_delta": float(gc_delta),
        "length_ratio": float(length_ratio),
    }


# -----------------------------
# Commands
# -----------------------------
def cmd_init(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, _, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    state = default_state()
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


def cmd_catalog_generate(args: argparse.Namespace) -> int:
    schema = parse_catalog_schema(args.schema)

    category_quotas = []
    category_candidates = {}
    for category in schema.categories:
        category_quotas.append((category.name, category.count))
        category_candidates[category.name] = read_catalog(category.source)

    selected = select_unique_accessions(
        category_quotas,
        category_candidates,
        seed=schema.seed,
        shuffle_within_category=schema.shuffle_within_category,
    )
    if not selected:
        raise ValueError("No accessions were selected from schema categories")

    header = [
        f"Generated by perceptrome catalog-generate",
        f"schema: {args.schema}",
        f"total_accessions: {len(selected)}",
    ]
    write_catalog(args.output, selected, header=header)

    print(f"Wrote catalog: {args.output}")
    print(f"  selected accessions: {len(selected)}")
    for category_name, quota in category_quotas:
        print(f"  - {category_name}: quota={quota}, candidates={len(category_candidates.get(category_name, []))}")
    return 0


def _default_split_path(io_cfg, split_name: str) -> str:
    state_dir = os.path.dirname(io_cfg.state_file) or "state"
    return os.path.join(state_dir, "splits", f"{split_name}.json")



def _new_experiment_id(prefix: str = "exp") -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def _run_local_io_cfg(io_cfg):
    layout = ensure_run_layout()
    io_cfg.checkpoints_dir = path_in_run(layout, "artifacts", "checkpoints")
    io_cfg.model_dir = path_in_run(layout, "artifacts", "model")
    return io_cfg


def _write_fetch_manifest(
    *,
    path: str,
    accession: str,
    source: str,
    cfg_path: str,
    cfg_hash: str,
    ncbi_cfg,
) -> str:
    return write_sidecar_run_manifest(
        target_path=path,
        run_kind="fetch_record",
        config_path=cfg_path,
        config_hash_value=cfg_hash,
        dataset_catalog_manifest={
            "accession": accession,
            "source": source,
            "artifact_path": path,
            "fetched_at": iso_now(),
        },
        provenance_metadata={
            "fetch": {
                "email": getattr(ncbi_cfg, "email", None),
                "max_retries": int(getattr(ncbi_cfg, "max_retries", 0)),
                "backoff_seconds": float(getattr(ncbi_cfg, "backoff_seconds", 0.0)),
            },
        },
    )


def _build_encoded_manifest_payload(
    *,
    accession: str,
    source: str,
    encoded_path: str,
    encoded_shape: Any,
    cfg_path: str,
    cfg_hash: str,
    tok: str,
    window_size: int,
    stride: int,
    frame: int,
    min_orf: int,
    pol: Dict[str, Any],
    protein_opts: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_kind": "encode_windows",
        "config_path": cfg_path,
        "config_hash_value": cfg_hash,
        "dataset_catalog_manifest": {
            "accession": accession,
            "source": source,
            "artifact_path": encoded_path,
            "encoded_at": iso_now(),
        },
        "tokenizer_encoding_config": {
            "tokenizer": tok,
            "window_size": int(window_size),
            "stride": int(stride),
            "frame_offset": int(frame),
            "min_orf_aa": int(min_orf),
            "max_windows_per_protein": pol.get("max_windows_per_protein"),
            "protein_len_min": pol.get("protein_len_min"),
            "protein_len_max": pol.get("protein_len_max"),
            "translation_only": bool(pol.get("translation_only", False)),
            "protein_opts": protein_opts,
        },
        "metrics": {
            "encoded_shape": list(encoded_shape) if encoded_shape is not None else None,
        },
    }


def _infer_top_level_type(accession: str, source: str) -> str:
    lowered = str(accession).lower()
    if source == "genbank" and ("virus" in lowered or lowered.startswith("nc_00")):
        return "virus"
    if source == "genbank":
        return "plasmid"
    return "genome"


def _build_and_write_bio_ast(accession: str, source: str, io_cfg) -> Optional[str]:
    src = source.lower()
    builder = BioASTBuilder()
    try:
        if src == "genbank":
            gb_path = os.path.join(io_cfg.cache_genbank_dir, f"{accession}.gb")
            sequence = parse_genbank_dna(gb_path)
            cds_features = parse_cds_features_from_genbank(gb_path)
        else:
            fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
            sequence = parse_fasta_sequence(fasta_path)
            cds_features = None

        built = builder.build(
            sequence=sequence,
            cds_features=cds_features,
            top_level_type=_infer_top_level_type(accession, src),
            accession=str(accession),
        )
    except Exception as exc:
        logging.warning("%s: failed to build bio AST (%s)", accession, exc)
        return None

    layout = ensure_run_layout()
    ast_dir = path_in_run(layout, "artifacts", "bio_ast")
    os.makedirs(ast_dir, exist_ok=True)
    out_path = os.path.join(ast_dir, f"{accession}.bio_ast.json")
    payload = {
        "ast": built.ast.to_dict(),
        "serialized_paths": built.to_serialized_paths(),
        "tree_message_passing": {
            key: value.tolist() for key, value in built.to_tree_message_passing_tensors().items()
        },
        "local_windows_shape": list(built.to_local_windows().shape),
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    update_run_manifest(layout, paths={"bio_ast": {str(accession): out_path}})
    return out_path


def _warn_if_encoded_manifest_incompatible(encoded_path: str, expected: Dict[str, Any]) -> None:
    mpath = sidecar_manifest_path(encoded_path)
    if not os.path.exists(mpath):
        logging.warning("Encoded cache manifest missing for %s; consider re-encoding.", encoded_path)
        return
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.warning("Failed reading encoded cache manifest %s: %s", mpath, e)
        return

    got = extract_tokenizer_encoding_config(data)
    mismatches = []
    for k in (
        "tokenizer",
        "window_size",
        "stride",
        "frame_offset",
        "min_orf_aa",
        "max_windows_per_protein",
        "protein_len_min",
        "protein_len_max",
        "translation_only",
    ):
        if got.get(k) != expected.get(k):
            mismatches.append((k, got.get(k), expected.get(k)))
    if mismatches:
        detail = ", ".join(f"{k}: have={a!r} want={b!r}" for (k, a, b) in mismatches)
        logging.warning("Encoded cache manifest mismatch for %s (%s).", encoded_path, detail)


def cmd_split_create(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, _, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    catalog_path = str(args.catalog)
    name = str(args.name)
    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    if train_ratio <= 0.0 or val_ratio < 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("Require train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.0")

    accessions = read_catalog(catalog_path)
    if len(accessions) < 3:
        raise ValueError("Need at least 3 accessions to create train/val/test splits")

    rng = random.Random(int(args.seed))
    shuffled = list(accessions)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        # Always keep at least one test accession.
        n_val = max(1, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)

    train_split = shuffled[:n_train]
    val_split = shuffled[n_train:n_train + n_val]
    test_split = shuffled[n_train + n_val:]

    out_path = str(args.out) if args.out else _default_split_path(io_cfg, name)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "name": name,
        "catalog_path": catalog_path,
        "seed": int(args.seed),
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": 1.0 - train_ratio - val_ratio,
        },
        "counts": {
            "total": n,
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split),
        },
        "splits": {
            "train": train_split,
            "val": val_split,
            "test": test_split,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(
        f"[split-create] name={name} total={n} train={len(train_split)} "
        f"val={len(val_split)} test={len(test_split)} -> {out_path}"
    )
    return 0


def cmd_split_show(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, _, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    split_name = str(args.name)
    split_path = str(args.path) if args.path else _default_split_path(io_cfg, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = data.get("counts", {})
    print(f"Split: {data.get('name', split_name)}")
    print(f"  path: {split_path}")
    print(f"  catalog: {data.get('catalog_path', '<unknown>')}")
    print(
        f"  counts: total={counts.get('total', '?')} "
        f"train={counts.get('train', '?')} val={counts.get('val', '?')} test={counts.get('test', '?')}"
    )
    for key in ("train", "val", "test"):
        vals = list((data.get("splits") or {}).get(key, []))
        head = ", ".join(vals[:5]) if vals else "<empty>"
        suffix = " ..." if len(vals) > 5 else ""
        print(f"  {key}: {head}{suffix}")
    return 0


def cmd_fetch_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    io_cfg = _run_local_io_cfg(io_cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    cfg_hash = config_hash(cfg)

    src = str(getattr(args, "source", None) or "fasta").lower()
    if src == "genbank":
        path = fetch_genbank(args.accession, io_cfg, ncbi_cfg, force=args.force)
    else:
        path = fetch_fasta(args.accession, io_cfg, ncbi_cfg, force=args.force)
    mpath = _write_fetch_manifest(
        path=path,
        accession=str(args.accession),
        source=src,
        cfg_path=str(args.config),
        cfg_hash=cfg_hash,
        ncbi_cfg=ncbi_cfg,
    )
    logging.info("Wrote fetch manifest: %s", mpath)
    return 0


def cmd_tensorboard(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, _, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)

    logdir = getattr(args, "logdir", None) or os.path.join(io_cfg.logs_dir, "tensorboard")
    os.makedirs(logdir, exist_ok=True)

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 6006))
    reload_interval = float(getattr(args, "reload_interval", 5.0))

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        logdir,
        "--host",
        host,
        "--port",
        str(port),
        "--reload_interval",
        str(reload_interval),
    ]

    path_prefix = getattr(args, "path_prefix", None)
    if path_prefix:
        cmd.extend(["--path_prefix", str(path_prefix)])

    print("Launching TensorBoard:")
    print(" ".join(cmd))
    print(f"Open: http://{host}:{port}")

    if getattr(args, "dry_run", False):
        return 0

    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        logging.error("TensorBoard is not installed. Install it with: pip install tensorboard")
        return 2


def cmd_encode_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    io_cfg = _run_local_io_cfg(io_cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    cfg_hash = config_hash(cfg)

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    src = _get_source(args, tok)
    pol = _resolve_proteome_params(args, train_cfg, state=None, tok=tok, src=src)
    protein_opts = pol.get("protein_opts") or {}

    record_path = _ensure_record(args.accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
    _write_fetch_manifest(
        path=record_path,
        accession=str(args.accession),
        source=src,
        cfg_path=str(args.config),
        cfg_hash=cfg_hash,
        ncbi_cfg=ncbi_cfg,
    )

    layout = ensure_run_layout()
    out_path = path_in_run(layout, "artifacts", f"{args.accession}.{tok}.encoded.npy")

    encoded = encode_accession(
        args.accession, io_cfg, window_size, stride,
        tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
        source=src,
        max_windows_per_protein=pol.get("max_windows_per_protein"),
        protein_len_min=pol.get("protein_len_min"),
        protein_len_max=pol.get("protein_len_max"),
        translation_only=bool(pol.get("translation_only", False)),
        protein_opts=protein_opts,
        save_to_disk=True, out_path=out_path,
    )
    payload = _build_encoded_manifest_payload(
        accession=str(args.accession),
        source=src,
        encoded_path=out_path,
        encoded_shape=getattr(encoded, "shape", None),
        cfg_path=str(args.config),
        cfg_hash=cfg_hash,
        tok=tok,
        window_size=window_size,
        stride=stride,
        frame=frame,
        min_orf=min_orf,
        pol=pol,
        protein_opts=protein_opts,
    )
    write_sidecar_run_manifest(target_path=out_path, **payload)
    update_run_manifest(layout, paths={"encoded": {str(args.accession): out_path}})
    ast_path = _build_and_write_bio_ast(args.accession, src, io_cfg)
    if ast_path:
        logging.info("%s: bio AST artifact written at %s", args.accession, ast_path)
    print(f"{args.accession}: encoded tokenizer={tok} source={src} -> shape={encoded.shape} saved={out_path}")
    return 0


def cmd_train_one(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="train_one", config_path=str(args.config), params=vars(args).copy())
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    data = result.data
    print(f"{data.get('accession', args.accession)}: train-one complete last_total={data.get('last_total_loss')}")
    return 0

def cmd_scope_one(args: argparse.Namespace) -> int:
    if curses is None:
        raise RuntimeError("curses not available")
    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    io_cfg = _run_local_io_cfg(io_cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    cfg_hash = config_hash(cfg)

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    src = _get_source(args, tok)
    pol = _resolve_proteome_params(args, train_cfg, state=None, tok=tok, src=src)
    protein_opts = pol.get("protein_opts") or {}

    record_path = _ensure_record(args.accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
    _write_fetch_manifest(
        path=record_path,
        accession=str(args.accession),
        source=src,
        cfg_path=str(args.config),
        cfg_hash=cfg_hash,
        ncbi_cfg=ncbi_cfg,
    )

    # NOTE: encoded_cache_path() must NOT receive grounded protein keys.
    enc_path = encoded_cache_path(
        io_cfg, args.accession, tok, window_size, stride, frame,
        source=src,
        **_cache_kwargs(tok, min_orf, pol),
    )

    if os.path.exists(enc_path) and not args.reencode:
        encoded = np.load(enc_path)
        _warn_if_encoded_manifest_incompatible(
            enc_path,
            {
                "tokenizer": tok,
                "window_size": int(window_size),
                "stride": int(stride),
                "frame_offset": int(frame),
                "min_orf_aa": int(min_orf),
                "max_windows_per_protein": pol.get("max_windows_per_protein"),
                "protein_len_min": pol.get("protein_len_min"),
                "protein_len_max": pol.get("protein_len_max"),
                "translation_only": bool(pol.get("translation_only", False)),
            },
        )
    else:
        encoded = encode_accession(
            args.accession, io_cfg, window_size, stride,
            tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
            source=src,
            max_windows_per_protein=pol.get("max_windows_per_protein"),
            protein_len_min=pol.get("protein_len_min"),
            protein_len_max=pol.get("protein_len_max"),
            translation_only=bool(pol.get("translation_only", False)),
            protein_opts=protein_opts,
            save_to_disk=True, out_path=enc_path,
        )
        write_sidecar_run_manifest(
            target_path=enc_path,
            **_build_encoded_manifest_payload(
                accession=str(args.accession),
                source=src,
                encoded_path=enc_path,
                encoded_shape=getattr(encoded, "shape", None),
                cfg_path=str(args.config),
                cfg_hash=cfg_hash,
                tok=tok,
                window_size=window_size,
                stride=stride,
                frame=frame,
                min_orf=min_orf,
                pol=pol,
                protein_opts=protein_opts,
            ),
        )

    errors = compute_window_errors(
        args.accession,
        encoded,
        io_cfg=io_cfg,
        train_cfg=train_cfg,
        tokenizer=tok,
        window_size_bp=window_size,
        loss_type=getattr(args, "loss_type", None),
    )
    metric = compute_gc_from_encoded(encoded, tokenizer=tok)

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
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    io_cfg = _run_local_io_cfg(io_cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    cfg_hash = config_hash(cfg)

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    src = _get_source(args, tok)
    pol = _resolve_proteome_params(args, train_cfg, state=None, tok=tok, src=src)
    protein_opts = pol.get("protein_opts") or {}

    record_path = _ensure_record(args.accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
    _write_fetch_manifest(
        path=record_path,
        accession=str(args.accession),
        source=src,
        cfg_path=str(args.config),
        cfg_hash=cfg_hash,
        ncbi_cfg=ncbi_cfg,
    )

    steps = args.steps or train_cfg.steps_per_plasmid
    batch_size = args.batch_size or train_cfg.batch_size

    enc_path = encoded_cache_path(
        io_cfg, args.accession, tok, window_size, stride, frame,
        source=src,
        **_cache_kwargs(tok, min_orf, pol),
    )

    if os.path.exists(enc_path) and not args.reencode:
        encoded = np.load(enc_path)
        _warn_if_encoded_manifest_incompatible(
            enc_path,
            {
                "tokenizer": tok,
                "window_size": int(window_size),
                "stride": int(stride),
                "frame_offset": int(frame),
                "min_orf_aa": int(min_orf),
                "max_windows_per_protein": pol.get("max_windows_per_protein"),
                "protein_len_min": pol.get("protein_len_min"),
                "protein_len_max": pol.get("protein_len_max"),
                "translation_only": bool(pol.get("translation_only", False)),
            },
        )
    else:
        encoded = encode_accession(
            args.accession, io_cfg, window_size, stride,
            tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
            source=src,
            max_windows_per_protein=pol.get("max_windows_per_protein"),
            protein_len_min=pol.get("protein_len_min"),
            protein_len_max=pol.get("protein_len_max"),
            translation_only=bool(pol.get("translation_only", False)),
            protein_opts=protein_opts,
            save_to_disk=True, out_path=enc_path,
        )
        write_sidecar_run_manifest(
            target_path=enc_path,
            **_build_encoded_manifest_payload(
                accession=str(args.accession),
                source=src,
                encoded_path=enc_path,
                encoded_shape=getattr(encoded, "shape", None),
                cfg_path=str(args.config),
                cfg_hash=cfg_hash,
                tok=tok,
                window_size=window_size,
                stride=stride,
                frame=frame,
                min_orf=min_orf,
                pol=pol,
                protein_opts=protein_opts,
            ),
        )

    metric = compute_gc_from_encoded(encoded, tokenizer=tok)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from ..model import get_device, load_or_init_model
    from ..encoding_main import tokenizer_meta

    device = get_device()
    seq_len, vocab_size = tokenizer_meta(tok, window_size)
    hidden_dim = train_cfg.hidden_dim
    model_type = train_cfg.model_type
    transformer_d_model = train_cfg.transformer_d_model
    transformer_nhead = train_cfg.transformer_nhead
    transformer_layers = train_cfg.transformer_layers
    transformer_dropout = train_cfg.transformer_dropout

    lt = (args.loss_type if getattr(args, "loss_type", None) is not None else ("ce" if tok == "aa" else "mse"))

    model, optimizer, global_step, ckpt_path = load_or_init_model(
        io_cfg=io_cfg, seq_len=seq_len, vocab_size=vocab_size,
        hidden_dim=hidden_dim, learning_rate=train_cfg.learning_rate,
        device=device, tokenizer=tok, loss_type=lt,
        model_type=model_type,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        beta_kl=train_cfg.beta_kl,
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
        loss_type=lt, seq_len=int(seq_len), vocab_size=int(vocab_size),
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
    spec = JobSpec(kind="stream", config_path=str(args.config), params=vars(args).copy())
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    print("[stream] Training complete.")
    return 0

def cmd_generate_plasmid(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="generate_plasmid", config_path=str(args.config), params=vars(args).copy())
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    print(f"[generate-plasmid] wrote {result.data.get('length')} bp -> {result.data.get('output', args.output)}")
    return 0

def cmd_validate_plasmid(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="validate_plasmid", config_path=str(args.config), params=vars(args).copy())
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    top_rows = result.data.get("results", [])
    print(f"[validate-plasmid] refs={len(top_rows)}")
    for i, row in enumerate(top_rows, start=1):
        print(f"{i:>4d} {row['accession']:<16s} {row['score']:.4f} {row['ref_len']}")
    return 0

def cmd_generate_protein(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="generate_protein", config_path=str(args.config), params=vars(args).copy())
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    print(f"[generate-protein] wrote {result.data.get('length')} aa -> {result.data.get('output', args.output)}")
    return 0

def cmd_pretrain(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="pretrain", config_path=str(args.config), params=vars(args).copy())
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    print("[pretrain] complete")
    metrics = result.data.get("metrics", {})
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {float(v):.6f}")
    return 0



def cmd_design_loop(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="design_loop", config_path=str(args.config), params=vars(args).copy())
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    best = result.data.get("best_candidate", {})
    print(f"[design-loop] rounds={result.data.get('rounds_completed')} best_score={float(best.get('score', 0.0)):.6f}")
    print(f"[design-loop] best fasta -> {result.data.get('best_fasta')}")
    return 0


def cmd_run_job_spec(args: argparse.Namespace) -> int:
    spec_payload = json.loads(str(args.spec_json))
    spec = JobSpec(
        kind=str(spec_payload["kind"]),
        config_path=str(spec_payload.get("config_path", args.config)),
        params=dict(spec_payload.get("params", {})),
    )
    engine = JobEngine(event_sink=lambda ev: print(json.dumps({"event": ev.stage, "message": ev.message, "data": ev.data})))
    result = engine.run(spec)
    print(json.dumps({"ok": result.ok, "exit_code": result.exit_code, "message": result.message, "data": result.data}))
    return 0 if result.ok else 1
