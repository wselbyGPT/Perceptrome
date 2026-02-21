import argparse
import datetime
import hashlib
import json
import logging
import os
import random
import subprocess
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


def _default_split_path(io_cfg, split_name: str) -> str:
    state_dir = os.path.dirname(io_cfg.state_file) or "state"
    return os.path.join(state_dir, "splits", f"{split_name}.json")


def _get_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        sha = out.decode("utf-8").strip()
        return sha or None
    except Exception:
        return None


def _config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _new_experiment_id(prefix: str = "exp") -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def _write_experiment_manifest(io_cfg, experiment_id: str, payload: Dict[str, Any]) -> str:
    exp_dir = os.path.join(io_cfg.logs_dir, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    out_path = os.path.join(exp_dir, f"{experiment_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def _iso_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sidecar_manifest_path(path: str) -> str:
    return f"{path}.manifest.json"


def _write_sidecar_manifest(target_path: str, payload: Dict[str, Any]) -> str:
    out_path = _sidecar_manifest_path(target_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def _write_fetch_manifest(
    *,
    path: str,
    accession: str,
    source: str,
    cfg_path: str,
    cfg_hash: str,
    ncbi_cfg,
) -> str:
    payload = {
        "schema_version": 1,
        "artifact_type": "fetched_record",
        "accession": accession,
        "source": source,
        "fetched_at": _iso_now(),
        "artifact_path": path,
        "software": {"git_sha": _get_git_sha()},
        "config": {"path": cfg_path, "sha256": cfg_hash},
        "fetch": {
            "email": getattr(ncbi_cfg, "email", None),
            "max_retries": int(getattr(ncbi_cfg, "max_retries", 0)),
            "backoff_seconds": float(getattr(ncbi_cfg, "backoff_seconds", 0.0)),
        },
    }
    return _write_sidecar_manifest(path, payload)


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
        "schema_version": 1,
        "artifact_type": "encoded_windows",
        "accession": accession,
        "source": source,
        "encoded_at": _iso_now(),
        "artifact_path": encoded_path,
        "shape": list(encoded_shape) if encoded_shape is not None else None,
        "software": {"git_sha": _get_git_sha()},
        "config": {"path": cfg_path, "sha256": cfg_hash},
        "encoding": {
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
    }


def _warn_if_encoded_manifest_incompatible(encoded_path: str, expected: Dict[str, Any]) -> None:
    mpath = _sidecar_manifest_path(encoded_path)
    if not os.path.exists(mpath):
        logging.warning("Encoded cache manifest missing for %s; consider re-encoding.", encoded_path)
        return
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.warning("Failed reading encoded cache manifest %s: %s", mpath, e)
        return

    got = data.get("encoding", {}) if isinstance(data, dict) else {}
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
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    cfg_hash = _config_hash(cfg)

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


def cmd_encode_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    cfg_hash = _config_hash(cfg)

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

    out_path = encoded_cache_path(
        io_cfg, args.accession, tok, window_size, stride, frame,
        source=src,
        **_cache_kwargs(tok, min_orf, pol),
    )

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
    _write_sidecar_manifest(out_path, payload)
    print(f"{args.accession}: encoded tokenizer={tok} source={src} -> shape={encoded.shape} saved={out_path}")
    return 0


def cmd_train_one(args: argparse.Namespace) -> int:
    run_started = _iso_now()
    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    state = load_state(io_cfg.state_file)
    cfg_hash = _config_hash(cfg)

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    src = _get_source(args, tok)
    pol = _resolve_proteome_params(args, train_cfg, state=state, tok=tok, src=src)
    protein_opts = pol.get("protein_opts") or {}

    batch_size = args.batch_size or train_cfg.batch_size
    steps = args.steps or train_cfg.steps_per_plasmid
    loss_type = getattr(args, "loss_type", None)

    experiment_id = str(getattr(args, "experiment_id", None) or _new_experiment_id("train"))

    record_path = _ensure_record(args.accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
    _write_fetch_manifest(
        path=record_path,
        accession=str(args.accession),
        source=src,
        cfg_path=str(args.config),
        cfg_hash=cfg_hash,
        ncbi_cfg=ncbi_cfg,
    )

    enc_path = encoded_cache_path(
        io_cfg, args.accession, tok, window_size, stride, frame,
        source=src,
        **_cache_kwargs(tok, min_orf, pol),
    )

    if os.path.exists(enc_path) and not getattr(args, "reencode", False):
        encoded = np.load(enc_path)
        logging.info(f"{args.accession}: using cached encoded at {enc_path} shape={encoded.shape}")
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
        _write_sidecar_manifest(
            enc_path,
            _build_encoded_manifest_payload(
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

    last_total = train_on_encoded(
        args.accession, encoded,
        steps=steps, batch_size=batch_size,
        state=state, io_cfg=io_cfg, train_cfg=train_cfg,
        tokenizer=tok, window_size_bp=window_size,
        loss_type=loss_type,
        mask_prob=pol.get("mask_prob"),
        span_mask_prob=pol.get("span_mask_prob"),
        span_mask_len=pol.get("span_mask_len"),
        run_id=getattr(args, "tb_run_id", None),
        tensorboard_log_every=getattr(args, "tb_log_every", None),
    )

    pvc = state["plasmid_visit_counts"]
    pvc[args.accession] = pvc.get(args.accession, 0) + 1
    save_state(io_cfg.state_file, state)

    cleanup_accession_files(args.accession, io_cfg, enc_path)

    run_completed = _iso_now()
    effective_loss_type = str(loss_type).lower() if loss_type is not None else ("ce" if tok == "aa" else "mse")
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "command": "train-one",
        "started_at": run_started,
        "completed_at": run_completed,
        "git_sha": _get_git_sha(),
        "config": {
            "path": str(args.config),
            "sha256": cfg_hash,
        },
        "inputs": {
            "accession": str(args.accession),
            "source": src,
            "tokenizer": tok,
            "window_size": int(window_size),
            "stride": int(stride),
            "frame_offset": int(frame),
            "min_orf_aa": int(min_orf),
            "encoded_path": enc_path,
        },
        "training": {
            "steps": int(steps),
            "batch_size": int(batch_size),
            "loss_type": effective_loss_type,
            "mask_prob": float(pol.get("mask_prob", 0.0)),
            "span_mask_prob": float(pol.get("span_mask_prob", 0.0)),
            "span_mask_len": int(pol.get("span_mask_len", 0)),
            "model_type": str(train_cfg.model_type),
            "hidden_dim": int(train_cfg.hidden_dim),
            "learning_rate": float(train_cfg.learning_rate),
        },
        "metrics": {
            "last_total_loss": float(last_total),
            "total_steps": int(state.get("total_steps", 0)),
        },
        "artifacts": {
            "checkpoint": os.path.join(io_cfg.checkpoints_dir, "latest.pt"),
            "state_file": io_cfg.state_file,
        },
        "cli_args": vars(args),
    }
    manifest_path = _write_experiment_manifest(io_cfg, experiment_id, manifest)

    print(f"{args.accession}: train-one tokenizer={tok} source={src} steps={steps} batch={batch_size} last_total={last_total:.6f}")
    print(f"[experiment] id={experiment_id} manifest={manifest_path}")
    return 0


def cmd_scope_one(args: argparse.Namespace) -> int:
    if curses is None:
        raise RuntimeError("curses not available")
    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    cfg_hash = _config_hash(cfg)

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
        _write_sidecar_manifest(
            enc_path,
            _build_encoded_manifest_payload(
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
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    cfg_hash = _config_hash(cfg)

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
        _write_sidecar_manifest(
            enc_path,
            _build_encoded_manifest_payload(
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
    run_started = _iso_now()
    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    cfg_hash = _config_hash(cfg)

    accessions = read_catalog(args.catalog)
    state = load_state(io_cfg.state_file)

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    src = _get_source(args, tok)

    batch_size = args.batch_size or train_cfg.batch_size
    steps_per_plasmid = args.steps_per_plasmid or train_cfg.steps_per_plasmid
    max_epochs = args.max_epochs or train_cfg.max_stream_epochs
    loss_type = getattr(args, "loss_type", None)
    experiment_id = str(getattr(args, "experiment_id", None) or _new_experiment_id("stream"))
    run_records = []
    last_total = None

    epoch = int(state.get("epoch", 0))

    while epoch < max_epochs:
        indices = list(range(len(accessions)))
        if train_cfg.shuffle_catalog:
            random.shuffle(indices)

        for idx in indices:
            acc = accessions[idx]
            pol = _resolve_proteome_params(args, train_cfg, state=state, tok=tok, src=src)
            protein_opts = pol.get("protein_opts") or {}

            record_path = _ensure_record(acc, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
            _write_fetch_manifest(
                path=record_path,
                accession=str(acc),
                source=src,
                cfg_path=str(args.config),
                cfg_hash=cfg_hash,
                ncbi_cfg=ncbi_cfg,
            )

            enc_path = encoded_cache_path(
                io_cfg, acc, tok, window_size, stride, frame,
                source=src,
                **_cache_kwargs(tok, min_orf, pol),
            )

            if os.path.exists(enc_path) and not getattr(args, "reencode", False):
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
                    acc, io_cfg, window_size, stride,
                    tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
                    source=src,
                    max_windows_per_protein=pol.get("max_windows_per_protein"),
                    protein_len_min=pol.get("protein_len_min"),
                    protein_len_max=pol.get("protein_len_max"),
                    translation_only=bool(pol.get("translation_only", False)),
                    protein_opts=protein_opts,
                    save_to_disk=True, out_path=enc_path,
                )
                _write_sidecar_manifest(
                    enc_path,
                    _build_encoded_manifest_payload(
                        accession=str(acc),
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

            last_total = train_on_encoded(
                acc, encoded,
                steps=steps_per_plasmid, batch_size=batch_size,
                state=state, io_cfg=io_cfg, train_cfg=train_cfg,
                tokenizer=tok, window_size_bp=window_size,
                loss_type=loss_type,
                mask_prob=pol.get("mask_prob"),
                span_mask_prob=pol.get("span_mask_prob"),
                span_mask_len=pol.get("span_mask_len"),
                run_id=getattr(args, "tb_run_id", None),
                tensorboard_log_every=getattr(args, "tb_log_every", None),
            )

            pvc = state["plasmid_visit_counts"]
            pvc[acc] = pvc.get(acc, 0) + 1
            state["current_index"] = idx
            state["epoch"] = epoch
            save_state(io_cfg.state_file, state)

            if getattr(args, "delete_cache", False):
                cleanup_accession_files(acc, io_cfg, enc_path)

            run_records.append(
                {
                    "accession": acc,
                    "epoch": int(epoch),
                    "encoded_path": enc_path,
                    "steps": int(steps_per_plasmid),
                    "last_total_loss": float(last_total),
                }
            )

        epoch += 1

    run_completed = _iso_now()
    effective_loss_type = str(loss_type).lower() if loss_type is not None else ("ce" if tok == "aa" else "mse")
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "command": "stream",
        "started_at": run_started,
        "completed_at": run_completed,
        "git_sha": _get_git_sha(),
        "config": {
            "path": str(args.config),
            "sha256": cfg_hash,
        },
        "inputs": {
            "catalog": str(args.catalog),
            "accession_count": len(accessions),
            "source": src,
            "tokenizer": tok,
            "window_size": int(window_size),
            "stride": int(stride),
            "frame_offset": int(frame),
            "min_orf_aa": int(min_orf),
        },
        "training": {
            "steps_per_plasmid": int(steps_per_plasmid),
            "batch_size": int(batch_size),
            "max_epochs": int(max_epochs),
            "loss_type": effective_loss_type,
            "model_type": str(train_cfg.model_type),
            "hidden_dim": int(train_cfg.hidden_dim),
            "learning_rate": float(train_cfg.learning_rate),
        },
        "metrics": {
            "total_steps": int(state.get("total_steps", 0)),
            "processed_accessions": len(run_records),
            "last_total_loss": (None if last_total is None else float(last_total)),
        },
        "artifacts": {
            "checkpoint": os.path.join(io_cfg.checkpoints_dir, "latest.pt"),
            "state_file": io_cfg.state_file,
        },
        "records": run_records,
        "cli_args": vars(args),
    }
    manifest_path = _write_experiment_manifest(io_cfg, experiment_id, manifest)

    print("[stream] Training complete.")
    print(f"[experiment] id={experiment_id} manifest={manifest_path}")
    return 0


def cmd_generate_plasmid(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    _, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

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
        num_candidates=int(getattr(args, "num_candidates", 1)),
        top_k=int(getattr(args, "top_k", 1)),
        target_gc=float(getattr(args, "target_gc", 0.5)),
        max_homopolymer=getattr(args, "max_homopolymer", None),
        summary_path=getattr(args, "summary_path", None),
        roundtrip_score=bool(getattr(args, "roundtrip_score", False)),
        name=args.name,
        output_path=args.output,
        tokenizer=tok,
    )
    print(f"[generate-plasmid] tokenizer={tok} wrote {len(seq)} bp -> {args.output}")
    return 0


def cmd_generate_protein(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    _, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

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
        reject=bool(getattr(args, "reject", False)),
        reject_tries=int(getattr(args, "reject_tries", 40)),
        reject_max_run=int(getattr(args, "reject_max_run", 10)),
        reject_max_x_frac=float(getattr(args, "reject_max_x_frac", 0.15)),
        num_candidates=int(getattr(args, "num_candidates", 1)),
        top_k=int(getattr(args, "top_k", 1)),
        max_homopolymer=getattr(args, "max_homopolymer", None),
        max_x_frac=getattr(args, "max_x_frac", None),
        max_internal_stops=int(getattr(args, "max_internal_stops", 0)),
        summary_path=getattr(args, "summary_path", None),
        roundtrip_score=bool(getattr(args, "roundtrip_score", False)),
    )
    print(f"[generate-protein] wrote {len(seq)} aa -> {args.output}")
    return 0
