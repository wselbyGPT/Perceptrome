import argparse
import json
import logging
import os
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from perceptrome.cli.common import (
    extract_configs, load_full_config,
    compute_gc_from_encoded, encode_accession,
    generate_plasmid_sequence, generate_protein_sequence,
    ensure_dirs, load_state, read_catalog, save_state, setup_logging, encoded_cache_path,
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
RUN_MANIFEST_SCHEMA_VERSION = "1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def runs_dir(base_dir: Optional[str] = None) -> Path:
    root = Path(base_dir or ".").resolve()
    return root / "state" / "runs"


def _run_manifest_path(run_id: str, base_dir: Optional[str] = None) -> Path:
    return runs_dir(base_dir) / f"{run_id}.json"


def _safe_split(command: str) -> list[str]:
    try:
        return shlex.split(command) if command else []
    except ValueError:
        return []


def _extract_output_paths(command: str) -> Dict[str, str]:
    output_paths: Dict[str, str] = {}
    tokens = _safe_split(command)

    capture_flags = {
        "--output": "output",
        "--out": "output",
        "-o": "output",
        "--log": "log",
        "--log-file": "log",
        "--state-file": "state_file",
        "--checkpoint": "checkpoint",
        "--save-dir": "save_dir",
    }
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if "=" in tok and tok.startswith("--"):
            key, value = tok.split("=", 1)
            if key in capture_flags and value:
                output_paths[capture_flags[key]] = value
        elif tok in capture_flags and i + 1 < len(tokens):
            output_paths[capture_flags[tok]] = tokens[i + 1]
            i += 1
        i += 1
    return output_paths


def create_run_manifest(
    command: str,
    *,
    base_dir: Optional[str] = None,
    source: str = "cli",
    status: str = "starting",
    workdir: Optional[str] = None,
    output_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    runs = runs_dir(base_dir)
    runs.mkdir(parents=True, exist_ok=True)

    now = _now_iso()
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "source": source,
        "status": status,
        "exit_code": None,
        "error": None,
        "workdir": workdir or str(Path(base_dir or ".").resolve()),
        "command": {
            "raw": command,
            "args": _safe_split(command),
        },
        "timestamps": {
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "updated_at": now,
        },
        "output_paths": {
            **_extract_output_paths(command),
            **(output_paths or {}),
        },
    }
    _run_manifest_path(run_id, base_dir).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def update_run_manifest(
    run_id: str,
    *,
    base_dir: Optional[str] = None,
    status: Optional[str] = None,
    exit_code: Optional[int] = None,
    error: Optional[str] = None,
    started: bool = False,
    finished: bool = False,
    output_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    path = _run_manifest_path(run_id, base_dir)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    now = _now_iso()

    if status is not None:
        manifest["status"] = status
    if exit_code is not None:
        manifest["exit_code"] = int(exit_code)
    if error is not None:
        manifest["error"] = error
    if started:
        manifest["timestamps"]["started_at"] = now
    if finished:
        manifest["timestamps"]["finished_at"] = now
    if output_paths:
        manifest.setdefault("output_paths", {}).update(output_paths)
    manifest["timestamps"]["updated_at"] = now

    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_run_manifest(run_id: str, *, base_dir: Optional[str] = None) -> Dict[str, Any]:
    return json.loads(_run_manifest_path(run_id, base_dir).read_text(encoding="utf-8"))


def list_run_manifests(*, base_dir: Optional[str] = None) -> list[Dict[str, Any]]:
    runs = runs_dir(base_dir)
    if not runs.exists():
        return []
    manifests: list[Dict[str, Any]] = []
    for path in sorted(runs.glob("*.json"), reverse=True):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return manifests


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


# -----------------------------
# Commands
# -----------------------------
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
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    src = str(getattr(args, "source", None) or "fasta").lower()
    if src == "genbank":
        fetch_genbank(args.accession, io_cfg, ncbi_cfg, force=args.force)
    else:
        fetch_fasta(args.accession, io_cfg, ncbi_cfg, force=args.force)
    return 0


def cmd_encode_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    src = _get_source(args, tok)
    pol = _resolve_proteome_params(args, train_cfg, state=None, tok=tok, src=src)
    protein_opts = pol.get("protein_opts") or {}

    _ensure_record(args.accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)

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
    print(f"{args.accession}: encoded tokenizer={tok} source={src} -> shape={encoded.shape} saved={out_path}")
    return 0


def cmd_train_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    state = load_state(io_cfg.state_file)

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

    _ensure_record(args.accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)

    enc_path = encoded_cache_path(
        io_cfg, args.accession, tok, window_size, stride, frame,
        source=src,
        **_cache_kwargs(tok, min_orf, pol),
    )

    if os.path.exists(enc_path) and not getattr(args, "reencode", False):
        encoded = np.load(enc_path)
        logging.info(f"{args.accession}: using cached encoded at {enc_path} shape={encoded.shape}")
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

    last_total = train_on_encoded(
        args.accession, encoded,
        steps=steps, batch_size=batch_size,
        state=state, io_cfg=io_cfg, train_cfg=train_cfg,
        tokenizer=tok, window_size_bp=window_size,
        loss_type=getattr(args, "loss_type", None),
        mask_prob=pol.get("mask_prob"),
        span_mask_prob=pol.get("span_mask_prob"),
        span_mask_len=pol.get("span_mask_len"),
    )

    pvc = state["plasmid_visit_counts"]
    pvc[args.accession] = pvc.get(args.accession, 0) + 1
    save_state(io_cfg.state_file, state)

    print(f"{args.accession}: train-one tokenizer={tok} source={src} steps={steps} batch={batch_size} last_total={last_total:.6f}")
    return 0


def cmd_scope_one(args: argparse.Namespace) -> int:
    if curses is None:
        raise RuntimeError("curses not available")
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    src = _get_source(args, tok)
    pol = _resolve_proteome_params(args, train_cfg, state=None, tok=tok, src=src)
    protein_opts = pol.get("protein_opts") or {}

    _ensure_record(args.accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)

    # NOTE: encoded_cache_path() must NOT receive grounded protein keys.
    enc_path = encoded_cache_path(
        io_cfg, args.accession, tok, window_size, stride, frame,
        source=src,
        **_cache_kwargs(tok, min_orf, pol),
    )

    if os.path.exists(enc_path) and not args.reencode:
        encoded = np.load(enc_path)
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
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    tok = _get_tok(args, train_cfg)
    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)

    src = _get_source(args, tok)
    pol = _resolve_proteome_params(args, train_cfg, state=None, tok=tok, src=src)
    protein_opts = pol.get("protein_opts") or {}

    _ensure_record(args.accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)

    steps = args.steps or train_cfg.steps_per_plasmid
    batch_size = args.batch_size or train_cfg.batch_size

    enc_path = encoded_cache_path(
        io_cfg, args.accession, tok, window_size, stride, frame,
        source=src,
        **_cache_kwargs(tok, min_orf, pol),
    )

    if os.path.exists(enc_path) and not args.reencode:
        encoded = np.load(enc_path)
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
    import random

    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

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

    epoch = int(state.get("epoch", 0))

    while epoch < max_epochs:
        indices = list(range(len(accessions)))
        if train_cfg.shuffle_catalog:
            random.shuffle(indices)

        for idx in indices:
            acc = accessions[idx]
            pol = _resolve_proteome_params(args, train_cfg, state=state, tok=tok, src=src)
            protein_opts = pol.get("protein_opts") or {}

            _ensure_record(acc, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)

            enc_path = encoded_cache_path(
                io_cfg, acc, tok, window_size, stride, frame,
                source=src,
                **_cache_kwargs(tok, min_orf, pol),
            )

            if os.path.exists(enc_path) and not getattr(args, "reencode", False):
                encoded = np.load(enc_path)
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

            _ = train_on_encoded(
                acc, encoded,
                steps=steps_per_plasmid, batch_size=batch_size,
                state=state, io_cfg=io_cfg, train_cfg=train_cfg,
                tokenizer=tok, window_size_bp=window_size,
                loss_type=getattr(args, "loss_type", None),
                mask_prob=pol.get("mask_prob"),
                span_mask_prob=pol.get("span_mask_prob"),
                span_mask_len=pol.get("span_mask_len"),
            )

            pvc = state["plasmid_visit_counts"]
            pvc[acc] = pvc.get(acc, 0) + 1
            state["current_index"] = idx
            state["epoch"] = epoch
            save_state(io_cfg.state_file, state)

            if getattr(args, "delete_cache", False):
                cleanup_accession_files(acc, io_cfg, enc_path)

        epoch += 1

    print("[stream] Training complete.")
    return 0


def cmd_generate_plasmid(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
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
        name=args.name,
        output_path=args.output,
        tokenizer=tok,
    )
    print(f"[generate-plasmid] tokenizer={tok} wrote {len(seq)} bp -> {args.output}")
    return 0


def cmd_generate_protein(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
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
    )
    print(f"[generate-protein] wrote {len(seq)} aa -> {args.output}")
    return 0


def cmd_runs_list(args: argparse.Namespace) -> int:
    records = list_run_manifests(base_dir=".")
    if not records:
        print("No run manifests found under state/runs.")
        return 0

    print(f"Found {len(records)} run(s):")
    for item in records:
        run_id = item.get("run_id", "?")
        status = item.get("status", "?")
        exit_code = item.get("exit_code")
        started_at = item.get("timestamps", {}).get("started_at") or "-"
        cmd = item.get("command", {}).get("raw", "")
        print(f"- {run_id}  status={status}  exit={exit_code}  started={started_at}")
        print(f"  cmd: {cmd}")
    return 0


def cmd_runs_show(args: argparse.Namespace) -> int:
    try:
        item = load_run_manifest(args.run_id, base_dir=".")
    except FileNotFoundError:
        print(f"Run manifest not found: {args.run_id}")
        return 1
    print(json.dumps(item, indent=2))
    return 0
