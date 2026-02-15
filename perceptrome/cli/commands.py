import argparse
import hashlib
import json
import logging
import os
import random
from datetime import datetime, timezone
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

from perceptrome.encoding.constants import GC_FRAC_PER_TOKEN


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


def _default_loss_type(tokenizer: str) -> str:
    return "ce" if str(tokenizer).lower() == "aa" else "mse"


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _gc_from_probs(probs: "Any", tok: str):
    import torch
    if tok == "base":
        gc = probs[..., 1] + probs[..., 2]
        return gc.mean(dim=1)
    if tok == "codon":
        weights = torch.tensor(GC_FRAC_PER_TOKEN, device=probs.device, dtype=probs.dtype)
        gc = (probs * weights.view(1, 1, -1)).sum(dim=-1)
        return gc.mean(dim=1)
    raise ValueError("GC heuristic only defined for base/codon")


def _evaluate_encoded(
    encoded: np.ndarray,
    *,
    io_cfg,
    train_cfg,
    tokenizer: str,
    window_size: int,
    loss_type: Optional[str],
    topk: int,
) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise RuntimeError("PyTorch not installed.") from e

    from perceptrome.encoding_main import tokenizer_meta
    from perceptrome.model import get_device, load_or_init_model, vae_loss

    tok = str(tokenizer).lower()
    lt = _default_loss_type(tok) if loss_type is None else str(loss_type).lower()
    seq_len, vocab_size = tokenizer_meta(tok, window_size)
    device = get_device()

    model, _, global_step, ckpt_path = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=train_cfg.hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tok,
        loss_type=lt,
        model_type=train_cfg.model_type,
        transformer_d_model=train_cfg.transformer_d_model,
        transformer_nhead=train_cfg.transformer_nhead,
        transformer_layers=train_cfg.transformer_layers,
        transformer_dropout=train_cfg.transformer_dropout,
    )

    model.eval()
    with torch.no_grad():
        wt = torch.from_numpy(encoded).to(device)
        if wt.size(0) == 0:
            raise ValueError("Encoded tensor contains zero windows; cannot evaluate.")
        x_flat = wt.view(wt.size(0), -1)
        recon_logits, mu, logvar = model(x_flat)
        total, recon, kl = vae_loss(
            recon_logits,
            x_flat,
            mu,
            logvar,
            train_cfg.beta_kl,
            lt,
            int(seq_len),
            int(vocab_size),
        )

        logits3 = recon_logits.view(wt.size(0), int(seq_len), int(vocab_size))
        probs = F.softmax(logits3, dim=-1) if lt == "ce" else torch.sigmoid(logits3)
        if lt != "ce":
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        target_idx = wt.argmax(dim=-1)
        pred_idx = probs.argmax(dim=-1)
        token_acc = (pred_idx == target_idx).float().mean().item()

        k = max(1, min(int(topk), int(vocab_size)))
        topk_idx = probs.topk(k=k, dim=-1).indices
        topk_acc = (topk_idx == target_idx.unsqueeze(-1)).any(dim=-1).float().mean().item()

        ce_mean = F.cross_entropy(logits3.view(-1, int(vocab_size)), target_idx.view(-1), reduction="mean")
        perplexity = float(torch.exp(ce_mean).item()) if lt == "ce" else None

        heuristics: Dict[str, Any]
        if tok in ("base", "codon"):
            gc_true = torch.from_numpy(compute_gc_from_encoded(encoded, tokenizer=tok)).to(device)
            gc_pred = _gc_from_probs(probs, tok)
            heuristics = {
                "type": "gc_distribution_drift",
                "gc_mean_true": float(gc_true.mean().item()),
                "gc_mean_pred": float(gc_pred.mean().item()),
                "gc_mean_abs_drift": float((gc_true - gc_pred).abs().mean().item()),
                "gc_std_true": float(gc_true.std(unbiased=False).item()),
                "gc_std_pred": float(gc_pred.std(unbiased=False).item()),
            }
        else:
            # AA vocab has no explicit stop token. We still expose stop-rate fields for comparability.
            has_stop_token = False
            stop_rate = 0.0
            internal_stop_rate = 0.0
            heuristics = {
                "type": "aa_stop_rate",
                "supports_stop_token": has_stop_token,
                "stop_rate": float(stop_rate),
                "internal_stop_rate": float(internal_stop_rate),
                "unknown_token_rate": float((pred_idx == (int(vocab_size) - 1)).float().mean().item()),
            }

    return {
        "num_windows": int(encoded.shape[0]),
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
        "checkpoint_path": ckpt_path,
        "global_step": int(global_step),
        "loss_type": lt,
        "metrics": {
            "total_loss": float(total.item()),
            "reconstruction_loss": float(recon.item()),
            "kl_term": float(kl.item()),
            "perplexity": perplexity,
            "token_accuracy": float(token_acc),
            "topk_accuracy": float(topk_acc),
            "topk": int(k),
        },
        "heuristics": heuristics,
    }


def _write_eval_report(io_cfg, payload: Dict[str, Any], prefix: str) -> str:
    os.makedirs(io_cfg.logs_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(io_cfg.logs_dir, f"{prefix}.{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out_path


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


def _eval_single_accession(args: argparse.Namespace, accession: str, tok: str) -> Dict[str, Any]:
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    frame = _get_frame(args, train_cfg)
    min_orf = _get_min_orf(args, train_cfg)
    window_size, stride = _pick_window_stride(args, train_cfg, tok)
    _validate_tok_params(tok, window_size, stride, frame)
    src = _get_source(args, tok)
    pol = _resolve_proteome_params(args, train_cfg, state=None, tok=tok, src=src)
    protein_opts = pol.get("protein_opts") or {}

    _set_deterministic_seed(int(getattr(args, "seed", 1337)))
    _ensure_record(accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)

    enc_path = encoded_cache_path(
        io_cfg,
        accession,
        tok,
        window_size,
        stride,
        frame,
        source=src,
        **_cache_kwargs(tok, min_orf, pol),
    )
    if os.path.exists(enc_path) and not bool(getattr(args, "reencode", False)):
        encoded = np.load(enc_path)
    else:
        encoded = encode_accession(
            accession,
            io_cfg,
            window_size,
            stride,
            tokenizer=tok,
            frame_offset=frame,
            min_orf_aa=min_orf,
            source=src,
            max_windows_per_protein=pol.get("max_windows_per_protein"),
            protein_len_min=pol.get("protein_len_min"),
            protein_len_max=pol.get("protein_len_max"),
            translation_only=bool(pol.get("translation_only", False)),
            protein_opts=protein_opts,
            save_to_disk=True,
            out_path=enc_path,
            rng_seed=int(getattr(args, "seed", 1337)),
        )

    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    eval_data = _evaluate_encoded(
        encoded,
        io_cfg=io_cfg,
        train_cfg=train_cfg,
        tokenizer=tok,
        window_size=window_size,
        loss_type=getattr(args, "loss_type", None),
        topk=int(getattr(args, "topk", 5)),
    )
    return {
        "metadata": {
            "accession": accession,
            "tokenizer": tok,
            "source": src,
            "seed": int(getattr(args, "seed", 1337)),
            "config_path": args.config,
            "config_hash": cfg_hash,
            "window_size": int(window_size),
            "stride": int(stride),
            "frame_offset": int(frame),
            "encoded_path": enc_path,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "evaluation": eval_data,
    }


def cmd_eval_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, train_cfg, io_cfg = extract_configs(cfg)
    tok = _get_tok(args, train_cfg)
    report = _eval_single_accession(args, args.accession, tok)
    out_path = _write_eval_report(io_cfg, report, prefix=f"eval_one.{args.accession}.{tok}")
    print(f"{args.accession}: eval-one tokenizer={tok} -> {out_path}")
    return 0


def cmd_eval_catalog(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, train_cfg, io_cfg = extract_configs(cfg)
    tok = _get_tok(args, train_cfg)
    accessions = read_catalog(args.catalog)

    reports = []
    for acc in accessions:
        reports.append(_eval_single_accession(args, acc, tok))

    payload = {
        "metadata": {
            "catalog": args.catalog,
            "num_accessions": len(accessions),
            "tokenizer": tok,
            "seed": int(getattr(args, "seed", 1337)),
            "config_path": args.config,
            "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:16],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "reports": reports,
    }
    out_path = _write_eval_report(io_cfg, payload, prefix=f"eval_catalog.{tok}")
    print(f"eval-catalog tokenizer={tok} accessions={len(accessions)} -> {out_path}")
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
