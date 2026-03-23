import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

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
from perceptrome.encoding.bio_ast_export import (
    build_bio_ast_export_artifacts,
    export_filenames,
    normalize_visualization_loader_payload,
    stable_json_dumps,
)
from perceptrome.io_utils import select_unique_accessions, write_catalog
from perceptrome.encoding.parse import parse_fasta_sequence, parse_genbank_dna
from perceptrome.encoding.encode import encode_sequence_one_hot
from perceptrome.pretrain import PretrainPipelineConfig, run_pretraining
from perceptrome.scoring import reference_score
from perceptrome.run_layout import ensure_run_layout, path_in_run, update_run_manifest
from perceptrome.jobs import JobEngine, JobSpec
from perceptrome.jobs.artifact_index import build_artifact_entry
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


def _artifact(artifact_id: str, role: str, path: str, artifact_type: Optional[str] = None, mime_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return build_artifact_entry(artifact_id=str(artifact_id), role=str(role), path=str(path), artifact_type=artifact_type, mime_type=mime_type, metadata=metadata)
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


def _reference_score(generated_seq: str, ref_seq: str) -> Dict[str, float]:
    return reference_score(generated_seq, ref_seq)


def _job_params(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "func" or callable(value):
            continue
        params[key] = value
    return params


def _normalize_ast_cli_args(params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(params)
    if normalized.get("ast_node_type_prompt") is None:
        normalized["ast_node_type_prompt"] = []
    if normalized.get("ast_region_span") is None:
        normalized["ast_region_span"] = []
    if normalized.get("ast_graph_mask") is None:
        normalized["ast_graph_mask"] = "none"
    if normalized.get("ast_graph_hop_limit") is None:
        normalized["ast_graph_hop_limit"] = 1
    if normalized.get("ast_mask_strength") is None:
        normalized["ast_mask_strength"] = 0.0
    return normalized


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


def _load_split_payload(split_path: str) -> Dict[str, Any]:
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload.get("splits"), dict):
        raise ValueError(f"Invalid split file (missing splits): {split_path}")
    return payload


def _split_signature(split_payload: Dict[str, Any]) -> Dict[str, List[str]]:
    splits = split_payload.get("splits") or {}
    return {
        "train": list(splits.get("train", [])),
        "val": list(splits.get("val", [])),
        "test": list(splits.get("test", [])),
    }


def _assert_split_parity(token_split: Dict[str, Any], ast_split: Dict[str, Any], token_path: str, ast_path: str) -> None:
    a = _split_signature(token_split)
    b = _split_signature(ast_split)
    if a != b:
        raise ValueError(
            "Split mismatch between baseline and AST runs. "
            f"baseline={token_path} ast={ast_path}. Ensure identical train/val/test accession lists."
        )


def _flatten_dict(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        p = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_dict(value, p))
        else:
            out[p] = value
    return out


def _config_diff(token_cfg: Dict[str, Any], ast_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    left = _flatten_dict(token_cfg)
    right = _flatten_dict(ast_cfg)
    keys = sorted(set(left) | set(right))
    diffs: List[Dict[str, Any]] = []
    for key in keys:
        if left.get(key) != right.get(key):
            diffs.append({"key": key, "token_baseline": left.get(key), "ast_enabled": right.get(key)})
    return diffs


def _assert_benchmark_config_parity(token_cfg: Dict[str, Any], ast_cfg: Dict[str, Any]) -> None:
    req = [
        "training.window_size",
        "training.stride",
        "training.tokenizer",
        "training.frame_offset",
        "training.min_orf_aa",
        "training.batch_size",
        "training.steps_per_plasmid",
    ]
    left = _flatten_dict(token_cfg)
    right = _flatten_dict(ast_cfg)
    mismatched = [k for k in req if left.get(k) != right.get(k)]
    if mismatched:
        details = ", ".join(f"{k}: baseline={left.get(k)!r} ast={right.get(k)!r}" for k in mismatched)
        raise ValueError(f"Config parity check failed for compare-lanes ({details})")


def _count_model_parameters(*, cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    from perceptrome.encoding_main import tokenizer_meta
    from perceptrome.model import get_device, load_or_init_model

    _, train_cfg, io_cfg = extract_configs(cfg)
    io_cfg = _run_local_io_cfg(io_cfg)
    ensure_dirs(io_cfg)
    tok = _get_tok(args, train_cfg)
    window_size, _ = _pick_window_stride(args, train_cfg, tok)
    seq_len, vocab_size = tokenizer_meta(tok, window_size)
    loss_type = getattr(args, "loss_type", None) or ("ce" if tok == "aa" else "mse")

    model, _, _, _ = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=train_cfg.hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=get_device(),
        tokenizer=tok,
        loss_type=loss_type,
        model_type=train_cfg.model_type,
        transformer_d_model=train_cfg.transformer_d_model,
        transformer_nhead=train_cfg.transformer_nhead,
        transformer_layers=train_cfg.transformer_layers,
        transformer_dropout=train_cfg.transformer_dropout,
        ast_tree_layers=train_cfg.ast_tree_layers,
        ast_motif_kernel_size=train_cfg.ast_motif_kernel_size,
        ast_motif_channels=train_cfg.ast_motif_channels,
        beta_kl=train_cfg.beta_kl,
    )
    return int(sum(int(p.numel()) for p in model.parameters()))


def _run_local_io_cfg(io_cfg):
    layout = ensure_run_layout()
    io_cfg.checkpoints_dir = path_in_run(layout, "artifacts", "checkpoints")
    io_cfg.model_dir = path_in_run(layout, "artifacts", "model")
    os.makedirs(io_cfg.model_dir, exist_ok=True)
    os.makedirs(io_cfg.checkpoints_dir, exist_ok=True)
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
        artifacts=[_artifact(f"fetch_{accession}", "dataset.fetch", path, artifact_type="record")],
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
        "artifacts": [_artifact(f"encoded_{accession}", "dataset.encoded", encoded_path, artifact_type="npy")],
    }


def _infer_top_level_type(accession: str, source: str) -> str:
    lowered = str(accession).lower()
    if source == "genbank" and ("virus" in lowered or lowered.startswith("nc_00")):
        return "virus"
    if source == "genbank":
        return "plasmid"
    return "genome"


def _build_bio_ast(accession: str, source: str, io_cfg):
    src = source.lower()
    builder = BioASTBuilder()
    if src == "genbank":
        gb_path = os.path.join(io_cfg.cache_genbank_dir, f"{accession}.gb")
        sequence = parse_genbank_dna(gb_path)
        cds_features = parse_cds_features_from_genbank(gb_path)
    else:
        fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
        sequence = parse_fasta_sequence(fasta_path)
        cds_features = None

    top_level_type = _infer_top_level_type(accession, src)
    return builder.build(
        sequence=sequence,
        cds_features=cds_features,
        top_level_type=top_level_type,
        accession=str(accession),
        source_format=str(src),
        molecule_type="DNA",
        topology=("circular" if str(top_level_type).lower() == "plasmid" else "linear"),
    )


def _collect_bio_ast_transforms(accession: str, source: str, io_cfg) -> Dict[str, Any]:
    built = _build_bio_ast(accession=accession, source=source, io_cfg=io_cfg)
    return build_bio_ast_export_artifacts(built, accession=str(accession), source=str(source))


def _build_and_write_bio_ast(accession: str, source: str, io_cfg) -> Optional[Dict[str, str]]:
    try:
        transforms = _collect_bio_ast_transforms(accession=accession, source=source, io_cfg=io_cfg)
    except Exception as exc:
        logging.warning("%s: failed to build bio AST (%s)", accession, exc)
        return None

    layout = ensure_run_layout()
    ast_dir = path_in_run(layout, "artifacts", os.path.join("bio_ast", str(accession)))
    os.makedirs(ast_dir, exist_ok=True)

    filenames = export_filenames()
    output_paths: Dict[str, str] = {}
    for key, filename in filenames.items():
        out_path = os.path.join(ast_dir, filename)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(stable_json_dumps(transforms[key]))
        output_paths[key] = out_path

    update_run_manifest(
        layout,
        paths={"bio_ast": {str(accession): output_paths}},
        artifacts=[_artifact(f"bio_ast_{accession}_{k}", f"bio_ast.{k}", v, artifact_type="json") for k, v in output_paths.items()],
    )
    return output_paths


def cmd_bio_ast_build(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, _, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    src = str(args.source).lower()
    _ensure_record(str(args.accession), src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=bool(getattr(args, "force", False)))

    outputs = _build_and_write_bio_ast(str(args.accession), src, io_cfg)
    if not outputs:
        return 1
    print(f"{args.accession}: bio-ast transforms written under {os.path.dirname(next(iter(outputs.values())))}")
    return 0


def cmd_bio_ast_export(args: argparse.Namespace) -> int:
    layout = ensure_run_layout()
    acc_dir = path_in_run(layout, "artifacts", os.path.join("bio_ast", str(args.accession)))
    transform_to_file = export_filenames()
    if args.transform == "all":
        payload = {}
        for key, filename in transform_to_file.items():
            path = os.path.join(acc_dir, filename)
            with open(path, "r", encoding="utf-8") as handle:
                payload[key] = normalize_visualization_loader_payload(json.load(handle))
    else:
        path = os.path.join(acc_dir, transform_to_file[args.transform])
        with open(path, "r", encoding="utf-8") as handle:
            payload = normalize_visualization_loader_payload(json.load(handle))

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(stable_json_dumps(payload))
    print(f"Exported {args.transform} for {args.accession} -> {args.output}")
    return 0


def cmd_bio_ast_inspect(args: argparse.Namespace) -> int:
    layout = ensure_run_layout()
    acc_dir = path_in_run(layout, "artifacts", os.path.join("bio_ast", str(args.accession)))
    filenames = export_filenames()
    with open(os.path.join(acc_dir, filenames["canonical_ast"]), "r", encoding="utf-8") as handle:
        ast_payload = json.load(handle)
    with open(os.path.join(acc_dir, filenames["graph_edges"]), "r", encoding="utf-8") as handle:
        edges = normalize_visualization_loader_payload(json.load(handle))
    nodes = ast_payload.get("nodes", []) if isinstance(ast_payload, dict) else []
    print(f"accession={args.accession}")
    print(f"nodes={len(nodes)}")
    print(f"edges={len(edges)}")
    return 0


def cmd_bio_ast_visualize(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, _, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)
    src = str(args.source).lower()
    _ensure_record(str(args.accession), src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=bool(getattr(args, "force", False)))

    outputs = _build_and_write_bio_ast(str(args.accession), src, io_cfg)
    if not outputs:
        return 1
    print(f"{args.accession}: visualization artifacts written")
    print(f"canonical_ast={outputs['canonical_ast']}")
    print(f"tree_json={outputs['tree_json']}")
    print(f"graph_json={outputs['graph_json']}")
    print(f"storage_map={outputs['storage_map']}")
    print(f"summary_json={outputs['summary_json']}")
    return 0


def _checkpoint_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_bio_ast_embeddings_for_accession(
    *,
    accession: str,
    source: str,
    io_cfg,
    ckpt_path: Optional[str],
    hidden_dim: int,
    ast_tree_layers: int,
    ast_motif_kernel_size: int,
    ast_motif_channels: int,
) -> Dict[str, Any]:
    import torch
    from perceptrome.model import BioASTEmbeddingAPI

    built = _build_bio_ast(accession=accession, source=source, io_cfg=io_cfg)
    seq_tokens = encode_sequence_one_hot(built.sequence, window_size=len(built.sequence), stride=max(1, len(built.sequence)))[0]
    tree_tensors = built.to_tree_message_passing_tensors()

    seq_tensor = torch.from_numpy(seq_tokens).unsqueeze(0).to(dtype=torch.float32)
    node_ids = torch.from_numpy(tree_tensors["node_type_ids"]).unsqueeze(0).to(dtype=torch.long)
    coords = torch.from_numpy(tree_tensors["coords"]).unsqueeze(0).to(dtype=torch.float32)
    strand = torch.from_numpy(tree_tensors["strand"]).unsqueeze(0).to(dtype=torch.long)

    embedder = BioASTEmbeddingAPI(
        seq_vocab_size=int(seq_tensor.shape[-1]),
        hidden_dim=int(hidden_dim),
        ast_tree_layers=int(ast_tree_layers),
        motif_kernel_size=int(ast_motif_kernel_size),
        motif_channels=int(ast_motif_channels),
    )
    embedder.eval()

    with torch.no_grad():
        embeddings = embedder(seq_tensor, node_ids, ast_coords=coords, ast_strand=strand)

    layout = ensure_run_layout()
    emb_dir = path_in_run(layout, "artifacts", os.path.join("embeddings", "bio_ast"))
    os.makedirs(emb_dir, exist_ok=True)

    fixed_path = os.path.join(emb_dir, f"{accession}.fixed.npy")
    token_path = os.path.join(emb_dir, f"{accession}.token.npy")
    node_path = os.path.join(emb_dir, f"{accession}.node.npy")
    meta_path = os.path.join(emb_dir, f"{accession}.metadata.json")

    np.save(fixed_path, embeddings.fixed.detach().cpu().numpy())
    np.save(token_path, embeddings.token.detach().cpu().numpy())
    np.save(node_path, embeddings.node.detach().cpu().numpy())

    metadata = {
        "schema_version": "bio_ast_embedding_v1",
        "checkpoint": {
            "path": ckpt_path,
            "sha256": _checkpoint_sha256(ckpt_path),
        },
        "ast_config": {
            "ast_tree_layers": int(ast_tree_layers),
            "ast_motif_kernel_size": int(ast_motif_kernel_size),
            "ast_motif_channels": int(ast_motif_channels),
        },
        "accession": str(accession),
        "sequence_length": int(len(built.sequence)),
        "node_count": int(len(built.ast.nodes)),
        "embedding_shapes": {
            "fixed": list(embeddings.fixed.shape),
            "token": list(embeddings.token.shape),
            "node": list(embeddings.node.shape),
        },
        "artifacts": {
            "fixed_npy": fixed_path,
            "token_npy": token_path,
            "node_npy": node_path,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    update_run_manifest(
        layout,
        paths={"embeddings": {"bio_ast": {accession: metadata["artifacts"]}}},
        artifacts=[
            _artifact(f"embedding_{accession}_fixed", "embeddings.fixed", fixed_path, artifact_type="npy"),
            _artifact(f"embedding_{accession}_token", "embeddings.token", token_path, artifact_type="npy"),
            _artifact(f"embedding_{accession}_node", "embeddings.node", node_path, artifact_type="npy"),
            _artifact(f"embedding_{accession}_metadata", "embeddings.metadata", meta_path, artifact_type="json"),
        ],
    )
    return {"fixed": fixed_path, "token": token_path, "node": node_path, "metadata": meta_path}



def cmd_bio_ast_embed_export(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    src = str(args.source).lower()
    accessions = []
    if getattr(args, "accession", None):
        accessions.extend([str(a) for a in args.accession])
    if getattr(args, "catalog", None):
        accessions.extend(read_catalog(str(args.catalog)))
    accessions = [a for a in accessions if a]
    if not accessions:
        raise ValueError("Provide at least one accession via --accession or --catalog")

    hidden_dim = int(getattr(args, "hidden_dim", None) or train_cfg.hidden_dim)
    ast_tree_layers = int(getattr(args, "ast_tree_layers", 4))
    ast_motif_kernel_size = int(getattr(args, "ast_motif_kernel_size", 7))
    ast_motif_channels = int(getattr(args, "ast_motif_channels", 64))

    ckpt_path = os.path.join(io_cfg.checkpoints_dir, "latest.pt")
    for accession in accessions:
        _ensure_record(accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=bool(getattr(args, "force", False)))
        outputs = _export_bio_ast_embeddings_for_accession(
            accession=accession,
            source=src,
            io_cfg=io_cfg,
            ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None,
            hidden_dim=hidden_dim,
            ast_tree_layers=ast_tree_layers,
            ast_motif_kernel_size=ast_motif_kernel_size,
            ast_motif_channels=ast_motif_channels,
        )
        print(f"{accession}: embeddings exported -> {outputs['metadata']}")
    return 0

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
    update_run_manifest(
        layout,
        paths={"encoded": {str(args.accession): out_path}},
        artifacts=[_artifact(f"encoded_{args.accession}", "dataset.encoded", out_path, artifact_type="npy")],
    )
    ast_outputs = _build_and_write_bio_ast(args.accession, src, io_cfg)
    if ast_outputs:
        logging.info("%s: bio AST artifacts written under %s", args.accession, os.path.dirname(next(iter(ast_outputs.values()))))
    print(f"{args.accession}: encoded tokenizer={tok} source={src} -> shape={encoded.shape} saved={out_path}")
    return 0


def cmd_train_one(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="train_one", config_path=str(args.config), params=_job_params(args))
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
    spec = JobSpec(kind="stream", config_path=str(args.config), params=_job_params(args))
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    print("[stream] Training complete.")
    return 0

def cmd_generate_plasmid(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="generate_plasmid", config_path=str(args.config), params=_normalize_ast_cli_args(_job_params(args)))
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    print(f"[generate-plasmid] wrote {result.data.get('length')} bp -> {result.data.get('output', args.output)}")
    return 0

def cmd_validate_plasmid(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="validate_plasmid", config_path=str(args.config), params=_job_params(args))
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    top_rows = result.data.get("results", [])
    print(f"[validate-plasmid] refs={len(top_rows)}")
    for i, row in enumerate(top_rows, start=1):
        print(f"{i:>4d} {row['accession']:<16s} {row['score']:.4f} {row['ref_len']}")
    return 0

def cmd_generate_protein(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="generate_protein", config_path=str(args.config), params=_normalize_ast_cli_args(_job_params(args)))
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    print(f"[generate-protein] wrote {result.data.get('length')} aa -> {result.data.get('output', args.output)}")
    return 0

def cmd_pretrain(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="pretrain", config_path=str(args.config), params=_job_params(args))
    result = JobEngine().run(spec)
    if not result.ok:
        raise RuntimeError(result.message)
    print("[pretrain] complete")
    metrics = result.data.get("metrics", {})
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {float(v):.6f}")
    return 0


def cmd_compare_lanes(args: argparse.Namespace) -> int:
    baseline_cfg = load_full_config(args.baseline_config or args.config)
    ast_cfg = load_full_config(args.ast_config or args.config)
    _assert_benchmark_config_parity(baseline_cfg, ast_cfg)

    _, _, io_cfg = extract_configs(baseline_cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    split_name = str(args.split_name)
    baseline_split_path = str(args.baseline_split_path) if getattr(args, "baseline_split_path", None) else _default_split_path(io_cfg, split_name)
    ast_split_path = str(args.ast_split_path) if getattr(args, "ast_split_path", None) else baseline_split_path
    baseline_split = _load_split_payload(baseline_split_path)
    ast_split = _load_split_payload(ast_split_path)
    _assert_split_parity(baseline_split, ast_split, baseline_split_path, ast_split_path)

    split_sig = _split_signature(baseline_split)
    train_accessions = list(split_sig.get("train", []))
    if not train_accessions:
        raise ValueError("compare-lanes requires a non-empty train split")

    layout = ensure_run_layout(run_id=(getattr(args, "experiment_id", None) or _new_experiment_id("compare_lanes")))
    catalog_path = path_in_run(layout, "inputs", f"{split_name}.train.catalog.txt")
    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_accessions) + "\n")

    baseline_params = _job_params(args)
    baseline_params["catalog"] = catalog_path
    baseline_params["max_epochs"] = int(args.max_epochs)
    baseline_params["model_type"] = str(args.baseline_model_type)

    ast_params = _job_params(args)
    ast_params["catalog"] = catalog_path
    ast_params["max_epochs"] = int(args.max_epochs)
    ast_params["model_type"] = str(args.ast_model_type)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    t0 = time.perf_counter()
    baseline_result = JobEngine().run(JobSpec(kind="stream", config_path=str(args.baseline_config or args.config), params=baseline_params))
    baseline_runtime = time.perf_counter() - t0
    if not baseline_result.ok:
        raise RuntimeError(f"baseline lane failed: {baseline_result.message}")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    t1 = time.perf_counter()
    ast_result = JobEngine().run(JobSpec(kind="stream", config_path=str(args.ast_config or args.config), params=ast_params))
    ast_runtime = time.perf_counter() - t1
    if not ast_result.ok:
        raise RuntimeError(f"AST lane failed: {ast_result.message}")

    baseline_manifest = str(baseline_result.data.get("manifest_path"))
    ast_manifest = str(ast_result.data.get("manifest_path"))
    if not baseline_manifest or not os.path.exists(baseline_manifest):
        raise RuntimeError("baseline metrics manifest missing")
    if not ast_manifest or not os.path.exists(ast_manifest):
        raise RuntimeError("AST metrics manifest missing")

    with open(baseline_manifest, "r", encoding="utf-8") as f:
        baseline_payload = json.load(f)
    with open(ast_manifest, "r", encoding="utf-8") as f:
        ast_payload = json.load(f)

    baseline_metrics = baseline_payload.get("metrics") or {}
    ast_metrics = ast_payload.get("metrics") or {}

    baseline_count = _count_model_parameters(cfg=baseline_cfg, args=argparse.Namespace(**baseline_params))
    ast_count = _count_model_parameters(cfg=ast_cfg, args=argparse.Namespace(**ast_params))

    rows = [
        {
            "lane": "token_baseline",
            "run_kind": "stream",
            "config_path": str(args.baseline_config or args.config),
            "split_path": baseline_split_path,
            "manifest_path": baseline_manifest,
            "processed_accessions": baseline_metrics.get("processed_accessions"),
            "last_total_loss": baseline_metrics.get("last_total_loss"),
            "runtime_seconds": baseline_runtime,
            "parameter_count": baseline_count,
            "seed": int(args.seed),
            "train_count": len(split_sig["train"]),
            "val_count": len(split_sig["val"]),
            "test_count": len(split_sig["test"]),
        },
        {
            "lane": "ast_enabled",
            "run_kind": "stream",
            "config_path": str(args.ast_config or args.config),
            "split_path": ast_split_path,
            "manifest_path": ast_manifest,
            "processed_accessions": ast_metrics.get("processed_accessions"),
            "last_total_loss": ast_metrics.get("last_total_loss"),
            "runtime_seconds": ast_runtime,
            "parameter_count": ast_count,
            "seed": int(args.seed),
            "train_count": len(split_sig["train"]),
            "val_count": len(split_sig["val"]),
            "test_count": len(split_sig["test"]),
        },
    ]

    diffs = _config_diff(baseline_cfg, ast_cfg)
    result_json = path_in_run(layout, "metrics", "compare_lanes.json")
    result_csv = path_in_run(layout, "metrics", "compare_lanes.csv")
    diff_json = path_in_run(layout, "provenance", "compare_lanes_config_diff.json")

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config_diffs": diffs}, f, indent=2)
        f.write("\n")
    with open(result_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(diff_json, "w", encoding="utf-8") as f:
        json.dump(diffs, f, indent=2)
        f.write("\n")

    update_run_manifest(
        layout,
        paths={
            "compare_lanes": {
                "catalog": catalog_path,
                "baseline_manifest": baseline_manifest,
                "ast_manifest": ast_manifest,
                "results_json": result_json,
                "results_csv": result_csv,
                "config_diff_json": diff_json,
            }
        },
        metrics={"compare_lanes": {"rows": rows}},
        provenance={
            "compare_lanes": {
                "seed": int(args.seed),
                "split_name": split_name,
                "baseline_split": baseline_split_path,
                "ast_split": ast_split_path,
                "config_diffs": diffs,
            }
        },
        artifacts=[
            _artifact("compare_lanes_json", "metrics.compare_lanes", result_json, artifact_type="json"),
            _artifact("compare_lanes_csv", "metrics.compare_lanes", result_csv, artifact_type="csv"),
            _artifact("compare_lanes_diff", "provenance.compare_lanes", diff_json, artifact_type="json"),
        ],
    )

    print(f"[compare-lanes] baseline={baseline_manifest}")
    print(f"[compare-lanes] ast={ast_manifest}")
    print(f"[compare-lanes] results_json={result_json}")
    print(f"[compare-lanes] results_csv={result_csv}")
    return 0


def cmd_design_loop(args: argparse.Namespace) -> int:
    spec = JobSpec(kind="design_loop", config_path=str(args.config), params=_normalize_ast_cli_args(_job_params(args)))
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
