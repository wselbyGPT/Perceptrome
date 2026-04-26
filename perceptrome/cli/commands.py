import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import random
import shutil
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
    resolve_run_local_io_cfg, write_tui_startup_context,
)
from perceptrome.config import extract_uniprot_config
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
from perceptrome.structure.colabfold_runner import resolve_colabfold_binary, run_colabfold_monomer
from perceptrome.structure.alphafold3_runner import (
    resolve_alphafold3_binary,
    resolve_alphafold3_db_dir,
    resolve_alphafold3_model_dir,
    run_alphafold3_monomer,
)
from perceptrome.structure.fold_manifest import build_fold_manifest_update
from perceptrome.structure.parsers import discover_alphafold3_outputs, discover_colabfold_outputs
from perceptrome.uniprot_api import build_count_query, fetch_uniprot_count
from perceptrome.uniprot_dataset import fetch_uniprot_dataset
from perceptrome.virus.catalog import (
    build_catalog_from_args,
    create_snapshot_bundle,
    inspect_manifest_payload,
    load_manifest,
    rebuild_from_manifest,
)
from perceptrome.virus.fetch import fetch_virus_from_args
from perceptrome.virus.splits import create_split_payload

from perceptrome.structure.summary import (
    FoldSummaryRecord,
    build_batch_summary,
    build_fold_summary_record,
    write_batch_summary_json,
    write_batch_summary_tsv,
    write_summary_json,
    write_summary_tsv,
)

from perceptrome.genome.annotator import (
    GenomeAnnotationResult,
    annotate_genome_input,
    read_annotation_json,
    write_annotation_json,
)
from perceptrome.genome.annotation_manifest import build_annotation_manifest_update
from perceptrome.genome.parsers import genome_input_files_in_dir
from perceptrome.genome.summary import (
    GenomeAnnotationRecord,
    build_batch_summary as build_genome_batch_summary,
    build_genome_annotation_record,
    write_batch_summary_json as write_genome_batch_summary_json,
    write_batch_summary_tsv as write_genome_batch_summary_tsv,
    write_summary_json as write_genome_summary_json,
    write_summary_tsv as write_genome_summary_tsv,
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


def _copy_run_input(layout, source_path: str, target_name: str) -> str:
    dst = path_in_run(layout, "inputs", target_name)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(source_path, dst)
    return dst


def _fasta_files_in_dir(input_dir: str) -> List[str]:
    files: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if low.endswith((".fasta", ".fa", ".faa")):
            files.append(path)
    return files


def _resolve_fold_summary_json(run_or_path: str) -> str:
    if os.path.isfile(run_or_path):
        return run_or_path
    run_path = os.path.join("runs", str(run_or_path), "outputs", "summary.json")
    if os.path.exists(run_path):
        return run_path
    batch_path = os.path.join("runs", str(run_or_path), "outputs", "batch_summary.json")
    if os.path.exists(batch_path):
        return batch_path
    raise FileNotFoundError(f"Unable to locate fold summary for: {run_or_path}")


def _discover_fold_outputs_for_engine(engine: str, fold_output_dir: str):
    if str(engine).lower() == "alphafold3":
        return discover_alphafold3_outputs(fold_output_dir)
    return discover_colabfold_outputs(fold_output_dir)


def _run_fold_one_internal(
    *,
    fasta_path: str,
    layout,
    engine: str,
    num_recycle: int,
    num_models: int,
    colabfold_bin: Optional[str],
    alphafold3_bin: Optional[str] = None,
    alphafold3_model_dir: Optional[str] = None,
    alphafold3_db_dir: Optional[str] = None,
    num_seeds: int = 1,
    num_diffusion_samples: int = 5,
) -> tuple[FoldSummaryRecord, str, str]:
    input_copy = _copy_run_input(layout, fasta_path, os.path.basename(fasta_path))
    protein_id = os.path.splitext(os.path.basename(fasta_path))[0]
    fold_output_dir = path_in_run(layout, "artifacts", os.path.join("fold", protein_id))
    engine_key = str(engine).lower()
    log_suffix = "alphafold3" if engine_key == "alphafold3" else "colabfold"
    stdout_log = path_in_run(layout, "provenance", f"{protein_id}.{log_suffix}.stdout.log")
    stderr_log = path_in_run(layout, "provenance", f"{protein_id}.{log_suffix}.stderr.log")
    started_at = iso_now()

    warnings: List[str] = []
    errors: List[str] = []
    status = "ok"

    if engine_key == "alphafold3":
        try:
            resolved_bin = resolve_alphafold3_binary(alphafold3_bin)
            resolved_model_dir = resolve_alphafold3_model_dir(alphafold3_model_dir)
            resolved_db_dir = resolve_alphafold3_db_dir(alphafold3_db_dir)
            result = run_alphafold3_monomer(
                fasta_path=input_copy,
                output_dir=fold_output_dir,
                alphafold3_bin=resolved_bin,
                model_dir=resolved_model_dir,
                db_dir=resolved_db_dir,
                stdout_log_path=stdout_log,
                stderr_log_path=stderr_log,
                job_name=protein_id,
                num_seeds=int(num_seeds),
                num_diffusion_samples=int(num_diffusion_samples),
            )
            if result.return_code != 0:
                status = "error"
                errors.append(f"alphafold3_exit_code={result.return_code}")
        except FileNotFoundError as err:
            status = "error"
            errors.append(str(err))
            with open(stderr_log, "a", encoding="utf-8") as handle:
                handle.write(str(err) + "\n")
    else:
        try:
            resolved_bin = resolve_colabfold_binary(colabfold_bin)
            result = run_colabfold_monomer(
                fasta_path=input_copy,
                output_dir=fold_output_dir,
                num_recycle=num_recycle,
                num_models=num_models,
                colabfold_bin=resolved_bin,
                stdout_log_path=stdout_log,
                stderr_log_path=stderr_log,
            )
            if result.return_code != 0:
                status = "error"
                errors.append(f"colabfold_exit_code={result.return_code}")
        except FileNotFoundError as err:
            status = "error"
            errors.append(str(err))
            with open(stderr_log, "a", encoding="utf-8") as handle:
                handle.write(str(err) + "\n")

    artifacts = _discover_fold_outputs_for_engine(engine_key, fold_output_dir)
    if status == "ok" and not artifacts.structures_pdb and not artifacts.structures_cif:
        status = "error"
        errors.append("No structure outputs (.pdb/.cif) discovered")

    record = build_fold_summary_record(
        protein_id=protein_id,
        source_input_path=input_copy,
        engine=engine,
        engine_status=status,
        artifacts=artifacts,
        warnings=warnings,
        errors=errors,
        started_at=started_at,
        completed_at=iso_now(),
    )
    return record, stdout_log, stderr_log


def _resume_fold_record_if_available(
    *,
    fasta_path: str,
    layout,
    engine: str,
) -> Optional[FoldSummaryRecord]:
    protein_id = os.path.splitext(os.path.basename(fasta_path))[0]
    fold_output_dir = path_in_run(layout, "artifacts", os.path.join("fold", protein_id))
    if not os.path.isdir(fold_output_dir):
        return None
    artifacts = _discover_fold_outputs_for_engine(engine, fold_output_dir)
    if not artifacts.structures_pdb and not artifacts.structures_cif:
        return None
    started_at = iso_now()
    return build_fold_summary_record(
        protein_id=protein_id,
        source_input_path=str(fasta_path),
        engine=engine,
        engine_status="ok",
        artifacts=artifacts,
        warnings=["resumed_from_existing_outputs"],
        errors=[],
        started_at=started_at,
        completed_at=iso_now(),
    )


# -----------------------------
# Commands
# -----------------------------
def _uniprot_query_with_mode(query: str, mode: str) -> str:
    return build_count_query(mode=str(mode or "all"), explicit_query=query, default_query="")


def cmd_uniprot_count(args: argparse.Namespace) -> int:
    try:
        cfg = load_full_config(args.config)
        _, _, io_cfg = extract_configs(cfg)
        uniprot_cfg = extract_uniprot_config(cfg)
        io_cfg = _run_local_io_cfg(io_cfg)
        ensure_dirs(io_cfg)
        query = str(getattr(args, "query", "") or uniprot_cfg.default_query)
        full_query = _uniprot_query_with_mode(query, args.mode)
        count_payload = fetch_uniprot_count(
            full_query,
            timeout=float(uniprot_cfg.request_timeout),
            max_retries=int(uniprot_cfg.retries),
            backoff_seconds=float(uniprot_cfg.backoff_seconds),
            base_url=str(uniprot_cfg.base_url),
        )
        total = int(count_payload["count"])
    except Exception as err:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(err)}))
        else:
            print(f"[uniprot-count] error: {err}")
        return 1

    result = {"ok": True, "query": str(query), "mode": str(args.mode), "effective_query": full_query, "count": total}
    if getattr(args, "json", False):
        print(json.dumps(result))
    else:
        print(f"[uniprot-count] mode={args.mode} count={total}")
    return 0


def cmd_uniprot_fetch(args: argparse.Namespace) -> int:
    try:
        cfg = load_full_config(args.config)
        _, _, io_cfg = extract_configs(cfg)
        uniprot_cfg = extract_uniprot_config(cfg)
        io_cfg = _run_local_io_cfg(io_cfg)
        ensure_dirs(io_cfg)

        query = str(getattr(args, "query", "") or uniprot_cfg.default_query)
        full_query = _uniprot_query_with_mode(query, args.mode if hasattr(args, "mode") else "all")
        records_per_shard = int(getattr(args, "records_per_shard", None) or uniprot_cfg.records_per_shard)
        if records_per_shard <= 0:
            raise ValueError("--records-per-shard must be > 0")

        output_dir = str(getattr(args, "output_dir", None) or os.path.join(io_cfg.cache_fasta_dir, "uniprot"))
        os.makedirs(output_dir, exist_ok=True)
        prefix = str(getattr(args, "prefix", "uniprot") or "uniprot")
        use_gzip = bool(getattr(args, "gzip_output", False) or uniprot_cfg.gzip_output)
        include_isoforms = bool(getattr(args, "include_isoforms", False) or uniprot_cfg.include_isoforms)
        count_only = bool(getattr(args, "count_only", False))
        prefix_path = os.path.join(output_dir, prefix)

        result = fetch_uniprot_dataset(
            query=full_query,
            include_isoforms=include_isoforms,
            prefix_path=prefix_path,
            records_per_shard=records_per_shard,
            use_gzip=use_gzip,
            resume=bool(getattr(args, "resume", False)),
            timeout=float(uniprot_cfg.request_timeout),
            max_retries=int(uniprot_cfg.retries),
            backoff_seconds=float(uniprot_cfg.backoff_seconds),
            count_only=count_only,
            base_url=str(uniprot_cfg.base_url),
        )

        live_count = result.get("live_count") or {}
        total = int(live_count.get("count") or 0)
        if count_only:
            payload = {
                "ok": True,
                "query": str(args.query),
                "effective_query": full_query,
                "count": total,
                "count_only": True,
                "count_source": live_count.get("count_source"),
            }
            if getattr(args, "json", False):
                print(json.dumps(payload))
            else:
                print(f"[uniprot-fetch] count={total}")
            return 0
        manifest = result.get("manifest") or {}
        shards = [str(item.get("path")) for item in (manifest.get("shards") or [])]
        payload = {
            "ok": True,
            "query": str(args.query),
            "effective_query": full_query,
            "count": total,
            "downloaded_records": int(manifest.get("total_records") or 0),
            "output_dir": output_dir,
            "shards": shards,
            "records_per_shard": records_per_shard,
            "include_isoforms": include_isoforms,
            "gzip_output": use_gzip,
            "resumed": bool(result.get("resumed", False)),
            "manifest_path": result.get("manifest_path"),
            "catalog_path": result.get("catalog_path"),
        }
        if getattr(args, "json", False):
            print(json.dumps(payload))
        else:
            print(f"[uniprot-fetch] query count={total} downloaded={payload['downloaded_records']}")
            print(f"[uniprot-fetch] wrote {len(shards)} shard(s) under {output_dir}")
            print(f"[uniprot-fetch] manifest: {result.get('manifest_path')}")
            print(f"[uniprot-fetch] catalog: {result.get('catalog_path')}")
        return 0
    except Exception as err:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(err)}))
        else:
            print(f"[uniprot-fetch] error: {err}")
        return 1


def cmd_virus_fetch(args: argparse.Namespace) -> int:
    try:
        cfg = load_full_config(args.config)
        _, _, io_cfg = extract_configs(cfg)
        io_cfg = _run_local_io_cfg(io_cfg)
        ensure_dirs(io_cfg)

        setattr(args, "_argv", list(sys.argv))
        result = fetch_virus_from_args(args, io_state_file=io_cfg.state_file)
        payload = {
            "ok": True,
            "source": result["source"],
            "run_dir": result["run_dir"],
            "staged_dir": result["staged_dir"],
            "manifest_path": result["manifest_path"],
            "archive_count": result["archive_count"],
            "indexed_file_count": result["indexed_file_count"],
        }
        if getattr(args, "json", False):
            print(json.dumps(payload))
        else:
            print(f"[virus-fetch] source={payload['source']}")
            print(f"[virus-fetch] run_dir={payload['run_dir']}")
            print(f"[virus-fetch] staged_dir={payload['staged_dir']}")
            print(f"[virus-fetch] manifest={payload['manifest_path']}")
            print(
                f"[virus-fetch] archives={payload['archive_count']} indexed_files={payload['indexed_file_count']}"
            )
        return 0
    except Exception as err:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(err)}))
        else:
            print(f"[virus-fetch] error: {err}")
        return 1


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




def cmd_virus_catalog_build(args: argparse.Namespace) -> int:
    try:
        result = build_catalog_from_args(args)
        print(f"Wrote catalog: {result['catalog_path']}")
        print(f"Manifest: {result['manifest_path']}")
        print(f"Accessions: {result['accession_count']}")
        print(f"SHA256: {result['accession_sha256']}")
        if result.get("snapshot_references"):
            print("Snapshots:")
            for ref in result["snapshot_references"]:
                print(f"  - {ref}")
        return 0
    except Exception as err:
        print(f"[virus-catalog build] error: {err}")
        return 1


def cmd_virus_catalog_inspect(args: argparse.Namespace) -> int:
    try:
        manifest = load_manifest(str(args.manifest))
        payload = inspect_manifest_payload(manifest, catalog_path=getattr(args, "catalog", None))
        print(f"Manifest: {args.manifest}")
        print(f"Catalog: {payload['catalog_path']} (exists={payload['catalog_exists']})")
        print(f"Count expected/actual: {payload['expected_count']}/{payload['actual_count']}")
        print(f"Hash expected: {payload['expected_hash']}")
        print(f"Hash actual:   {payload['actual_hash'] or '<missing>'}")
        print(f"Hash match:    {payload['hash_matches']}")
        print(f"Query mode: {payload['query_mode']}")
        print(f"Filters: {payload['filters'] if payload['filters'] else '[]'}")
        print(f"Includes: {payload['includes'] if payload['includes'] else '[]'}")
        print(f"Snapshot refs: {len(payload['snapshot_references'])}")
        if payload["snapshot_references"]:
            for ref in payload["snapshot_references"]:
                print(f"  - {ref}")
        if payload["accession_preview"]:
            print("Accession preview:")
            for acc in payload["accession_preview"]:
                print(f"  - {acc}")
        return 0
    except Exception as err:
        print(f"[virus-catalog inspect] error: {err}")
        return 1


def cmd_virus_catalog_rebuild(args: argparse.Namespace) -> int:
    try:
        result = rebuild_from_manifest(
            manifest_path=str(args.manifest),
            output_catalog=getattr(args, "output", None),
            datasets_bin=getattr(args, "datasets_bin", None),
        )
        status = "MATCH" if result["match"] else "MISMATCH"
        print(f"Manifest: {result['manifest_path']}")
        print(f"Catalog: {result['catalog_path']}")
        print(f"Stored count/hash: {result['stored_count']} / {result['stored_hash']}")
        print(f"Actual count/hash: {result['actual_count']} / {result['actual_hash']}")
        print(f"Rebuild status: {status}")
        return 0 if result["match"] else 2
    except Exception as err:
        print(f"[virus-catalog rebuild] error: {err}")
        return 1


def cmd_virus_catalog_snapshot(args: argparse.Namespace) -> int:
    try:
        result = create_snapshot_bundle(
            catalog_path=str(args.catalog),
            manifest_path=str(args.manifest),
            metadata_files=list(getattr(args, "metadata", None) or []),
            snapshot_dir=getattr(args, "snapshot_dir", None),
        )
        print(f"Snapshot bundle: {result['bundle_dir']}")
        print(f"Bundle manifest: {result['bundle_manifest_path']}")
        print(f"Files captured: {len(result['files'])}")
        return 0
    except Exception as err:
        print(f"[virus-catalog snapshot] error: {err}")
        return 1


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
    return resolve_run_local_io_cfg(io_cfg)


def _summarize_float_series(values: Any) -> Dict[str, Any]:
    arr = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    if arr.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _write_scope_summary_artifact(*, args: argparse.Namespace, default_name: str, payload: Dict[str, Any]) -> str:
    out_path = getattr(args, "summary_artifact", None)
    if not out_path:
        layout = ensure_run_layout()
        out_path = path_in_run(layout, "artifacts", default_name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"summary_json={out_path}")
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


def cmd_fold_one(args: argparse.Namespace) -> int:
    layout = ensure_run_layout(run_id=getattr(args, "run_id", None))
    record, stdout_log, stderr_log = _run_fold_one_internal(
        fasta_path=str(args.fasta),
        layout=layout,
        engine=str(getattr(args, "engine", "colabfold")),
        num_recycle=int(getattr(args, "num_recycle", 3)),
        num_models=int(getattr(args, "num_models", 5)),
        colabfold_bin=getattr(args, "colabfold_bin", None),
        alphafold3_bin=getattr(args, "alphafold3_bin", None),
        alphafold3_model_dir=getattr(args, "alphafold3_model_dir", None),
        alphafold3_db_dir=getattr(args, "alphafold3_db_dir", None),
        num_seeds=int(getattr(args, "num_seeds", 1) or 1),
        num_diffusion_samples=int(getattr(args, "num_diffusion_samples", 5) or 5),
    )

    summary_json = path_in_run(layout, "outputs", "summary.json")
    summary_tsv = path_in_run(layout, "outputs", "summary.tsv")
    write_summary_json(summary_json, [record])
    write_summary_tsv(summary_tsv, [record])

    manifest_update = build_fold_manifest_update(
        command_name="fold_one",
        run_id=layout.run_id,
        summary_json_path=summary_json,
        summary_tsv_path=summary_tsv,
        stdout_log_path=stdout_log,
        stderr_log_path=stderr_log,
        records=[record],
    )
    update_run_manifest(layout, **manifest_update)
    print(f"fold-one run_id={layout.run_id} protein_id={record.protein_id} status={record.engine_status}")
    print(f"summary_json={summary_json}")
    print(f"summary_tsv={summary_tsv}")
    return 0 if record.engine_status == "ok" else 2


def cmd_fold_batch(args: argparse.Namespace) -> int:
    layout = ensure_run_layout(run_id=getattr(args, "run_id", None))
    files = _fasta_files_in_dir(str(args.input_dir))
    if not files:
        raise FileNotFoundError(f"No FASTA files found in {args.input_dir}")
    started_at = iso_now()
    records: List[FoldSummaryRecord] = []
    shared_stdout = path_in_run(layout, "provenance", "fold_batch.stdout.log")
    shared_stderr = path_in_run(layout, "provenance", "fold_batch.stderr.log")

    min_len = getattr(args, "min_protein_aa", None)
    max_len = getattr(args, "max_protein_aa", None)
    keep_going = bool(getattr(args, "keep_going", False))
    resume = bool(getattr(args, "resume", False))
    skipped_by_length_min = 0
    skipped_by_length_max = 0
    resumed_success_count = 0

    for fasta_path in files:
        seq = parse_fasta_sequence(fasta_path)
        if min_len is not None and len(seq) < int(min_len):
            skipped_by_length_min += 1
            continue
        if max_len is not None and len(seq) > int(max_len):
            skipped_by_length_max += 1
            continue
        if resume:
            resumed_record = _resume_fold_record_if_available(
                fasta_path=fasta_path,
                layout=layout,
                engine=str(getattr(args, "engine", "colabfold")),
            )
            if resumed_record is not None:
                resumed_success_count += 1
                records.append(resumed_record)
                continue
        record, stdout_log, stderr_log = _run_fold_one_internal(
            fasta_path=fasta_path,
            layout=layout,
            engine=str(getattr(args, "engine", "colabfold")),
            num_recycle=int(getattr(args, "num_recycle", 3)),
            num_models=int(getattr(args, "num_models", 5)),
            colabfold_bin=getattr(args, "colabfold_bin", None),
            alphafold3_bin=getattr(args, "alphafold3_bin", None),
            alphafold3_model_dir=getattr(args, "alphafold3_model_dir", None),
            alphafold3_db_dir=getattr(args, "alphafold3_db_dir", None),
            num_seeds=int(getattr(args, "num_seeds", 1) or 1),
            num_diffusion_samples=int(getattr(args, "num_diffusion_samples", 5) or 5),
        )
        records.append(record)
        for src, dst in ((stdout_log, shared_stdout), (stderr_log, shared_stderr)):
            if os.path.exists(src):
                with open(src, "r", encoding="utf-8", errors="ignore") as hsrc, open(dst, "a", encoding="utf-8") as hdst:
                    hdst.write(hsrc.read())
                    hdst.write("\n")
        if record.engine_status != "ok" and not keep_going:
            break

    summary_json = path_in_run(layout, "outputs", "summary.json")
    summary_tsv = path_in_run(layout, "outputs", "summary.tsv")
    batch_json = path_in_run(layout, "outputs", "batch_summary.json")
    batch_tsv = path_in_run(layout, "outputs", "batch_summary.tsv")
    write_summary_json(summary_json, records)
    write_summary_tsv(summary_tsv, records)
    skipped_count = skipped_by_length_min + skipped_by_length_max
    batch = build_batch_summary(
        layout.run_id,
        str(getattr(args, "engine", "colabfold")),
        records,
        started_at,
        total_inputs=len(files),
        skipped_count=skipped_count,
        metadata={
            "attempted_count": len(records) - resumed_success_count,
            "successful_folds": sum(1 for row in records if row.engine_status == "ok"),
            "failed_attempts": sum(1 for row in records if row.engine_status != "ok"),
            "skip_reasons": {
                "length_below_min": skipped_by_length_min,
                "length_above_max": skipped_by_length_max,
            },
            "resumed_success_count": resumed_success_count,
            "resume_enabled": resume,
        },
    )
    write_batch_summary_json(batch_json, batch, records)
    write_batch_summary_tsv(batch_tsv, batch)

    manifest_update = build_fold_manifest_update(
        command_name="fold_batch",
        run_id=layout.run_id,
        summary_json_path=summary_json,
        summary_tsv_path=summary_tsv,
        stdout_log_path=shared_stdout,
        stderr_log_path=shared_stderr,
        records=records,
    )
    paths = manifest_update.get("paths") or {}
    if isinstance(paths, dict):
        fold_paths = dict(paths.get("fold") or {})
        fold_paths["batch_summary_json"] = batch_json
        fold_paths["batch_summary_tsv"] = batch_tsv
        paths["fold"] = fold_paths
    manifest_update["paths"] = paths
    update_run_manifest(layout, **manifest_update)
    print(f"fold-batch run_id={layout.run_id} records={len(records)} failed={batch.failed_count}")
    print(f"batch_summary_json={batch_json}")
    print(f"batch_summary_tsv={batch_tsv}")
    return 0 if batch.failed_count == 0 else 2


def cmd_fold_inspect(args: argparse.Namespace) -> int:
    summary_path = _resolve_fold_summary_json(str(args.run_or_path))
    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("records") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        rows = []
    print(f"summary={summary_path}")
    print(f"records={len(rows)}")
    ok_count = sum(1 for row in rows if isinstance(row, dict) and row.get("engine_status") == "ok")
    print(f"ok={ok_count} failed={len(rows) - ok_count}")
    for row in rows[:10]:
        if not isinstance(row, dict):
            continue
        print(
            f"- protein_id={row.get('protein_id')} status={row.get('engine_status')} "
            f"aa_length={row.get('aa_length')} mean_plddt={row.get('mean_plddt')}"
        )
    return 0


def cmd_fold_export(args: argparse.Namespace) -> int:
    summary_path = _resolve_fold_summary_json(str(args.run_or_path))
    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("records") if isinstance(payload, dict) else []
    rows = rows if isinstance(rows, list) else []

    if os.path.isfile(str(args.run_or_path)):
        output_dir = str(getattr(args, "output_dir", None) or os.path.dirname(summary_path) or ".")
    else:
        layout = ensure_run_layout(run_id=str(args.run_or_path))
        output_dir = str(getattr(args, "output_dir", None) or layout.outputs_dir)
    os.makedirs(output_dir, exist_ok=True)

    export_json = os.path.join(output_dir, "fold_export.json")
    export_tsv = os.path.join(output_dir, "fold_export.tsv")
    with open(export_json, "w", encoding="utf-8") as handle:
        json.dump({"records": rows}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    keys = ["protein_id", "engine_status", "aa_length", "mean_plddt", "min_plddt", "max_plddt", "ptm", "rank_1_structure_path"]
    with open(export_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, dialect="excel-tab")
        writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow({k: row.get(k) for k in keys})

    print(f"fold-export json={export_json}")
    print(f"fold-export tsv={export_tsv}")
    return 0


def _genome_accession_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return stem or "genome"


def _run_genome_annotate_one_internal(
    *,
    input_path: str,
    layout,
) -> tuple[GenomeAnnotationRecord, str, str]:
    accession = _genome_accession_from_path(input_path)
    input_copy = _copy_run_input(layout, input_path, os.path.basename(input_path))
    annotation_dir = path_in_run(layout, "artifacts", os.path.join("genome", accession))
    os.makedirs(annotation_dir, exist_ok=True)
    annotation_json_path = os.path.join(annotation_dir, "annotation.json")
    stdout_log = path_in_run(layout, "provenance", f"{accession}.genome.stdout.log")
    stderr_log = path_in_run(layout, "provenance", f"{accession}.genome.stderr.log")
    os.makedirs(os.path.dirname(stdout_log) or ".", exist_ok=True)
    open(stdout_log, "a").close()
    open(stderr_log, "a").close()

    started_at = iso_now()
    warnings: List[str] = []
    errors: List[str] = []
    status = "ok"
    result: Optional[GenomeAnnotationResult] = None

    try:
        result = annotate_genome_input(input_path=input_copy, accession=accession)
        write_annotation_json(annotation_json_path, result)
        with open(stdout_log, "a", encoding="utf-8") as handle:
            handle.write(
                f"annotated accession={accession} sequence_length={result.sequence_length} "
                f"genes={result.counts.gene_count} promoters={result.counts.promoter_count}\n"
            )
    except Exception as err:
        status = "error"
        errors.append(str(err))
        with open(stderr_log, "a", encoding="utf-8") as handle:
            handle.write(f"{type(err).__name__}: {err}\n")

    record = build_genome_annotation_record(
        accession=accession,
        source_input_path=input_copy,
        annotation_status=status,
        annotation_json_path=annotation_json_path if status == "ok" else None,
        result=result,
        warnings=warnings,
        errors=errors,
        started_at=started_at,
        completed_at=iso_now(),
    )
    return record, stdout_log, stderr_log


def _resume_genome_annotation_record_if_available(
    *,
    input_path: str,
    layout,
) -> Optional[GenomeAnnotationRecord]:
    accession = _genome_accession_from_path(input_path)
    annotation_dir = path_in_run(layout, "artifacts", os.path.join("genome", accession))
    annotation_json_path = os.path.join(annotation_dir, "annotation.json")
    if not os.path.isfile(annotation_json_path):
        return None
    try:
        result = read_annotation_json(annotation_json_path)
    except Exception:
        return None
    started_at = iso_now()
    return build_genome_annotation_record(
        accession=accession,
        source_input_path=input_path,
        annotation_status="ok",
        annotation_json_path=annotation_json_path,
        result=result,
        started_at=started_at,
        completed_at=started_at,
    )


def cmd_genome_annotate_batch(args: argparse.Namespace) -> int:
    layout = ensure_run_layout(run_id=getattr(args, "run_id", None))
    files = genome_input_files_in_dir(str(args.input_dir))
    if not files:
        raise FileNotFoundError(
            f"No GenBank or FASTA inputs found in {args.input_dir} "
            "(expected .gb/.gbk/.genbank/.fa/.fasta/.fna)"
        )
    started_at = iso_now()
    records: List[GenomeAnnotationRecord] = []
    shared_stdout = path_in_run(layout, "provenance", "genome_annotate_batch.stdout.log")
    shared_stderr = path_in_run(layout, "provenance", "genome_annotate_batch.stderr.log")
    os.makedirs(os.path.dirname(shared_stdout) or ".", exist_ok=True)
    open(shared_stdout, "a").close()
    open(shared_stderr, "a").close()

    min_len = getattr(args, "min_seq_len", None)
    max_len = getattr(args, "max_seq_len", None)
    keep_going = bool(getattr(args, "keep_going", False))
    resume = bool(getattr(args, "resume", False))
    skipped_by_length_min = 0
    skipped_by_length_max = 0
    skipped_by_load_error = 0
    resumed_success_count = 0

    for input_path in files:
        try:
            from perceptrome.genome.parsers import load_genome_input

            contents = load_genome_input(input_path)
            seq_len = len(contents.sequence)
        except Exception as err:
            skipped_by_load_error += 1
            with open(shared_stderr, "a", encoding="utf-8") as handle:
                handle.write(f"load_error path={input_path} error={err}\n")
            if not keep_going:
                break
            continue

        if min_len is not None and seq_len < int(min_len):
            skipped_by_length_min += 1
            continue
        if max_len is not None and seq_len > int(max_len):
            skipped_by_length_max += 1
            continue

        if resume:
            resumed = _resume_genome_annotation_record_if_available(
                input_path=input_path, layout=layout
            )
            if resumed is not None:
                resumed_success_count += 1
                records.append(resumed)
                continue

        record, stdout_log, stderr_log = _run_genome_annotate_one_internal(
            input_path=input_path, layout=layout
        )
        records.append(record)
        for src, dst in ((stdout_log, shared_stdout), (stderr_log, shared_stderr)):
            if os.path.exists(src):
                with open(src, "r", encoding="utf-8", errors="ignore") as hsrc, open(
                    dst, "a", encoding="utf-8"
                ) as hdst:
                    hdst.write(hsrc.read())
                    hdst.write("\n")
        if record.annotation_status != "ok" and not keep_going:
            break

    summary_json = path_in_run(layout, "outputs", "genome_summary.json")
    summary_tsv = path_in_run(layout, "outputs", "genome_summary.tsv")
    batch_json = path_in_run(layout, "outputs", "genome_batch_summary.json")
    batch_tsv = path_in_run(layout, "outputs", "genome_batch_summary.tsv")
    write_genome_summary_json(summary_json, records)
    write_genome_summary_tsv(summary_tsv, records)
    skipped_count = skipped_by_length_min + skipped_by_length_max + skipped_by_load_error
    batch = build_genome_batch_summary(
        layout.run_id,
        records,
        started_at,
        total_inputs=len(files),
        skipped_count=skipped_count,
        metadata={
            "attempted_count": len(records) - resumed_success_count,
            "successful_annotations": sum(1 for row in records if row.annotation_status == "ok"),
            "failed_attempts": sum(1 for row in records if row.annotation_status != "ok"),
            "skip_reasons": {
                "length_below_min": skipped_by_length_min,
                "length_above_max": skipped_by_length_max,
                "load_error": skipped_by_load_error,
            },
            "resumed_success_count": resumed_success_count,
            "resume_enabled": resume,
        },
    )
    write_genome_batch_summary_json(batch_json, batch, records)
    write_genome_batch_summary_tsv(batch_tsv, batch)

    manifest_update = build_annotation_manifest_update(
        command_name="genome_annotate_batch",
        run_id=layout.run_id,
        summary_json_path=summary_json,
        summary_tsv_path=summary_tsv,
        stdout_log_path=shared_stdout,
        stderr_log_path=shared_stderr,
        records=records,
    )
    paths = manifest_update.get("paths") or {}
    if isinstance(paths, dict):
        genome_paths = dict(paths.get("genome") or {})
        genome_paths["batch_summary_json"] = batch_json
        genome_paths["batch_summary_tsv"] = batch_tsv
        paths["genome"] = genome_paths
    manifest_update["paths"] = paths
    update_run_manifest(layout, **manifest_update)

    print(
        f"genome-annotate-batch run_id={layout.run_id} records={len(records)} "
        f"failed={batch.failed_count} skipped={batch.skipped_count}"
    )
    print(f"genome_summary_json={summary_json}")
    print(f"genome_batch_summary_json={batch_json}")
    return 0 if batch.failed_count == 0 else 2


def _resolve_genome_summary_json(run_or_path: str) -> str:
    if os.path.isfile(run_or_path):
        return run_or_path
    run_path = os.path.join("runs", str(run_or_path), "outputs", "genome_summary.json")
    if os.path.exists(run_path):
        return run_path
    batch_path = os.path.join("runs", str(run_or_path), "outputs", "genome_batch_summary.json")
    if os.path.exists(batch_path):
        return batch_path
    raise FileNotFoundError(f"Unable to locate genome summary for: {run_or_path}")


def cmd_genome_annotate_one(args: argparse.Namespace) -> int:
    layout = ensure_run_layout(run_id=getattr(args, "run_id", None))
    record, stdout_log, stderr_log = _run_genome_annotate_one_internal(
        input_path=str(args.input),
        layout=layout,
    )
    summary_json = path_in_run(layout, "outputs", "genome_summary.json")
    summary_tsv = path_in_run(layout, "outputs", "genome_summary.tsv")
    write_genome_summary_json(summary_json, [record])
    write_genome_summary_tsv(summary_tsv, [record])

    manifest_update = build_annotation_manifest_update(
        command_name="genome_annotate_one",
        run_id=layout.run_id,
        summary_json_path=summary_json,
        summary_tsv_path=summary_tsv,
        stdout_log_path=stdout_log,
        stderr_log_path=stderr_log,
        records=[record],
    )
    update_run_manifest(layout, **manifest_update)
    print(f"genome-annotate-one run_id={layout.run_id} accession={record.accession} status={record.annotation_status}")
    print(f"genome_summary_json={summary_json}")
    print(f"genome_summary_tsv={summary_tsv}")
    return 0 if record.annotation_status == "ok" else 2


def cmd_genome_annotate_inspect(args: argparse.Namespace) -> int:
    summary_path = _resolve_genome_summary_json(str(args.run_or_path))
    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("records") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        rows = []
    print(f"summary={summary_path}")
    print(f"records={len(rows)}")
    ok_count = sum(1 for row in rows if isinstance(row, dict) and row.get("annotation_status") == "ok")
    print(f"ok={ok_count} failed={len(rows) - ok_count}")
    for row in rows[:10]:
        if not isinstance(row, dict):
            continue
        print(
            f"- accession={row.get('accession')} status={row.get('annotation_status')} "
            f"seq_len={row.get('sequence_length')} genes={row.get('gene_count')} "
            f"cds={row.get('cds_count')} promoters={row.get('promoter_count')} "
            f"nodes={row.get('total_nodes')} edges={row.get('total_edges')}"
        )
    return 0


def cmd_genome_annotate_export(args: argparse.Namespace) -> int:
    summary_path = _resolve_genome_summary_json(str(args.run_or_path))
    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("records") if isinstance(payload, dict) else []
    rows = rows if isinstance(rows, list) else []

    if os.path.isfile(str(args.run_or_path)):
        output_dir = str(getattr(args, "output_dir", None) or os.path.dirname(summary_path) or ".")
    else:
        layout = ensure_run_layout(run_id=str(args.run_or_path))
        output_dir = str(getattr(args, "output_dir", None) or layout.outputs_dir)
    os.makedirs(output_dir, exist_ok=True)

    export_json = os.path.join(output_dir, "genome_export.json")
    export_tsv = os.path.join(output_dir, "genome_export.tsv")
    with open(export_json, "w", encoding="utf-8") as handle:
        json.dump({"records": rows}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    keys = [
        "accession", "annotation_status", "source_format", "cds_source",
        "sequence_length", "gene_count", "cds_count", "promoter_count",
        "operator_count", "rbs_count", "terminator_count",
        "total_nodes", "total_edges", "annotation_json_path",
    ]
    with open(export_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, dialect="excel-tab")
        writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow({k: row.get(k) for k in keys})

    print(f"genome-annotate-export json={export_json}")
    print(f"genome-annotate-export tsv={export_tsv}")
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


def cmd_virus_split_create(args: argparse.Namespace) -> int:
    try:
        cfg = load_full_config(args.config)
        _, _, io_cfg = extract_configs(cfg)
        io_cfg = _run_local_io_cfg(io_cfg)
        ensure_dirs(io_cfg)

        payload = create_split_payload(args)
        out_path = str(args.out) if args.out else _default_split_path(io_cfg, str(args.name))
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

        counts = payload.get("counts") or {}
        print(
            f"[virus-split-create] name={payload.get('name')} strategy={payload.get('strategy')} "
            f"total={counts.get('total', '?')} train={counts.get('train', '?')} "
            f"val={counts.get('val', '?')} test={counts.get('test', '?')} -> {out_path}"
        )
        return 0
    except Exception as err:
        print(f"[virus-split-create] error: {err}")
        return 1


def cmd_virus_split_show(args: argparse.Namespace) -> int:
    try:
        cfg = load_full_config(args.config)
        _, _, io_cfg = extract_configs(cfg)
        io_cfg = _run_local_io_cfg(io_cfg)
        ensure_dirs(io_cfg)

        split_name = str(args.name)
        split_path = str(args.path) if args.path else _default_split_path(io_cfg, split_name)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        counts = data.get("counts", {})
        source = data.get("source") or {}
        print(f"Split: {data.get('name', split_name)}")
        print(f"  path: {split_path}")
        print(f"  strategy: {data.get('strategy', '<unknown>')}")
        print(f"  catalog: {source.get('catalog_path', data.get('catalog_path', '<unknown>'))}")
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
    except Exception as err:
        print(f"[virus-split-show] error: {err}")
        return 1


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
    if curses is None and not bool(getattr(args, "no_curses", False)):
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
    layout = ensure_run_layout()
    summary_payload = {
        "run_id": layout.run_id,
        "job_id": str(args.accession),
        "command": "scope-one",
        "accession": str(args.accession),
        "tokenizer": tok,
        "source": src,
        "window_size": int(window_size),
        "stride": int(stride),
        "fps": float(args.fps),
        "loss_type": str(getattr(args, "loss_type", None) or ("ce" if tok == "aa" else "mse")),
        "encoded_path": enc_path,
        "errors": _summarize_float_series(errors),
        "gc": _summarize_float_series(metric),
    }
    _write_scope_summary_artifact(args=args, default_name=f"{args.accession}.scope_one.summary.json", payload=summary_payload)
    write_tui_startup_context(
        config_path=str(args.config),
        run_id=layout.run_id,
        job_id=str(args.accession),
        panel="overview",
        detail_surface="logs",
    )

    if bool(getattr(args, "no_curses", False)):
        return 0

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
    if curses is None and not bool(getattr(args, "no_curses", False)):
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
    layout = ensure_run_layout()
    summary_payload = {
        "run_id": layout.run_id,
        "job_id": str(args.accession),
        "command": "scope-stream",
        "accession": str(args.accession),
        "tokenizer": tok,
        "source": src,
        "window_size": int(window_size),
        "stride": int(stride),
        "fps": float(args.fps),
        "update_every": int(args.update_every),
        "steps_target": int(steps),
        "batch_size": int(batch_size),
        "loss_type": lt,
        "encoded_path": enc_path,
        "gc": _summarize_float_series(metric),
    }
    _write_scope_summary_artifact(args=args, default_name=f"{args.accession}.scope_stream.summary.json", payload=summary_payload)
    write_tui_startup_context(
        config_path=str(args.config),
        run_id=layout.run_id,
        job_id=str(args.accession),
        panel="overview",
        detail_surface="logs",
    )

    if bool(getattr(args, "no_curses", False)):
        return 0

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


def cmd_latent_cluster(args: argparse.Namespace) -> int:
    from perceptrome.encoding_main import tokenizer_meta
    from perceptrome.model import get_device, load_or_init_model
    from perceptrome.latent_cluster import (
        LatentClusterRecord,
        LatentClusterSummary,
        utc_now as _lc_now,
        run_kmeans,
        run_umap,
        write_cluster_records_json,
        write_cluster_records_tsv,
        write_cluster_summary_json,
    )

    if torch is None:
        raise RuntimeError("PyTorch is required for latent-cluster. Install torch first.")

    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    io_cfg = _run_local_io_cfg(io_cfg)
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

    catalog_path = str(args.catalog)
    accessions = read_catalog(catalog_path)
    if not accessions:
        raise ValueError(f"No accessions found in catalog: {catalog_path}")

    n_clusters = int(getattr(args, "n_clusters", 8))
    n_neighbors = int(getattr(args, "umap_n_neighbors", 15))
    min_dist = float(getattr(args, "umap_min_dist", 0.1))

    layout = ensure_run_layout(run_id=getattr(args, "run_id", None))

    # Load model once; reuse across all accessions.
    device = get_device()
    seq_len, vocab_size = tokenizer_meta(tok, window_size)
    loss_type = "ce" if tok == "aa" else "mse"
    model, _, _, _ = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=train_cfg.hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tok,
        loss_type=loss_type,
        model_type=train_cfg.model_type,
        transformer_d_model=train_cfg.transformer_d_model,
        transformer_nhead=train_cfg.transformer_nhead,
        transformer_layers=train_cfg.transformer_layers,
        transformer_dropout=train_cfg.transformer_dropout,
        beta_kl=train_cfg.beta_kl,
        ast_tree_layers=train_cfg.ast_tree_layers,
        ast_motif_kernel_size=train_cfg.ast_motif_kernel_size,
        ast_motif_channels=train_cfg.ast_motif_channels,
        hierarchical_latent_dim=train_cfg.hierarchical_latent_dim,
        ast_node_type_vocab_size=train_cfg.ast_node_type_vocab_size,
        hierarchical_ablation_mode=train_cfg.hierarchical_ablation_mode,
    )
    model.eval()

    started_at = iso_now()
    mu_list: List[np.ndarray] = []
    window_counts: List[int] = []
    valid_accessions: List[str] = []
    failed_accessions: List[str] = []

    print(f"latent-cluster: encoding {len(accessions)} accessions (tok={tok} src={src})")
    cache_kw = _cache_kwargs(tok, min_orf, pol)

    for i, accession in enumerate(accessions):
        try:
            _ensure_record(accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
            enc_path = encoded_cache_path(io_cfg, accession, tok, window_size, stride, frame, source=src, **cache_kw)
            if os.path.exists(enc_path):
                encoded = np.load(enc_path)
            else:
                encoded = encode_accession(
                    accession, io_cfg, window_size, stride,
                    tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
                    source=src,
                    max_windows_per_protein=pol.get("max_windows_per_protein"),
                    protein_len_min=pol.get("protein_len_min"),
                    protein_len_max=pol.get("protein_len_max"),
                    translation_only=bool(pol.get("translation_only", False)),
                    protein_opts=protein_opts,
                    save_to_disk=True, out_path=enc_path,
                )
            if encoded is None or encoded.shape[0] == 0:
                failed_accessions.append(accession)
                continue

            with torch.no_grad():
                wt = torch.from_numpy(encoded).to(device)
                N = int(wt.size(0))
                x_flat = wt.view(N, -1)
                mu, _ = model.encode(x_flat)
                # Mean-pool windows → one vector per accession.
                mu_vec = mu.mean(dim=0).cpu().numpy().astype(np.float32)

            mu_list.append(mu_vec)
            window_counts.append(N)
            valid_accessions.append(accession)
        except Exception as err:
            logging.warning("latent-cluster: skipping accession=%s error=%s", accession, err)
            failed_accessions.append(accession)

        if (i + 1) % 25 == 0 or (i + 1) == len(accessions):
            print(f"  {i + 1}/{len(accessions)} encoded  ok={len(valid_accessions)} failed={len(failed_accessions)}")

    n_valid = len(valid_accessions)
    if n_valid < 2:
        raise RuntimeError(
            f"Need at least 2 successfully encoded accessions, got {n_valid}. "
            f"Check model checkpoint and catalog paths."
        )

    mu_matrix = np.stack(mu_list, axis=0)  # (N, latent_dim)
    latent_dim = int(mu_matrix.shape[1])

    effective_clusters = min(n_clusters, n_valid)
    effective_neighbors = min(n_neighbors, n_valid - 1)

    print(f"Running k-means: n={n_valid} k={effective_clusters} latent_dim={latent_dim}")
    labels, centroids = run_kmeans(mu_matrix, n_clusters=effective_clusters)

    print(f"Running UMAP: n_neighbors={effective_neighbors} min_dist={min_dist}")
    projections = run_umap(mu_matrix, n_neighbors=effective_neighbors, min_dist=min_dist)

    records: List[LatentClusterRecord] = []
    for j, accession in enumerate(valid_accessions):
        mu_vec = mu_list[j]
        records.append(LatentClusterRecord(
            accession=accession,
            cluster_id=int(labels[j]),
            umap_x=float(projections[j, 0]),
            umap_y=float(projections[j, 1]),
            mu_norm=float(np.linalg.norm(mu_vec)),
            mu_mean=float(float(mu_vec.mean())),
            n_windows=window_counts[j],
        ))

    cluster_sizes: Dict[str, int] = {}
    for lbl in labels:
        k = str(int(lbl))
        cluster_sizes[k] = cluster_sizes.get(k, 0) + 1

    records_json = path_in_run(layout, "outputs", "latent_cluster_records.json")
    records_tsv = path_in_run(layout, "outputs", "latent_cluster_records.tsv")
    summary_json = path_in_run(layout, "outputs", "latent_cluster_summary.json")
    vectors_npy = path_in_run(layout, "artifacts", "latent_vectors.npy")

    write_cluster_records_json(records_json, records)
    write_cluster_records_tsv(records_tsv, records)

    summary = LatentClusterSummary(
        run_id=layout.run_id,
        catalog_path=catalog_path,
        n_accessions=n_valid,
        n_clusters=effective_clusters,
        latent_dim=latent_dim,
        umap_n_neighbors=effective_neighbors,
        umap_min_dist=min_dist,
        failed_accessions=failed_accessions,
        cluster_sizes=cluster_sizes,
        started_at=started_at,
        completed_at=iso_now(),
    )
    write_cluster_summary_json(summary_json, summary, records, centroids)

    os.makedirs(os.path.dirname(vectors_npy) or ".", exist_ok=True)
    np.save(vectors_npy, mu_matrix)

    update_run_manifest(
        layout,
        paths={"latent_cluster": {
            "records_json": records_json,
            "records_tsv": records_tsv,
            "summary_json": summary_json,
            "vectors_npy": vectors_npy,
        }},
        artifacts=[
            _artifact("latent_cluster_records_json", "latent.cluster.records", records_json, artifact_type="json"),
            _artifact("latent_cluster_records_tsv", "latent.cluster.records", records_tsv, artifact_type="tsv"),
            _artifact("latent_cluster_summary", "latent.cluster.summary", summary_json, artifact_type="json"),
            _artifact("latent_vectors_npy", "latent.vectors", vectors_npy, artifact_type="npy"),
        ],
    )

    print(f"latent-cluster run_id={layout.run_id} accessions={n_valid} clusters={effective_clusters} failed={len(failed_accessions)}")
    for cluster_key, sz in sorted(cluster_sizes.items(), key=lambda x: int(x[0])):
        print(f"  cluster {cluster_key}: {sz} accessions")
    print(f"records_json={records_json}")
    print(f"records_tsv={records_tsv}")
    print(f"summary_json={summary_json}")
    return 0


def cmd_latent_interpolate(args: argparse.Namespace) -> int:
    from perceptrome.encoding_main import tokenizer_meta
    from perceptrome.model import get_device, load_or_init_model
    from perceptrome.sampling import SamplingConfig, sample_window
    from perceptrome.latent_interp import (
        InterpStep,
        InterpSummary,
        tokens_to_sequence,
        write_interp_fasta,
        write_interp_summary_json,
    )

    if torch is None:
        raise RuntimeError("PyTorch is required for latent-interpolate.")

    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    io_cfg = _run_local_io_cfg(io_cfg)
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

    acc_a = str(args.acc_a)
    acc_b = str(args.acc_b)
    n_steps = max(2, int(getattr(args, "steps", 8)))
    n_windows = max(1, int(getattr(args, "n_windows", 1)))
    temperature = float(getattr(args, "temperature", 0.0))

    layout = ensure_run_layout(run_id=getattr(args, "run_id", None))

    # Load model once.
    device = get_device()
    seq_len, vocab_size = tokenizer_meta(tok, window_size)
    loss_type = "ce" if tok == "aa" else "mse"
    model, _, _, _ = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=train_cfg.hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tok,
        loss_type=loss_type,
        model_type=train_cfg.model_type,
        transformer_d_model=train_cfg.transformer_d_model,
        transformer_nhead=train_cfg.transformer_nhead,
        transformer_layers=train_cfg.transformer_layers,
        transformer_dropout=train_cfg.transformer_dropout,
        beta_kl=train_cfg.beta_kl,
        ast_tree_layers=train_cfg.ast_tree_layers,
        ast_motif_kernel_size=train_cfg.ast_motif_kernel_size,
        ast_motif_channels=train_cfg.ast_motif_channels,
        hierarchical_latent_dim=train_cfg.hierarchical_latent_dim,
        ast_node_type_vocab_size=train_cfg.ast_node_type_vocab_size,
        hierarchical_ablation_mode=train_cfg.hierarchical_ablation_mode,
    )
    model.eval()

    cache_kw = _cache_kwargs(tok, min_orf, pol)
    started_at = iso_now()

    def _get_mu(accession: str) -> np.ndarray:
        _ensure_record(accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
        enc_path = encoded_cache_path(io_cfg, accession, tok, window_size, stride, frame, source=src, **cache_kw)
        if os.path.exists(enc_path):
            encoded = np.load(enc_path)
        else:
            encoded = encode_accession(
                accession, io_cfg, window_size, stride,
                tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
                source=src,
                max_windows_per_protein=pol.get("max_windows_per_protein"),
                protein_len_min=pol.get("protein_len_min"),
                protein_len_max=pol.get("protein_len_max"),
                translation_only=bool(pol.get("translation_only", False)),
                protein_opts=protein_opts,
                save_to_disk=True, out_path=enc_path,
            )
        if encoded is None or encoded.shape[0] == 0:
            raise ValueError(f"Accession {accession!r} produced no encoded windows.")
        with torch.no_grad():
            wt = torch.from_numpy(encoded).to(device)
            N = int(wt.size(0))
            mu, _ = model.encode(wt.view(N, -1))
            return mu.mean(dim=0).cpu().numpy().astype(np.float32)

    print(f"latent-interpolate: encoding {acc_a}")
    mu_a = _get_mu(acc_a)
    print(f"latent-interpolate: encoding {acc_b}")
    mu_b = _get_mu(acc_b)

    latent_dim = int(mu_a.shape[0])
    s_cfg = SamplingConfig(strategy="temperature", temperature=max(1e-6, temperature)) if temperature > 0 else None

    interp_steps: List[InterpStep] = []
    with torch.no_grad():
        for i in range(n_steps):
            t = 0.0 if n_steps == 1 else float(i) / float(n_steps - 1)
            z_np = (1.0 - t) * mu_a + t * mu_b
            z = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0).to(device)

            seq_parts: List[str] = []
            for _ in range(n_windows):
                logits_flat = model.decode(z)
                logits = logits_flat.view(seq_len, vocab_size).cpu().numpy()
                if s_cfg is None:
                    tokens = np.argmax(logits, axis=-1).tolist()
                else:
                    tokens = sample_window(logits, s_cfg)
                seq_parts.append(tokens_to_sequence(tokens, tok))

            interp_steps.append(InterpStep(step=i, t=t, sequence="".join(seq_parts)))

    output_length = n_windows * window_size
    output_name = str(getattr(args, "output", None) or "latent_interp.fasta")
    fasta_path = path_in_run(layout, "outputs", os.path.basename(output_name))
    summary_json_path = path_in_run(layout, "outputs", "latent_interp_summary.json")

    write_interp_fasta(fasta_path, acc_a, acc_b, interp_steps)

    summary = InterpSummary(
        run_id=layout.run_id,
        acc_a=acc_a,
        acc_b=acc_b,
        steps=n_steps,
        latent_dim=latent_dim,
        tokenizer=tok,
        n_windows=n_windows,
        window_size=window_size,
        output_length=output_length,
        output_fasta=fasta_path,
        started_at=started_at,
        completed_at=iso_now(),
        metadata={"temperature": temperature, "source": src},
    )
    write_interp_summary_json(summary_json_path, summary, interp_steps)

    update_run_manifest(
        layout,
        paths={"latent_interp": {
            "fasta": fasta_path,
            "summary_json": summary_json_path,
        }},
        artifacts=[
            _artifact("latent_interp_fasta", "latent.interp.fasta", fasta_path, artifact_type="fasta"),
            _artifact("latent_interp_summary", "latent.interp.summary", summary_json_path, artifact_type="json"),
        ],
    )

    print(f"latent-interpolate run_id={layout.run_id} steps={n_steps} latent_dim={latent_dim} tok={tok}")
    print(f"fasta={fasta_path}")
    print(f"summary_json={summary_json_path}")
    return 0


def _resolve_cluster_summary_json(run_or_path: str) -> str:
    if os.path.isfile(run_or_path):
        return run_or_path
    candidate = os.path.join("runs", str(run_or_path), "outputs", "latent_cluster_summary.json")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Unable to locate latent cluster summary for: {run_or_path}")


def _derive_vectors_npy(summary_path: str) -> str:
    # summary lives at <run_root>/outputs/latent_cluster_summary.json
    # vectors live at  <run_root>/artifacts/latent_vectors.npy
    run_root = os.path.dirname(os.path.dirname(os.path.abspath(summary_path)))
    return os.path.join(run_root, "artifacts", "latent_vectors.npy")


def cmd_latent_seeds(args: argparse.Namespace) -> int:
    summary_path = _resolve_cluster_summary_json(str(args.run_or_path))

    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = payload.get("records") or []
    centroids_raw = payload.get("centroids") or []
    if not records:
        raise ValueError(f"No records found in {summary_path}")
    if not centroids_raw:
        raise ValueError(f"No centroids found in {summary_path} — was this produced by latent-cluster?")

    vectors_path = str(getattr(args, "vectors", None) or "")
    if not vectors_path:
        vectors_path = _derive_vectors_npy(summary_path)
    if not os.path.isfile(vectors_path):
        raise FileNotFoundError(
            f"latent_vectors.npy not found at {vectors_path}. "
            "Pass --vectors to specify the path explicitly."
        )

    mu_matrix = np.load(vectors_path).astype(np.float32)        # (N, latent_dim)
    centroids_arr = np.array(centroids_raw, dtype=np.float32)   # (k, latent_dim)
    include_outliers = bool(getattr(args, "outliers", False))

    # Group record row-indices by cluster_id.
    cluster_to_indices: Dict[int, List[int]] = {}
    for i, rec in enumerate(records):
        if isinstance(rec, dict):
            cid = int(rec.get("cluster_id", -1))
            cluster_to_indices.setdefault(cid, []).append(i)

    seeds: Dict[str, Any] = {}
    selected: List[str] = []    # de-duplicated, insertion-ordered

    for cid in sorted(cluster_to_indices.keys()):
        indices = cluster_to_indices[cid]
        if cid >= len(centroids_arr):
            continue                    # centroid index out of range — skip gracefully
        centroid = centroids_arr[cid]

        vecs = mu_matrix[indices]                               # (n, latent_dim)
        dists = np.linalg.norm(vecs - centroid, axis=1)        # (n,)

        arch_local = int(np.argmin(dists))
        arch_idx = indices[arch_local]
        arch_acc = str(records[arch_idx]["accession"])
        arch_dist = float(dists[arch_local])

        entry: Dict[str, Any] = {
            "cluster_id": cid,
            "size": len(indices),
            "archetype": arch_acc,
            "archetype_dist": arch_dist,
        }
        if arch_acc not in selected:
            selected.append(arch_acc)

        if include_outliers:
            out_local = int(np.argmax(dists))
            out_idx = indices[out_local]
            out_acc = str(records[out_idx]["accession"])
            out_dist = float(dists[out_local])
            entry["outlier"] = out_acc
            entry["outlier_dist"] = out_dist
            if out_acc not in selected:
                selected.append(out_acc)

        seeds[str(cid)] = entry

    # Resolve output directory.
    if os.path.isfile(str(args.run_or_path)):
        default_out = os.path.dirname(summary_path) or "."
    else:
        default_out = os.path.join("runs", str(args.run_or_path), "outputs")
    output_dir = str(getattr(args, "output_dir", None) or default_out)
    os.makedirs(output_dir, exist_ok=True)

    catalog_path = os.path.join(output_dir, "latent_seeds.catalog.txt")
    seeds_json_path = os.path.join(output_dir, "latent_seeds.json")

    with open(catalog_path, "w", encoding="utf-8") as f:
        for acc in selected:
            f.write(acc + "\n")

    seeds_payload: Dict[str, Any] = {
        "source_summary": summary_path,
        "vectors_npy": vectors_path,
        "include_outliers": include_outliers,
        "n_clusters": len(seeds),
        "n_selected": len(selected),
        "clusters": seeds,
        "catalog": catalog_path,
    }
    with open(seeds_json_path, "w", encoding="utf-8") as f:
        json.dump(seeds_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    kind = "archetypes + outliers" if include_outliers else "archetypes"
    print(f"latent-seeds clusters={len(seeds)} selected={len(selected)} ({kind})")
    for cid_str, entry in seeds.items():
        line = (
            f"  cluster {cid_str}: archetype={entry['archetype']} "
            f"dist={entry['archetype_dist']:.4f} size={entry['size']}"
        )
        if include_outliers and "outlier" in entry:
            line += f"  outlier={entry['outlier']} dist={entry['outlier_dist']:.4f}"
        print(line)
    print(f"catalog={catalog_path}")
    print(f"seeds_json={seeds_json_path}")
    return 0


def cmd_score_one(args: argparse.Namespace) -> int:
    from perceptrome.elbo_score import ELBORecord, score_windows, write_elbo_json
    from perceptrome.encoding_main import tokenizer_meta
    from perceptrome.model import get_device, load_or_init_model

    if torch is None:
        raise RuntimeError("PyTorch is required for score-one.")

    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    io_cfg = _run_local_io_cfg(io_cfg)
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

    device = get_device()
    seq_len, vocab_size = tokenizer_meta(tok, window_size)
    loss_type = "ce" if tok == "aa" else "mse"
    model, _, _, _ = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=train_cfg.hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tok,
        loss_type=loss_type,
        model_type=train_cfg.model_type,
        transformer_d_model=train_cfg.transformer_d_model,
        transformer_nhead=train_cfg.transformer_nhead,
        transformer_layers=train_cfg.transformer_layers,
        transformer_dropout=train_cfg.transformer_dropout,
        beta_kl=train_cfg.beta_kl,
        ast_tree_layers=train_cfg.ast_tree_layers,
        ast_motif_kernel_size=train_cfg.ast_motif_kernel_size,
        ast_motif_channels=train_cfg.ast_motif_channels,
        hierarchical_latent_dim=train_cfg.hierarchical_latent_dim,
        ast_node_type_vocab_size=train_cfg.ast_node_type_vocab_size,
        hierarchical_ablation_mode=train_cfg.hierarchical_ablation_mode,
    )
    model.eval()

    accession = str(args.accession)
    _ensure_record(accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
    cache_kw = _cache_kwargs(tok, min_orf, pol)
    enc_path = encoded_cache_path(io_cfg, accession, tok, window_size, stride, frame, source=src, **cache_kw)

    if os.path.exists(enc_path):
        encoded = np.load(enc_path)
    else:
        encoded = encode_accession(
            accession, io_cfg, window_size, stride,
            tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
            source=src,
            max_windows_per_protein=pol.get("max_windows_per_protein"),
            protein_len_min=pol.get("protein_len_min"),
            protein_len_max=pol.get("protein_len_max"),
            translation_only=bool(pol.get("translation_only", False)),
            protein_opts=protein_opts,
            save_to_disk=True, out_path=enc_path,
        )

    if encoded is None or encoded.shape[0] == 0:
        raise RuntimeError(f"score-one: no windows encoded for {accession}")

    elbo, recon_loss, kl = score_windows(model, encoded, seq_len, vocab_size, loss_type, device)
    n_windows = int(encoded.shape[0])

    record = ELBORecord(
        accession=accession,
        status="ok",
        elbo=elbo,
        recon_loss=recon_loss,
        kl=kl,
        n_windows=n_windows,
        sequence_length=n_windows * int(seq_len),
    )

    layout = ensure_run_layout(run_id=getattr(args, "run_id", None))
    out_json = path_in_run(layout, "outputs", f"{accession}_elbo.json")
    write_elbo_json(out_json, [record])
    update_run_manifest(
        layout,
        paths={"elbo_score": {accession: out_json}},
        artifacts=[_artifact(f"elbo_{accession}", "latent.elbo_score", out_json, artifact_type="json")],
    )
    print(
        f"score-one {accession}: elbo={elbo:.4f} "
        f"recon_loss={recon_loss:.4f} kl={kl:.4f} "
        f"n_windows={n_windows} -> {out_json}"
    )
    return 0


def cmd_score_batch(args: argparse.Namespace) -> int:
    from perceptrome.elbo_score import (
        ELBORecord,
        ELBOSummary,
        score_windows,
        utc_now as _elbo_now,
        write_elbo_json,
        write_elbo_tsv,
    )
    from perceptrome.encoding_main import tokenizer_meta
    from perceptrome.model import get_device, load_or_init_model

    if torch is None:
        raise RuntimeError("PyTorch is required for score-batch.")

    cfg = load_full_config(args.config)
    cfg = _apply_cli_training_overrides(cfg, args)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    io_cfg = _run_local_io_cfg(io_cfg)
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

    catalog_path = str(args.catalog)
    accessions = read_catalog(catalog_path)
    if not accessions:
        raise ValueError(f"No accessions found in catalog: {catalog_path}")

    keep_going = bool(getattr(args, "keep_going", False))

    device = get_device()
    seq_len, vocab_size = tokenizer_meta(tok, window_size)
    loss_type = "ce" if tok == "aa" else "mse"
    model, _, _, _ = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=train_cfg.hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tok,
        loss_type=loss_type,
        model_type=train_cfg.model_type,
        transformer_d_model=train_cfg.transformer_d_model,
        transformer_nhead=train_cfg.transformer_nhead,
        transformer_layers=train_cfg.transformer_layers,
        transformer_dropout=train_cfg.transformer_dropout,
        beta_kl=train_cfg.beta_kl,
        ast_tree_layers=train_cfg.ast_tree_layers,
        ast_motif_kernel_size=train_cfg.ast_motif_kernel_size,
        ast_motif_channels=train_cfg.ast_motif_channels,
        hierarchical_latent_dim=train_cfg.hierarchical_latent_dim,
        ast_node_type_vocab_size=train_cfg.ast_node_type_vocab_size,
        hierarchical_ablation_mode=train_cfg.hierarchical_ablation_mode,
    )
    model.eval()

    started_at = _elbo_now()
    records: List[ELBORecord] = []
    cache_kw = _cache_kwargs(tok, min_orf, pol)

    print(f"score-batch: scoring {len(accessions)} accessions (tok={tok} src={src})")

    for i, accession in enumerate(accessions):
        try:
            _ensure_record(accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
            enc_path = encoded_cache_path(io_cfg, accession, tok, window_size, stride, frame, source=src, **cache_kw)
            if os.path.exists(enc_path):
                encoded = np.load(enc_path)
            else:
                encoded = encode_accession(
                    accession, io_cfg, window_size, stride,
                    tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf,
                    source=src,
                    max_windows_per_protein=pol.get("max_windows_per_protein"),
                    protein_len_min=pol.get("protein_len_min"),
                    protein_len_max=pol.get("protein_len_max"),
                    translation_only=bool(pol.get("translation_only", False)),
                    protein_opts=protein_opts,
                    save_to_disk=True, out_path=enc_path,
                )
            if encoded is None or encoded.shape[0] == 0:
                records.append(ELBORecord(
                    accession=accession, status="error", elbo=0.0,
                    recon_loss=0.0, kl=0.0, n_windows=0, sequence_length=0,
                    error="no windows encoded",
                ))
                continue

            elbo, recon_loss, kl = score_windows(model, encoded, seq_len, vocab_size, loss_type, device)
            n_windows = int(encoded.shape[0])
            records.append(ELBORecord(
                accession=accession,
                status="ok",
                elbo=elbo,
                recon_loss=recon_loss,
                kl=kl,
                n_windows=n_windows,
                sequence_length=n_windows * int(seq_len),
            ))
        except Exception as err:
            logging.warning("score-batch: skipping %s: %s", accession, err)
            records.append(ELBORecord(
                accession=accession, status="error", elbo=0.0,
                recon_loss=0.0, kl=0.0, n_windows=0, sequence_length=0,
                error=str(err),
            ))
            if not keep_going:
                raise

        if (i + 1) % 25 == 0 or (i + 1) == len(accessions):
            n_ok = sum(1 for r in records if r.status == "ok")
            print(f"  [{i + 1}/{len(accessions)}] scored={n_ok} failed={len(records) - n_ok}")

    completed_at = _elbo_now()
    ok_records = [r for r in records if r.status == "ok"]
    n_ok = len(ok_records)
    n_failed = len(records) - n_ok

    mean_elbo = float(np.mean([r.elbo for r in ok_records])) if ok_records else 0.0
    mean_recon = float(np.mean([r.recon_loss for r in ok_records])) if ok_records else 0.0
    mean_kl = float(np.mean([r.kl for r in ok_records])) if ok_records else 0.0

    summary = ELBOSummary(
        run_id="",
        catalog_path=catalog_path,
        n_scored=n_ok,
        n_failed=n_failed,
        tokenizer=tok,
        model_type=str(train_cfg.model_type),
        window_size=int(window_size),
        loss_type=loss_type,
        started_at=started_at,
        completed_at=completed_at,
        mean_elbo=mean_elbo,
        mean_recon_loss=mean_recon,
        mean_kl=mean_kl,
        metadata={},
    )

    layout = ensure_run_layout(run_id=getattr(args, "run_id", None))
    out_json = path_in_run(layout, "outputs", "elbo_summary.json")
    out_tsv = path_in_run(layout, "outputs", "elbo_summary.tsv")
    from dataclasses import asdict as _asdict
    import json as _json
    summary_with_run = ELBOSummary(**{**_asdict(summary), "run_id": layout.run_id})
    write_elbo_json(out_json, records, summary=summary_with_run)
    write_elbo_tsv(out_tsv, records)
    update_run_manifest(
        layout,
        paths={"elbo_score": {"summary_json": out_json, "summary_tsv": out_tsv}},
        artifacts=[
            _artifact("elbo_summary_json", "latent.elbo_summary", out_json, artifact_type="json"),
            _artifact("elbo_summary_tsv", "latent.elbo_summary", out_tsv, artifact_type="tsv"),
        ],
    )

    # Print top-5 and bottom-5 by ELBO
    if ok_records:
        sorted_ok = sorted(ok_records, key=lambda r: r.elbo, reverse=True)
        print(f"\nTop-5 (highest ELBO — best fit to model):")
        for r in sorted_ok[:5]:
            print(f"  {r.accession:20s}  elbo={r.elbo:+.4f}  recon={r.recon_loss:.4f}  kl={r.kl:.4f}")
        if len(sorted_ok) > 5:
            print(f"\nBottom-5 (lowest ELBO — worst fit):")
            for r in sorted_ok[-5:]:
                print(f"  {r.accession:20s}  elbo={r.elbo:+.4f}  recon={r.recon_loss:.4f}  kl={r.kl:.4f}")

    print(
        f"\nscore-batch run_id={layout.run_id} scored={n_ok} failed={n_failed} "
        f"mean_elbo={mean_elbo:.4f} mean_recon={mean_recon:.4f} mean_kl={mean_kl:.4f}"
    )
    print(f"json={out_json}")
    print(f"tsv={out_tsv}")
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
