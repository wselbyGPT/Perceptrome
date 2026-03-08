from __future__ import annotations

import json
import os
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np

from perceptrome.cli.common import (
    _ensure_record,
    _get_frame,
    _get_grounded,
    _get_min_orf,
    _get_source,
    _get_tok,
    cleanup_accession_files,
    encode_accession,
    encoded_cache_path,
    ensure_dirs,
    extract_configs,
    load_full_config,
    load_state,
    read_catalog,
    save_state,
    setup_logging,
    train_on_encoded,
)
from perceptrome.encoding.parse import parse_fasta_sequence
from perceptrome.design_loop import run_design_loop
from perceptrome.generate import (
    generate_plasmid_sequence,
    generate_protein_sequence,
    parse_ast_conditioning_config,
)
from perceptrome.jobs.artifact_index import build_artifact_entry
from perceptrome.jobs.manifest_writer import config_hash, write_experiment_run_manifest
from perceptrome.pretrain import PretrainPipelineConfig, run_pretraining
from perceptrome.run_layout import ensure_run_layout, path_in_run, update_run_manifest
from perceptrome.scoring import reference_score

JobKind = Literal["train_one", "stream", "generate_plasmid", "generate_protein", "validate_plasmid", "pretrain", "design_loop"]


@dataclass(slots=True)
class JobSpec:
    kind: JobKind
    config_path: str = "config/stream_config.yaml"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobEvent:
    stage: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobResult:
    ok: bool
    exit_code: int
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    events: list[JobEvent] = field(default_factory=list)


class JobCancelledError(Exception):
    pass


class JobEngine:
    def __init__(self, event_sink: Optional[Callable[[JobEvent], None]] = None, cancel_event: Optional[threading.Event] = None):
        self._event_sink = event_sink
        self._events: list[JobEvent] = []
        self._cancel_event = cancel_event

    def _check_cancel(self) -> None:
        if self._cancel_event and self._cancel_event.is_set():
            raise JobCancelledError("Run canceled")

    def _emit(self, stage: str, message: str, **data: Any) -> None:
        if stage != "canceled":
            self._check_cancel()
        event = JobEvent(stage=stage, message=message, data=data)
        self._events.append(event)
        if self._event_sink:
            self._event_sink(event)

    @staticmethod
    def _artifact_entry(*, artifact_id: str, role: str, path: str, artifact_type: str | None = None, mime_type: str | None = None, metadata: Optional[Dict[str, Any]] = None, parents: Optional[list[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return build_artifact_entry(artifact_id=artifact_id, role=role, path=path, artifact_type=artifact_type, mime_type=mime_type, metadata=metadata, parents=parents)

    @staticmethod
    def _lineage_ref(*, path: str | None = None, artifact_id: str | None = None, relation: str | None = None) -> Dict[str, Any]:
        ref: Dict[str, Any] = {}
        if artifact_id:
            ref["artifact_id"] = str(artifact_id)
        if path:
            ref["path"] = str(path)
        if relation:
            ref["relation"] = str(relation)
        return ref

    @classmethod
    def _collect_existing_parent_refs(cls, pairs: list[tuple[str, str | None, str]]) -> list[Dict[str, Any]]:
        refs: list[Dict[str, Any]] = []
        for artifact_id, path, relation in pairs:
            if not path:
                continue
            if os.path.exists(path):
                refs.append(cls._lineage_ref(artifact_id=artifact_id, path=path, relation=relation))
        return refs

    @classmethod
    def _artifacts_for_paths(cls, mappings: Dict[str, str], role_prefix: str, artifact_type: str | None = None) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        for key, path in mappings.items():
            rows.append(cls._artifact_entry(artifact_id=key, role=f"{role_prefix}.{key}", path=path, artifact_type=artifact_type))
        return rows

    def _write_run_manifest(self, *, io_cfg: Any, spec: JobSpec, run_id: str, **sections: Any) -> str:
        path = write_experiment_run_manifest(
            logs_dir=io_cfg.logs_dir,
            experiment_id=run_id,
            run_kind=spec.kind,
            config_path=spec.config_path,
            config_hash_value=config_hash(load_full_config(spec.config_path)),
            **sections,
        )
        self._emit("manifest", "run manifest written", path=path)
        return path

    def run(self, spec: JobSpec) -> JobResult:
        self._events = []
        self._emit("start", f"Starting job kind={spec.kind}")
        try:
            self._check_cancel()
            handlers = {
                "train_one": self._run_train_one,
                "stream": self._run_stream,
                "generate_plasmid": self._run_generate_plasmid,
                "generate_protein": self._run_generate_protein,
                "validate_plasmid": self._run_validate_plasmid,
                "pretrain": self._run_pretrain,
                "design_loop": self._run_design_loop,
            }
            data = handlers[spec.kind](spec)
            self._emit("done", "Job finished successfully")
            return JobResult(ok=True, exit_code=0, message="ok", data=data, events=list(self._events))
        except JobCancelledError:
            self._emit("canceled", "Job canceled")
            return JobResult(ok=False, exit_code=130, message="canceled", data={"canceled": True}, events=list(self._events))
        except Exception as exc:  # noqa: BLE001
            self._emit("error", f"Job failed: {exc}", error=repr(exc))
            return JobResult(ok=False, exit_code=1, message=str(exc), events=list(self._events))

    @staticmethod
    def _pick_window_stride(params: Dict[str, Any], train_cfg: Any, tok: str) -> tuple[int, int]:
        ws = params.get("window_size")
        st = params.get("stride")
        if tok == "aa":
            ws = ws if ws is not None else (getattr(train_cfg, "protein_window_aa", None) or getattr(train_cfg, "window_size", None))
            st = st if st is not None else (getattr(train_cfg, "protein_stride_aa", None) or getattr(train_cfg, "stride", None))
        else:
            ws = ws if ws is not None else getattr(train_cfg, "window_size", None)
            st = st if st is not None else getattr(train_cfg, "stride", None)
        if ws is None or st is None:
            raise ValueError("window_size/stride not set")
        return int(ws), int(st)

    @staticmethod
    def _validate_tok_params(tok: str, window_size: int, stride: int, frame_offset: int) -> None:
        if tok == "codon":
            if window_size % 3 != 0 or stride % 3 != 0:
                raise ValueError("codon tokenizer requires window/stride divisible by 3")
            if frame_offset not in (0, 1, 2):
                raise ValueError("frame_offset must be 0,1,2 for codon")

    @staticmethod
    def _resolve_proteome_params(params: Dict[str, Any], train_cfg: Any, tok: str, src: str) -> Dict[str, Any]:
        args = type("A", (), params)
        return {
            "max_windows_per_protein": params.get("max_windows_per_protein"),
            "protein_len_min": params.get("protein_len_min"),
            "protein_len_max": params.get("protein_len_max"),
            "translation_only": bool(params.get("translation_only", False)),
            "protein_opts": _get_grounded(args, train_cfg, tok, src),
        }

    @staticmethod
    def _cache_kwargs(tok: str, min_orf: int, pol: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "min_orf_aa": (min_orf if tok == "aa" else None),
            "max_windows_per_protein": (pol.get("max_windows_per_protein") if tok == "aa" else None),
            "protein_len_min": (pol.get("protein_len_min") if tok == "aa" else None),
            "protein_len_max": (pol.get("protein_len_max") if tok == "aa" else None),
            "translation_only": (bool(pol.get("translation_only", False)) if tok == "aa" else False),
        }

    def _run_train_one(self, spec: JobSpec) -> Dict[str, Any]:
        cfg = load_full_config(spec.config_path)
        ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
        ensure_dirs(io_cfg)
        setup_logging(io_cfg.logs_dir)
        state = load_state(io_cfg.state_file)
        params = dict(spec.params)

        args = type("A", (), params)
        tok = _get_tok(args, train_cfg)
        frame = _get_frame(args, train_cfg)
        min_orf = _get_min_orf(args, train_cfg)
        window_size, stride = self._pick_window_stride(params, train_cfg, tok)
        self._validate_tok_params(tok, window_size, stride, frame)
        src = _get_source(args, tok)
        pol = self._resolve_proteome_params(params, train_cfg, tok, src)

        accession = str(params["accession"])
        _ensure_record(accession, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
        enc_path = encoded_cache_path(io_cfg, accession, tok, window_size, stride, frame, source=src, **self._cache_kwargs(tok, min_orf, pol))
        used_cached_encoding = os.path.exists(enc_path) and not bool(params.get("reencode", False))
        if used_cached_encoding:
            encoded = np.load(enc_path)
        else:
            encoded = encode_accession(accession, io_cfg, window_size, stride, tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf, source=src, max_windows_per_protein=pol.get("max_windows_per_protein"), protein_len_min=pol.get("protein_len_min"), protein_len_max=pol.get("protein_len_max"), translation_only=bool(pol.get("translation_only", False)), protein_opts=pol.get("protein_opts") or {}, save_to_disk=True, out_path=enc_path)
        self._emit("encode", f"encoded {accession}", path=enc_path)

        steps = int(params.get("steps") or train_cfg.steps_per_plasmid)
        batch_size = int(params.get("batch_size") or train_cfg.batch_size)
        last_total = train_on_encoded(
            accession,
            encoded,
            steps=steps,
            batch_size=batch_size,
            state=state,
            io_cfg=io_cfg,
            train_cfg=train_cfg,
            tokenizer=tok,
            window_size_bp=window_size,
            loss_type=params.get("loss_type"),
            run_id=params.get("tb_run_id"),
            tensorboard_log_every=params.get("tb_log_every"),
        )
        state["plasmid_visit_counts"][accession] = state["plasmid_visit_counts"].get(accession, 0) + 1
        save_state(io_cfg.state_file, state)
        cleanup_accession_files(accession, io_cfg, enc_path)
        self._emit("train", f"trained {accession}", loss=float(last_total))

        run_id = str(params.get("manifest_id") or f"train_one_{accession}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        run_parents = self._collect_existing_parent_refs([
            ("encoded_windows", enc_path if used_cached_encoding else None, "consumed.encoded_cache"),
        ])
        manifest_path = self._write_run_manifest(
            io_cfg=io_cfg,
            spec=spec,
            run_id=run_id,
            dataset_catalog_manifest={"accession": accession, "source": src, "encoded_path": enc_path},
            tokenizer_encoding_config={
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
            model_objective_config={
                "model_type": getattr(train_cfg, "model_type", None),
                "loss_type": params.get("loss_type"),
                "steps": steps,
                "batch_size": batch_size,
            },
            metrics={"last_total_loss": float(last_total), "encoded_shape": list(getattr(encoded, "shape", []))},
            provenance_metadata={"state_file": io_cfg.state_file},
            run_parents=run_parents,
            artifacts=[
                self._artifact_entry(
                    artifact_id="encoded_windows",
                    role="dataset.encoded",
                    path=enc_path,
                    artifact_type="npy",
                    parents=[self._lineage_ref(path=enc_path, artifact_id="encoded_windows", relation="consumed.encoded_cache")] if used_cached_encoding else None,
                ),
            ],
        )
        return {"accession": accession, "last_total_loss": float(last_total), "manifest_path": manifest_path}

    def _run_stream(self, spec: JobSpec) -> Dict[str, Any]:
        cfg = load_full_config(spec.config_path)
        ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
        ensure_dirs(io_cfg)
        setup_logging(io_cfg.logs_dir)
        state = load_state(io_cfg.state_file)
        params = dict(spec.params)
        accessions = read_catalog(str(params["catalog"]))

        args = type("A", (), params)
        tok = _get_tok(args, train_cfg)
        frame = _get_frame(args, train_cfg)
        min_orf = _get_min_orf(args, train_cfg)
        window_size, stride = self._pick_window_stride(params, train_cfg, tok)
        src = _get_source(args, tok)

        steps = int(params.get("steps_per_plasmid") or train_cfg.steps_per_plasmid)
        batch = int(params.get("batch_size") or train_cfg.batch_size)
        max_epochs = int(params.get("max_epochs") or train_cfg.max_stream_epochs)
        epoch = int(state.get("epoch", 0))
        processed = 0
        last_total = 0.0
        for_epoch_losses: list[float] = []
        run_parent_refs: list[Dict[str, Any]] = []
        while epoch < max_epochs:
            self._check_cancel()
            indices = list(range(len(accessions)))
            if train_cfg.shuffle_catalog:
                random.shuffle(indices)
            for idx in indices:
                self._check_cancel()
                acc = accessions[idx]
                pol = self._resolve_proteome_params(params, train_cfg, tok, src)
                _ensure_record(acc, src, io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=False)
                enc_path = encoded_cache_path(io_cfg, acc, tok, window_size, stride, frame, source=src, **self._cache_kwargs(tok, min_orf, pol))
                used_cached_encoding = os.path.exists(enc_path) and not bool(params.get("reencode", False))
                if used_cached_encoding:
                    encoded = np.load(enc_path)
                    run_parent_refs.append(self._lineage_ref(artifact_id=f"encoded_windows:{acc}", path=enc_path, relation="consumed.encoded_cache"))
                else:
                    encoded = encode_accession(acc, io_cfg, window_size, stride, tokenizer=tok, frame_offset=frame, min_orf_aa=min_orf, source=src, max_windows_per_protein=pol.get("max_windows_per_protein"), protein_len_min=pol.get("protein_len_min"), protein_len_max=pol.get("protein_len_max"), translation_only=bool(pol.get("translation_only", False)), protein_opts=pol.get("protein_opts") or {}, save_to_disk=True, out_path=enc_path)
                last_total = float(train_on_encoded(acc, encoded, steps=steps, batch_size=batch, state=state, io_cfg=io_cfg, train_cfg=train_cfg, tokenizer=tok, window_size_bp=window_size, loss_type=params.get("loss_type"), run_id=params.get("tb_run_id"), tensorboard_log_every=params.get("tb_log_every")))
                for_epoch_losses.append(last_total)
                state["plasmid_visit_counts"][acc] = state["plasmid_visit_counts"].get(acc, 0) + 1
                state["current_index"] = idx
                state["epoch"] = epoch
                save_state(io_cfg.state_file, state)
                if bool(params.get("delete_cache", False)):
                    cleanup_accession_files(acc, io_cfg, enc_path)
                processed += 1
                self._emit("stream_step", f"trained {acc}", epoch=epoch, loss=float(last_total))
            epoch += 1

        run_id = str(params.get("manifest_id") or f"stream_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        manifest_path = self._write_run_manifest(
            io_cfg=io_cfg,
            spec=spec,
            run_id=run_id,
            dataset_catalog_manifest={"catalog": str(params["catalog"]), "source": src, "num_accessions": len(accessions)},
            tokenizer_encoding_config={
                "tokenizer": tok,
                "window_size": int(window_size),
                "stride": int(stride),
                "frame_offset": int(frame),
                "min_orf_aa": int(min_orf),
            },
            model_objective_config={"model_type": getattr(train_cfg, "model_type", None), "loss_type": params.get("loss_type")},
            metrics={"processed_accessions": processed, "last_total_loss": float(last_total), "losses": for_epoch_losses},
            provenance_metadata={"state_file": io_cfg.state_file, "max_epochs": max_epochs},
            run_parents=run_parent_refs,
            artifacts=[self._artifact_entry(artifact_id="stream_state", role="provenance.state", path=io_cfg.state_file, artifact_type="json")],
        )
        return {"processed_accessions": processed, "manifest_path": manifest_path}

    def _run_generate_plasmid(self, spec: JobSpec) -> Dict[str, Any]:
        cfg = load_full_config(spec.config_path)
        _, train_cfg, io_cfg = extract_configs(cfg)
        ensure_dirs(io_cfg)
        setup_logging(io_cfg.logs_dir)
        p = dict(spec.params)
        tokenizer = str(p.get("tokenizer") or getattr(train_cfg, "tokenizer", "base"))
        ast_conditioning = parse_ast_conditioning_config(
            ast_artifact=p.get("ast_artifact"),
            ast_node_type_prompt=p.get("ast_node_type_prompt"),
            ast_region_span=p.get("ast_region_span"),
            ast_graph_mask=p.get("ast_graph_mask"),
            ast_graph_hop_limit=int(p.get("ast_graph_hop_limit", 1)),
            ast_mask_strength=float(p.get("ast_mask_strength", 0.0)),
        )
        layout = ensure_run_layout()
        output = path_in_run(layout, "outputs", os.path.basename(str(p.get("output", "generated/novel_plasmid.fasta"))))
        checkpoint_path = os.path.join(str(getattr(io_cfg, "checkpoints_dir", "") or ""), "latest.pt")
        ast_artifact_path = str(getattr(ast_conditioning, "artifact_path", "") or "")
        run_parents = self._collect_existing_parent_refs([
            ("model.checkpoint", checkpoint_path, "consumed.checkpoint"),
            ("conditioning.ast", ast_artifact_path, "consumed.ast_artifact"),
        ])
        seq = generate_plasmid_sequence(train_cfg=train_cfg, io_cfg=io_cfg, length_bp=int(p.get("length_bp", 10000)), num_windows=p.get("num_windows"), window_size_bp=int(p.get("window_size") or train_cfg.window_size), seed=p.get("seed"), latent_scale=float(p.get("latent_scale", 1.0)), temperature=float(p.get("temperature", 1.0)), gc_bias=float(p.get("gc_bias", 1.0)), num_candidates=int(p.get("num_candidates", 1)), top_k=int(p.get("top_k", 1)), target_gc=float(p.get("target_gc", 0.5)), max_homopolymer=p.get("max_homopolymer"), summary_path=p.get("summary_path"), top_k_output_path=p.get("top_k_output"), roundtrip_score=bool(p.get("roundtrip_score", False)), recon_weight=float(p.get("recon_weight", 0.1)), name=str(p.get("name", "perceptrome_plasmid_1")), output_path=output, tokenizer=tokenizer, provenance_inputs={"config": str(spec.config_path)}, ast_conditioning=ast_conditioning)
        self._emit("generate", "plasmid generated", output=output, length=len(seq))
        run_id = str(p.get("manifest_id") or f"generate_plasmid_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        manifest_path = self._write_run_manifest(
            io_cfg=io_cfg,
            spec=spec,
            run_id=run_id,
            tokenizer_encoding_config={"tokenizer": tokenizer, "window_size": int(p.get("window_size") or train_cfg.window_size)},
            model_objective_config={"model_type": getattr(train_cfg, "model_type", None)},
            generated_sequences={
                "entries": [
                    {
                        "type": "plasmid",
                        "name": str(p.get("name", "perceptrome_plasmid_1")),
                        "output": output,
                        "summary_json": path_in_run(layout, "outputs", f"{os.path.basename(output)}.summary.json"),
                        "top_k_output": path_in_run(layout, "outputs", os.path.basename(str(p.get("top_k_output") or f"{os.path.basename(output)}.top{int(p.get('top_k', 1))}.fasta"))),
                        "length_bp": len(seq),
                        "top_k": int(p.get("top_k", 1)),
                        "num_candidates": int(p.get("num_candidates", 1)),
                        "tokenizer": tokenizer,
                    }
                ]
            },
            training_metrics={"generate_plasmid": {"length_bp": len(seq)}},
            metrics={"length_bp": len(seq)},
            run_parents=run_parents,
            run_children=[self._lineage_ref(artifact_id="generated_plasmid", path=output, relation="produced.generated_sequence")],
            artifacts=[
                self._artifact_entry(
                    artifact_id="generated_plasmid",
                    role="generated.sequence",
                    path=output,
                    artifact_type="fasta",
                    parents=run_parents,
                )
            ],
        )
        update_run_manifest(layout, paths={"generated": {"plasmid_fasta": output, "manifest": manifest_path}}, artifacts=[self._artifact_entry(artifact_id="generated_plasmid", role="generated.sequence", path=output, artifact_type="fasta", parents=run_parents)])
        return {"output": output, "length": len(seq), "manifest_path": manifest_path}

    def _run_generate_protein(self, spec: JobSpec) -> Dict[str, Any]:
        cfg = load_full_config(spec.config_path)
        _, train_cfg, io_cfg = extract_configs(cfg)
        ensure_dirs(io_cfg)
        setup_logging(io_cfg.logs_dir)
        p = dict(spec.params)
        ast_conditioning = parse_ast_conditioning_config(
            ast_artifact=p.get("ast_artifact"),
            ast_node_type_prompt=p.get("ast_node_type_prompt"),
            ast_region_span=p.get("ast_region_span"),
            ast_graph_mask=p.get("ast_graph_mask"),
            ast_graph_hop_limit=int(p.get("ast_graph_hop_limit", 1)),
            ast_mask_strength=float(p.get("ast_mask_strength", 0.0)),
        )
        layout = ensure_run_layout()
        output = path_in_run(layout, "outputs", os.path.basename(str(p.get("output", "generated/novel_protein.faa"))))
        checkpoint_path = os.path.join(str(getattr(io_cfg, "checkpoints_dir", "") or ""), "latest.pt")
        ast_artifact_path = str(getattr(ast_conditioning, "artifact_path", "") or "")
        run_parents = self._collect_existing_parent_refs([
            ("model.checkpoint", checkpoint_path, "consumed.checkpoint"),
            ("conditioning.ast", ast_artifact_path, "consumed.ast_artifact"),
        ])
        seq = generate_protein_sequence(train_cfg=train_cfg, io_cfg=io_cfg, length_aa=int(p.get("length_aa", 600)), num_windows=p.get("num_windows"), window_aa=int(p.get("window_aa") or train_cfg.protein_window_aa), seed=p.get("seed"), latent_scale=float(p.get("latent_scale", 1.0)), temperature=float(p.get("temperature", 1.0)), name=str(p.get("name", "perceptrome_protein_1")), output_path=output, reject=bool(p.get("reject", False)), reject_tries=int(p.get("reject_tries", 40)), reject_max_run=int(p.get("reject_max_run", 10)), reject_max_x_frac=float(p.get("reject_max_x_frac", 0.15)), num_candidates=int(p.get("num_candidates", 1)), top_k=int(p.get("top_k", 1)), max_homopolymer=p.get("max_homopolymer"), max_x_frac=p.get("max_x_frac"), max_internal_stops=int(p.get("max_internal_stops", 0)), summary_path=p.get("summary_path"), top_k_output_path=p.get("top_k_output"), roundtrip_score=bool(p.get("roundtrip_score", False)), recon_weight=float(p.get("recon_weight", 0.1)), provenance_inputs={"config": str(spec.config_path)}, ast_conditioning=ast_conditioning)
        self._emit("generate", "protein generated", output=output, length=len(seq))
        run_id = str(p.get("manifest_id") or f"generate_protein_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        manifest_path = self._write_run_manifest(
            io_cfg=io_cfg,
            spec=spec,
            run_id=run_id,
            tokenizer_encoding_config={"tokenizer": "aa", "window_size": int(p.get("window_aa") or train_cfg.protein_window_aa)},
            model_objective_config={"model_type": getattr(train_cfg, "model_type", None)},
            generated_sequences={
                "entries": [
                    {
                        "type": "protein",
                        "name": str(p.get("name", "perceptrome_protein_1")),
                        "output": output,
                        "summary_json": path_in_run(layout, "outputs", f"{os.path.basename(output)}.summary.json"),
                        "top_k_output": path_in_run(layout, "outputs", os.path.basename(str(p.get("top_k_output") or f"{os.path.basename(output)}.top{int(p.get('top_k', 1))}.fasta"))),
                        "length_aa": len(seq),
                        "top_k": int(p.get("top_k", 1)),
                        "num_candidates": int(p.get("num_candidates", 1)),
                        "tokenizer": "aa",
                    }
                ]
            },
            training_metrics={"generate_protein": {"length_aa": len(seq)}},
            metrics={"length_aa": len(seq)},
            run_parents=run_parents,
            run_children=[self._lineage_ref(artifact_id="generated_protein", path=output, relation="produced.generated_sequence")],
            artifacts=[
                self._artifact_entry(
                    artifact_id="generated_protein",
                    role="generated.sequence",
                    path=output,
                    artifact_type="fasta",
                    parents=run_parents,
                )
            ],
        )
        update_run_manifest(layout, paths={"generated": {"protein_faa": output, "manifest": manifest_path}}, artifacts=[self._artifact_entry(artifact_id="generated_protein", role="generated.sequence", path=output, artifact_type="fasta", parents=run_parents)])
        return {"output": output, "length": len(seq), "manifest_path": manifest_path}

    def _run_validate_plasmid(self, spec: JobSpec) -> Dict[str, Any]:
        cfg = load_full_config(spec.config_path)
        ncbi_cfg, _, io_cfg = extract_configs(cfg)
        ensure_dirs(io_cfg)
        p = dict(spec.params)
        generated_fasta = str(p["generated_fasta"])
        catalog_path = str(p["catalog"])
        generated_seq = parse_fasta_sequence(generated_fasta)
        rows = []
        for accession in read_catalog(catalog_path):
            ref_path = _ensure_record(accession, "fasta", io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=bool(p.get("force_fetch", False)))
            ref_seq = parse_fasta_sequence(ref_path)
            row_scores = reference_score(generated_seq, ref_seq)
            rows.append({
                "accession": accession,
                "ref_path": ref_path,
                "ref_len": len(ref_seq),
                **row_scores,
            })
        rows.sort(key=lambda r: r["score"], reverse=True)
        top_n = max(1, int(p.get("top_n", 5)))
        top_rows = rows[:top_n]
        layout = ensure_run_layout()
        output_json = p.get("output_json")
        payload = {"generated_fasta": generated_fasta, "catalog": catalog_path, "top_n": top_n, "results": top_rows}
        out_path = path_in_run(layout, "outputs", os.path.basename(str(output_json or "validation.json")))
        machine_json = path_in_run(layout, "artifacts", "validation_results.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        with open(machine_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        if output_json:
            user_out = str(output_json)
            os.makedirs(os.path.dirname(user_out) or ".", exist_ok=True)
            with open(user_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
        self._emit("validate", "validation complete", top_n=top_n)

        run_id = str(p.get("manifest_id") or f"validate_plasmid_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        manifest_path = self._write_run_manifest(
            io_cfg=io_cfg,
            spec=spec,
            run_id=run_id,
            dataset_catalog_manifest={"catalog": catalog_path, "generated_fasta": generated_fasta},
            validation_results={
                "top_n": top_n,
                "results": top_rows,
                "output_json": out_path,
                "machine_artifact_json": machine_json,
            },
            training_metrics={"validate_plasmid": {"evaluated": len(rows), "top_score": float(top_rows[0]["score"]) if top_rows else None}},
            metrics={"evaluated": len(rows), "top_score": float(top_rows[0]["score"]) if top_rows else None},
            artifacts=self._artifacts_for_paths({"report_json": out_path, "machine_report_json": machine_json}, "validation", artifact_type="json") + ([self._artifact_entry(artifact_id="user_report_json", role="validation.user_report", path=str(output_json), artifact_type="json")] if output_json else []),
        )
        update_run_manifest(
            layout,
            paths={"validation": {"report_json": out_path, "machine_report_json": machine_json, "manifest": manifest_path}},
            validation_results={"validate_plasmid": {"top_n": top_n, "output_json": out_path, "machine_artifact_json": machine_json}},
            artifacts=self._artifacts_for_paths({"report_json": out_path, "machine_report_json": machine_json}, "validation", artifact_type="json") + ([self._artifact_entry(artifact_id="user_report_json", role="validation.user_report", path=str(output_json), artifact_type="json")] if output_json else []),
        )
        return {"results": top_rows, "top_n": top_n, "manifest_path": manifest_path, "output_json": out_path, "machine_artifact_json": machine_json}

    def _run_pretrain(self, spec: JobSpec) -> Dict[str, Any]:
        cfg = load_full_config(spec.config_path)
        pre_cfg = dict(cfg.get("pretrain", {}) or {})
        _, _, io_cfg = extract_configs(cfg)
        ensure_dirs(io_cfg)
        setup_logging(io_cfg.logs_dir)
        p = dict(spec.params)
        dataset_path = str(p.get("dataset") or pre_cfg.get("dataset_path", "")).strip()
        if not dataset_path:
            raise ValueError("Pretraining requires --dataset")
        pipeline_cfg = PretrainPipelineConfig(
            dataset_path=dataset_path,
            vocab_size=int(p.get("vocab_size") or pre_cfg.get("vocab_size", 32)),
            batch_size=int(p.get("batch_size") or pre_cfg.get("batch_size", 16)),
            epochs=int(p.get("epochs") or pre_cfg.get("epochs", 1)),
            hidden_size=int(p.get("hidden_size") or pre_cfg.get("hidden_size", 256)),
            lr=float(p.get("learning_rate") or pre_cfg.get("learning_rate", 1e-4)),
            output_dir=path_in_run(ensure_run_layout(), "artifacts", os.path.basename(str(p.get("output_dir") or pre_cfg.get("output_dir", "model/pretrain")))),
            enable_mlm=bool(pre_cfg.get("enable_mlm", True)) if not p.get("disable_mlm", False) else False,
            enable_sme=bool(pre_cfg.get("enable_sme", True)) if not p.get("disable_sme", False) else False,
            enable_contrastive=bool(pre_cfg.get("enable_contrastive", True)) if not p.get("disable_contrastive", False) else False,
            seed=p.get("seed") if p.get("seed") is not None else pre_cfg.get("seed"),
            provenance_inputs={"config": str(spec.config_path), "dataset": str(dataset_path)},
        )
        metrics = run_pretraining(pipeline_cfg)
        self._emit("pretrain", "pretraining complete", metrics=metrics)

        run_id = str(p.get("manifest_id") or f"pretrain_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        manifest_path = self._write_run_manifest(
            io_cfg=io_cfg,
            spec=spec,
            run_id=run_id,
            dataset_catalog_manifest={"dataset_path": dataset_path},
            tokenizer_encoding_config={"vocab_size": int(pipeline_cfg.vocab_size)},
            model_objective_config={
                "model": {"hidden_size": int(pipeline_cfg.hidden_size)},
                "objective": {
                    "enable_mlm": bool(pipeline_cfg.enable_mlm),
                    "enable_sme": bool(pipeline_cfg.enable_sme),
                    "enable_contrastive": bool(pipeline_cfg.enable_contrastive),
                },
            },
            checkpoints={"output_dir": str(pipeline_cfg.output_dir), "index_json": str(metrics.get("checkpoint_index_json", ""))},
            pretraining_metrics=dict(metrics),
            metrics=dict(metrics),
            artifacts=self._artifacts_for_paths({"checkpoint_index_json": str(metrics.get("checkpoint_index_json", "")), "final_metrics_json": str(metrics.get("final_metrics_json", ""))}, "pretrain", artifact_type="json"),
        )
        return {"metrics": metrics, "completed_at": datetime.now(timezone.utc).isoformat(), "manifest_path": manifest_path}

    def _run_design_loop(self, spec: JobSpec) -> Dict[str, Any]:
        cfg = load_full_config(spec.config_path)
        ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
        ensure_dirs(io_cfg)
        setup_logging(io_cfg.logs_dir)
        p = dict(spec.params)
        ast_conditioning = parse_ast_conditioning_config(
            ast_artifact=p.get("ast_artifact"),
            ast_node_type_prompt=p.get("ast_node_type_prompt"),
            ast_region_span=p.get("ast_region_span"),
            ast_graph_mask=p.get("ast_graph_mask"),
            ast_graph_hop_limit=int(p.get("ast_graph_hop_limit", 1)),
            ast_mask_strength=float(p.get("ast_mask_strength", 0.0)),
        )

        layout = ensure_run_layout()
        catalog_path = str(p["catalog"])
        checkpoint_path = os.path.join(str(getattr(io_cfg, "checkpoints_dir", "") or ""), "latest.pt")
        ast_artifact_path = str(getattr(ast_conditioning, "artifact_path", "") or "")
        run_parents = self._collect_existing_parent_refs([
            ("model.checkpoint", checkpoint_path, "consumed.checkpoint"),
            ("conditioning.ast", ast_artifact_path, "consumed.ast_artifact"),
            ("dataset.catalog", catalog_path, "consumed.catalog"),
        ])
        references = []
        for accession in read_catalog(catalog_path):
            ref_path = _ensure_record(accession, "fasta", io_cfg=io_cfg, ncbi_cfg=ncbi_cfg, force=bool(p.get("force_fetch", False)))
            references.append(parse_fasta_sequence(ref_path))

        result = run_design_loop(
            train_cfg=train_cfg,
            io_cfg=io_cfg,
            rounds=int(p.get("rounds", p.get("generations", 5))),
            population_size=int(p.get("population_size", 8)),
            survivor_count=int(p.get("survivor_count", 3)),
            mutation_rate=float(p.get("mutation_rate", 0.05)),
            mutation_scale=float(p.get("mutation_scale", 1.0)),
            crossover_rate=float(p.get("crossover_rate", 0.5)),
            early_stop_threshold=(None if p.get("early_stop_threshold") is None else float(p.get("early_stop_threshold"))),
            target_gc=float(p.get("target_gc", 0.5)),
            max_homopolymer=int(p.get("max_homopolymer", 8)),
            length_bp=int(p.get("length_bp", 4000)),
            references=references,
            seed=p.get("seed"),
            enable_sequence_operators=bool(p.get("enable_sequence_operators", False)),
            sequence_operator_top_k=int(p.get("sequence_operator_top_k", 3)),
            emit=lambda stage, message: self._emit(stage, message),
            ast_conditioning=ast_conditioning,
        )

        best = result.get("best_candidate", {})
        best_fasta = path_in_run(layout, "outputs", os.path.basename(str(p.get("output") or "design_loop_best.fasta")))
        with open(best_fasta, "w", encoding="utf-8") as f:
            f.write(f">{str(p.get('name', 'design_loop_best'))}|score={float(best.get('score', 0.0)):.6f}\n")
            seq = str(best.get("sequence", ""))
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")

        summary_json = path_in_run(layout, "artifacts", "design_loop_summary.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")

        evolution_history = dict(result.get("evolution_history") or {})
        lineage_generations = list(evolution_history.get("generations") or [])
        lineage_path = path_in_run(layout, "artifacts", "evolution_lineage.json")
        summary_rows = list((result.get("best_candidates") or []))
        for generation in lineage_generations:
            for row_idx, candidate in enumerate(generation.get("candidates") or []):
                candidate["artifact_paths"] = {
                    "fasta": best_fasta if int(candidate.get("candidate_id", -1)) == int(best.get("candidate_id", -1)) else None,
                    "summary_row": f"summary_rows[generation={int(generation.get('generation_index', 0))}][row={row_idx}]",
                }
        lineage_payload = {
            "run_id": layout.run_id,
            "lineage_version": 1,
            "summary_rows": summary_rows,
            "evolution_history": {
                "generations": lineage_generations,
            },
        }
        with open(lineage_path, "w", encoding="utf-8") as f:
            json.dump(lineage_payload, f, indent=2)
            f.write("\n")

        self._emit("design_loop", "design loop complete", rounds=result.get("rounds_completed"), best_score=best.get("score"))

        run_id = str(p.get("manifest_id") or f"design_loop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        manifest_path = self._write_run_manifest(
            io_cfg=io_cfg,
            spec=spec,
            run_id=run_id,
            dataset_catalog_manifest={"catalog": catalog_path, "references": len(references)},
            generated_sequences={
                "entries": [
                    {
                        "type": "plasmid",
                        "name": str(p.get("name", "design_loop_best")),
                        "output": best_fasta,
                        "length_bp": len(str(best.get("sequence", ""))),
                        "score": float(best.get("score", 0.0)),
                    }
                ],
                "lineage_artifact": lineage_path,
            },
            validation_results={
                "design_loop": {
                    "best_metrics": best.get("metrics", {}),
                    "rounds_completed": result.get("rounds_completed"),
                    "evolution_lineage": lineage_path,
                }
            },
            evolution_history={"generations": lineage_generations},
            training_metrics={"design_loop": {"rounds_completed": result.get("rounds_completed"), "best_score": float(best.get("score", 0.0))}},
            metrics={"rounds_completed": result.get("rounds_completed"), "best_score": float(best.get("score", 0.0))},
            provenance_metadata={"summary_json": summary_json},
            run_parents=run_parents,
            run_children=[self._lineage_ref(artifact_id="best_fasta", path=best_fasta, relation="produced.generated_sequence")],
            artifacts=[
                self._artifact_entry(artifact_id="best_fasta", role="design_loop.best_fasta", path=best_fasta, parents=run_parents),
                self._artifact_entry(artifact_id="summary_json", role="design_loop.summary_json", path=summary_json),
                self._artifact_entry(artifact_id="evolution_lineage", role="design_loop.evolution_lineage", path=lineage_path),
            ],
        )
        update_run_manifest(
            layout,
            paths={
                "design_loop": {
                    "best_fasta": best_fasta,
                    "summary_json": summary_json,
                    "evolution_lineage": lineage_path,
                    "manifest": manifest_path,
                }
            },
            metrics={"design_loop": {"best_score": float(best.get("score", 0.0)), "rounds_completed": result.get("rounds_completed")}},
            generated_sequences={"design_loop": {"best_fasta": best_fasta, "evolution_lineage": lineage_path}},
            validation_results={"design_loop": {"summary_json": summary_json, "evolution_lineage": lineage_path}},
            evolution_history={"generations": lineage_generations},
            artifacts=[
                self._artifact_entry(artifact_id="best_fasta", role="design_loop.best_fasta", path=best_fasta, parents=run_parents),
                self._artifact_entry(artifact_id="summary_json", role="design_loop.summary_json", path=summary_json),
                self._artifact_entry(artifact_id="evolution_lineage", role="design_loop.evolution_lineage", path=lineage_path),
            ],
        )
        return {
            "best_candidate": best,
            "best_fasta": best_fasta,
            "summary_json": summary_json,
            "lineage_json": lineage_path,
            "manifest_path": manifest_path,
            "rounds_completed": result.get("rounds_completed"),
        }
