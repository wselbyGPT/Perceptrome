import csv
import json
import logging, os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .config import TrainingConfig, IOConfig
from .model import get_device, load_or_init_model
from .encoding_main import tokenizer_meta, IDX_TO_CODON, CODON_VOCAB_SIZE, GC_COUNT_PER_TOKEN, IDX_TO_AA, AA_VOCAB_SIZE
from .jobs.provenance import collect_and_write_provenance, resolve_seed, set_global_seeds
from .run_layout import ensure_run_layout, path_in_run, update_run_manifest
from .scorecard import (
    build_plasmid_scorecard,
    build_protein_scorecard,
    gc_fraction as scorecard_gc_fraction,
    max_homopolymer_run as scorecard_max_homopolymer_run,
)


def _run_local_io_cfg(io_cfg: IOConfig) -> IOConfig:
    layout = ensure_run_layout()
    return IOConfig(
        cache_fasta_dir=io_cfg.cache_fasta_dir,
        cache_genbank_dir=io_cfg.cache_genbank_dir,
        cache_encoded_dir=io_cfg.cache_encoded_dir,
        model_dir=layout.artifacts_dir,
        checkpoints_dir=os.path.join(layout.artifacts_dir, "checkpoints"),
        logs_dir=io_cfg.logs_dir,
        state_file=io_cfg.state_file,
    )



@dataclass(frozen=True)
class AstConditioningConfig:
    artifact_path: Optional[str] = None
    node_type_prompts: Tuple[str, ...] = ()
    region_spans: Tuple[Tuple[int, int], ...] = ()
    graph_mask: str = "none"
    graph_hop_limit: int = 1
    mask_strength: float = 0.0


def _parse_region_span(span: str) -> Tuple[int, int]:
    raw = str(span or "").strip()
    if ":" not in raw:
        raise ValueError(f"Invalid AST region span '{span}'. Expected START:END")
    a, b = raw.split(":", 1)
    start = int(a)
    end = int(b)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid AST region span '{span}'. Expected 0 <= START < END")
    return (start, end)


def parse_ast_conditioning_config(
    *,
    ast_artifact: Optional[str],
    ast_node_type_prompt: Optional[Sequence[str]],
    ast_region_span: Optional[Sequence[str]],
    ast_graph_mask: Optional[str],
    ast_graph_hop_limit: int,
    ast_mask_strength: float,
) -> Optional[AstConditioningConfig]:
    has_inputs = any(
        [
            ast_artifact,
            ast_node_type_prompt,
            ast_region_span,
            ast_graph_mask not in (None, "none"),
            float(ast_mask_strength or 0.0) > 0.0,
        ]
    )
    if not has_inputs:
        return None

    node_prompts = tuple(sorted({str(v).strip().lower() for v in (ast_node_type_prompt or []) if str(v).strip()}))
    spans = tuple(_parse_region_span(v) for v in (ast_region_span or []))
    graph_mask = str(ast_graph_mask or "none").strip().lower()
    if graph_mask not in {"none", "neighbors", "successors"}:
        raise ValueError(f"Unsupported --ast-graph-mask: {graph_mask}")

    mask_strength = max(0.0, min(1.0, float(ast_mask_strength or 0.0)))
    hop_limit = max(1, int(ast_graph_hop_limit or 1))

    return AstConditioningConfig(
        artifact_path=(str(ast_artifact) if ast_artifact else None),
        node_type_prompts=node_prompts,
        region_spans=spans,
        graph_mask=graph_mask,
        graph_hop_limit=hop_limit,
        mask_strength=mask_strength,
    )


def _load_ast_conditioning_payload(cfg: AstConditioningConfig) -> Dict[str, Any]:
    artifact_payload: Dict[str, Any] = {}
    if cfg.artifact_path:
        with open(cfg.artifact_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            artifact_payload = loaded

    node_counts: Dict[str, int] = {}
    for node in artifact_payload.get("nodes", []) if isinstance(artifact_payload.get("nodes"), list) else []:
        if not isinstance(node, dict):
            continue
        ntype = str(node.get("node_type", "")).strip().lower()
        if not ntype:
            continue
        node_counts[ntype] = int(node_counts.get(ntype, 0) + 1)

    edges = artifact_payload.get("edges")
    edge_count = len(edges) if isinstance(edges, list) else 0

    return {
        "artifact": artifact_payload,
        "node_type_counts": node_counts,
        "edge_count": int(edge_count),
    }


def _build_ast_guided_mask(vocab_size: int, conditioning: Optional[AstConditioningConfig], details: Optional[Dict[str, Any]] = None) -> np.ndarray:
    weights = np.ones((vocab_size,), dtype=np.float64)
    if conditioning is None or conditioning.mask_strength <= 0.0:
        return weights

    node_counts = dict((details or {}).get("node_type_counts") or {})
    total_nodes = float(sum(max(0, int(v)) for v in node_counts.values()))
    prompt_mass = float(sum(max(0, int(node_counts.get(p, 0))) for p in conditioning.node_type_prompts))
    prompt_ratio = (prompt_mass / total_nodes) if total_nodes > 0.0 else 0.0

    span_bp = sum(max(0, int(e) - int(s)) for s, e in conditioning.region_spans)
    span_factor = min(1.0, span_bp / 1000.0) if span_bp > 0 else 0.0

    graph_factor = 0.0
    if conditioning.graph_mask != "none":
        edges = float((details or {}).get("edge_count") or 0.0)
        graph_factor = min(1.0, (edges / 100.0) * min(3, conditioning.graph_hop_limit) / 3.0)

    emphasis = max(0.0, min(1.0, (0.5 * prompt_ratio) + (0.3 * span_factor) + (0.2 * graph_factor)))
    if emphasis <= 0.0:
        return weights

    gc_target = min(0.95, max(0.05, 0.5 + ((emphasis - 0.5) * conditioning.mask_strength * 0.6)))
    for idx in range(vocab_size):
        gc_ratio = 0.0
        if vocab_size == 4:
            gc_ratio = 1.0 if idx in (1, 2) else 0.0
        elif 0 <= idx < len(GC_COUNT_PER_TOKEN):
            gc_ratio = float(GC_COUNT_PER_TOKEN[idx]) / 3.0
        gc_delta = gc_ratio - 0.5
        weights[idx] = max(1e-6, 1.0 + (2.0 * conditioning.mask_strength * gc_delta * (gc_target - 0.5) * 2.0))
    return weights


def ast_conditioning_metadata(conditioning: Optional[AstConditioningConfig], details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if conditioning is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "artifact_path": conditioning.artifact_path,
        "node_type_prompts": list(conditioning.node_type_prompts),
        "region_spans": [[int(s), int(e)] for s, e in conditioning.region_spans],
        "graph_guided_mask": {
            "mode": conditioning.graph_mask,
            "hop_limit": int(conditioning.graph_hop_limit),
            "mask_strength": float(conditioning.mask_strength),
            "artifact_edge_count": int((details or {}).get("edge_count", 0)),
        },
        "node_type_counts": dict((details or {}).get("node_type_counts") or {}),
    }

def _sample_from_logits(logits: np.ndarray, temperature: float) -> int:
    """Sample an index from a logits vector using softmax( logits / T )."""
    x = logits.astype(np.float64)
    T = max(1e-3, float(temperature))
    x = x / T
    x = x - np.max(x)
    w = np.exp(x)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return int(np.random.randint(0, w.shape[0]))
    w /= s
    return int(np.random.choice(w.shape[0], p=w))


def _passes_protein_filters(seq: str, max_run: int, max_x_frac: float) -> bool:
    if not seq:
        return False
    if max_x_frac is not None and max_x_frac >= 0:
        xf = seq.count("X") / float(len(seq))
        if xf > float(max_x_frac):
            return False
    if max_run is not None and max_run > 0:
        run = 1
        best = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                run += 1
                if run > best:
                    best = run
            else:
                run = 1
        if best > int(max_run):
            return False
    return True




def _max_homopolymer_run(seq: str) -> int:
    return scorecard_max_homopolymer_run(seq)


def _gc_fraction(seq: str) -> float:
    return scorecard_gc_fraction(seq)


def _plasmid_candidate_score(
    *,
    gc_dev: float,
    homopolymer_run: int,
    max_homopolymer: Optional[int],
    recon: Optional[float],
    recon_weight: float,
) -> Tuple[float, float]:
    run_pen = 0.0 if max_homopolymer is None else max(0, homopolymer_run - max_homopolymer) / max(1.0, float(max_homopolymer))
    score = -float(gc_dev) - float(run_pen) - (float(recon_weight) * float(recon) if recon is not None else 0.0)
    return float(score), float(run_pen)


def _protein_candidate_score(
    *,
    seq: str,
    max_homopolymer: int,
    max_x_frac: float,
    max_internal_stops: int,
    recon: Optional[float],
    recon_weight: float,
    allowed: Optional[set] = None,
) -> Dict[str, float]:
    scorecard = build_protein_scorecard(
        seq,
        {
            "max_homopolymer": max_homopolymer,
            "max_x_frac": max_x_frac,
            "max_internal_stops": max_internal_stops,
            "roundtrip_recon": recon,
            "recon_weight": recon_weight,
            "allowed": allowed,
        },
    )
    metrics = scorecard["metrics"]
    return {
        "score": float(metrics["score"]),
        "max_homopolymer": float(metrics["max_homopolymer"]),
        "x_fraction": float(metrics["x_fraction"]),
        "invalid_fraction": float(metrics["invalid_fraction"]),
        "stop_count": float(metrics["stop_count"]),
    }

def _make_out_paths(output_path: str, summary_path: Optional[str]) -> Tuple[str, str]:
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    summary_json = summary_path if summary_path else f"{output_path}.summary.json"
    summary_csv = f"{summary_json}.csv" if not summary_json.endswith(".csv") else summary_json
    return summary_json, summary_csv


def _write_candidate_summary(summary_json: str, summary_csv: str, payload: Dict[str, object], rows: List[Dict[str, object]]) -> None:
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_top_k_fasta(path: str, name: str, ranked: List[Dict[str, object]], top_k: int) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rank, cand in enumerate(ranked[:top_k], start=1):
            seq = str(cand.get("sequence", ""))
            idx = cand.get("candidate", rank - 1)
            score = cand.get("score", 0.0)
            f.write(f">{name}|rank={rank}|candidate={idx}|score={float(score):.6f}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")


def _roundtrip_recon_score(model, seq: str, tok: str, seq_len: int, vocab_size: int, device: "torch.device") -> Optional[float]:
    if torch is None:
        return None
    tok = tok.lower()
    if tok == "base":
        map_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        tokens = [map_idx.get(ch, 0) for ch in seq[:seq_len]]
    elif tok == "codon":
        codons = [seq[i:i + 3] for i in range(0, min(len(seq), seq_len * 3), 3)]
        codon_to_idx = {c: i for i, c in enumerate(IDX_TO_CODON)}
        tokens = [codon_to_idx.get(c, CODON_VOCAB_SIZE - 1) for c in codons[:seq_len]]
    else:
        aa_to_idx = {a: i for i, a in enumerate(IDX_TO_AA)}
        tokens = [aa_to_idx.get(ch, AA_VOCAB_SIZE - 1) for ch in seq[:seq_len]]

    if len(tokens) < seq_len:
        pad = (vocab_size - 1) if tok in ("codon", "aa") else 0
        tokens.extend([pad] * (seq_len - len(tokens)))

    x = torch.zeros((1, seq_len, vocab_size), dtype=torch.float32, device=device)
    for j, idx in enumerate(tokens[:seq_len]):
        x[0, j, int(idx)] = 1.0

    with torch.no_grad():
        logits_flat = model.decode(model.encode(x.view(1, -1))[0])
        logits = logits_flat.view(1, seq_len, vocab_size)
        if tok == "aa":
            import torch.nn.functional as F
            targets = torch.tensor(tokens[:seq_len], dtype=torch.long, device=device).view(1, seq_len)
            ce = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction="mean")
            return float(ce.item())
        recon = torch.sigmoid(logits)
        mse = (recon - x).pow(2).mean()
        return float(mse.item())

def generate_plasmid_sequence(
    train_cfg: TrainingConfig,
    io_cfg: IOConfig,
    length_bp: int,
    num_windows: Optional[int],
    window_size_bp: int,
    seed: Optional[int],
    latent_scale: float,
    temperature: float,
    gc_bias: float,
    name: str,
    output_path: str,
    tokenizer: str,
    num_candidates: int = 1,
    top_k: int = 1,
    target_gc: Optional[float] = None,
    max_homopolymer: Optional[int] = None,
    summary_path: Optional[str] = None,
    top_k_output_path: Optional[str] = None,
    roundtrip_score: bool = False,
    recon_weight: float = 0.1,
    provenance_inputs: Optional[Dict[str, str]] = None,
    ast_conditioning: Optional[AstConditioningConfig] = None,
    scorecard_reference_neighbors: Optional[Sequence[Dict[str, str]]] = None,
    scorecard_reference_top_n: int = 5,
    scorecard_motifs: Optional[Dict[str, str]] = None,
) -> str:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    tok = tokenizer.lower()
    if tok not in ("base", "codon"):
        raise ValueError("generate_plasmid_sequence only supports base|codon")

    if tok == "codon" and length_bp % 3 != 0:
        length_bp = (length_bp // 3) * 3

    seed_info = resolve_seed(seed)
    set_global_seeds(int(seed_info["value"]))

    layout = ensure_run_layout()
    collect_and_write_provenance(
        layout=layout,
        run_kind="generate_plasmid",
        seed_info=seed_info,
        input_paths=provenance_inputs or {},
    )
    io_cfg = _run_local_io_cfg(io_cfg)
    output_path = path_in_run(layout, "outputs", os.path.basename(output_path))
    if summary_path:
        summary_path = path_in_run(layout, "outputs", os.path.basename(summary_path))
    if top_k_output_path:
        top_k_output_path = path_in_run(layout, "outputs", os.path.basename(top_k_output_path))

    device = get_device()
    seq_len, vocab_size = tokenizer_meta(tok, window_size_bp)
    hidden_dim = train_cfg.hidden_dim
    model_type = train_cfg.model_type
    transformer_d_model = train_cfg.transformer_d_model
    transformer_nhead = train_cfg.transformer_nhead
    transformer_layers = train_cfg.transformer_layers
    transformer_dropout = train_cfg.transformer_dropout
    latent_dim = transformer_d_model if str(model_type).lower() == "transformer" else hidden_dim

    model, optimizer, global_step, ckpt_path = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tok,
        loss_type="mse",
        model_type=model_type,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        beta_kl=train_cfg.beta_kl,
    )
    model.eval()

    if num_windows is not None:
        n_windows = int(num_windows)
        target_bp = n_windows * window_size_bp
    else:
        n_windows = (length_bp + window_size_bp - 1) // window_size_bp
        target_bp = length_bp

    temperature = float(temperature)
    latent_scale = float(latent_scale)
    gc_bias = float(gc_bias)
    ast_details = _load_ast_conditioning_payload(ast_conditioning) if ast_conditioning is not None else None
    ast_vocab_mask = _build_ast_guided_mask(vocab_size=vocab_size, conditioning=ast_conditioning, details=ast_details)

    def _sample_once() -> str:
        seq_parts: List[str] = []
        with torch.no_grad():
            for _ in range(n_windows):
                z = torch.randn(1, latent_dim, device=device) * latent_scale
                logits_flat = model.decode(z)
                logits = logits_flat.view(seq_len, vocab_size).cpu().numpy()
                if tok == "base":
                    idx_to_base = ["A", "C", "G", "T"]
                    for j in range(seq_len):
                        w = 1.0 / (1.0 + np.exp(-logits[j]))
                        if gc_bias != 1.0:
                            w[1] *= gc_bias
                            w[2] *= gc_bias
                        w = w * ast_vocab_mask[: w.shape[0]]
                        idx = _sample_from_logits(np.log(np.clip(w, 1e-9, None)), temperature)
                        seq_parts.append(idx_to_base[idx])
                else:
                    for j in range(seq_len):
                        w = 1.0 / (1.0 + np.exp(-logits[j]))
                        if gc_bias != 1.0:
                            w *= (gc_bias ** GC_COUNT_PER_TOKEN[: w.shape[0]])
                        w = w * ast_vocab_mask[: w.shape[0]]
                        idx = _sample_from_logits(np.log(np.clip(w, 1e-9, None)), temperature)
                        seq_parts.append(IDX_TO_CODON[idx])
        return "".join(seq_parts)[:target_bp]

    nc = max(1, int(num_candidates))
    top_k = max(1, min(int(top_k), nc))
    target_gc = 0.5 if target_gc is None else float(target_gc)
    max_homopolymer = int(max_homopolymer) if max_homopolymer is not None else None

    candidates: List[Dict[str, object]] = []
    for i in range(nc):
        seq = _sample_once()
        recon = _roundtrip_recon_score(model, seq, tok, seq_len, vocab_size, device) if roundtrip_score else None
        scorecard = build_plasmid_scorecard(
            seq,
            {
                "sequence_id": f"candidate-{i}",
                "tokenizer": tok,
                "model_type": model_type,
                "target_gc": target_gc,
                "max_homopolymer": max_homopolymer,
                "roundtrip_recon": recon,
                "recon_weight": float(recon_weight),
                "reference_neighbors": list(scorecard_reference_neighbors or []),
                "reference_top_n": int(scorecard_reference_top_n),
                "motifs": scorecard_motifs,
                "min_orf_aa": int(getattr(train_cfg, "min_orf_aa", 90)),
            },
        )
        metrics = scorecard["metrics"]
        candidates.append({
            "candidate": i,
            "sequence": seq,
            "length": len(seq),
            "gc_fraction": metrics.get("gc_fraction"),
            "gc_deviation": metrics.get("gc_deviation"),
            "max_homopolymer": metrics.get("max_homopolymer"),
            "homopolymer_penalty": metrics.get("homopolymer_penalty"),
            "roundtrip_recon": recon,
            "recon_weight": float(recon_weight),
            "score": metrics.get("score"),
            "scorecard_version": scorecard.get("scorecard_version"),
            "risk_flags": scorecard.get("risk_flags", []),
            "summary": scorecard.get("summary", {}),
            "scorecard": scorecard,
        })

    ranked = sorted(candidates, key=lambda x: float(x["score"]), reverse=True)
    winner = ranked[0]
    seq = str(winner["sequence"])

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

    summary_json, summary_csv = _make_out_paths(output_path, summary_path)
    top_k_output = top_k_output_path if top_k_output_path else f"{output_path}.top{top_k}.fasta"
    _write_top_k_fasta(top_k_output, name, ranked, top_k)
    scorecards = [dict(c.get("scorecard") or {}) for c in ranked]
    aggregate = {
        "num_candidates": len(scorecards),
        "avg_gc_fraction": float(np.mean([float((s.get("metrics") or {}).get("gc_fraction", 0.0)) for s in scorecards])) if scorecards else 0.0,
        "avg_repeat_density": float(np.mean([float((s.get("metrics") or {}).get("repeat_density", 0.0)) for s in scorecards])) if scorecards else 0.0,
        "avg_orf_count": float(np.mean([float((s.get("metrics") or {}).get("orf_count", 0.0)) for s in scorecards])) if scorecards else 0.0,
        "risk_flag_counts": {},
    }
    for card in scorecards:
        for flag in card.get("risk_flags", []) or []:
            code = str((flag or {}).get("code") or "unknown")
            aggregate["risk_flag_counts"][code] = int(aggregate["risk_flag_counts"].get(code, 0) + 1)

    _write_candidate_summary(
        summary_json,
        summary_csv,
        {
            "mode": "plasmid",
            "scorecard_version": ranked[0].get("scorecard_version") if ranked else None,
            "tokenizer": tok,
            "num_candidates": nc,
            "top_k": top_k,
            "target_gc": target_gc,
            "max_homopolymer": max_homopolymer,
            "recon_weight": float(recon_weight),
            "ast_conditioning": ast_conditioning_metadata(ast_conditioning, ast_details),
            "winner": {k: v for k, v in winner.items() if k != "sequence"},
            "top_candidates": [{k: v for k, v in c.items() if k != "sequence"} for c in ranked[:top_k]],
            "scorecards": [{k: v for k, v in c.items() if k != "sequence"} for c in ranked],
            "scorecard_aggregate": aggregate,
            "top_k_output_path": top_k_output,
            "output_path": output_path,
        },
        [{k: v for k, v in c.items() if k != "sequence"} for c in ranked],
    )
    logging.info("[generate-plasmid] wrote candidate summary: %s", summary_json)
    update_run_manifest(
        layout,
        paths={
            "generated": {
                "plasmid_fasta": output_path,
                "plasmid_top_k_fasta": top_k_output,
                "plasmid_summary_json": summary_json,
                "plasmid_summary_csv": summary_csv,
            }
        },
        generated_sequences={
            "plasmid": {
                "output": output_path,
                "top_k_output": top_k_output,
                "summary_json": summary_json,
                "summary_csv": summary_csv,
                "name": name,
                "tokenizer": tok,
                "length_bp": len(seq),
                "top_k": top_k,
                "num_candidates": nc,
                "seed": int(seed_info["value"]),
            }
        },
        metrics={"generate_plasmid": {"length_bp": len(seq), "top_k": top_k, "seed": int(seed_info["value"])}},
        provenance={"rng": {"seed": int(seed_info["value"]), "source": str(seed_info["source"])}}
    )

    return seq

def generate_protein_sequence(
    train_cfg: TrainingConfig,
    io_cfg: IOConfig,
    length_aa: int,
    num_windows: Optional[int],
    window_aa: int,
    seed: Optional[int],
    latent_scale: float,
    temperature: float,
    name: str,
    output_path: str,
    reject: bool = False,
    reject_tries: int = 40,
    reject_max_run: int = 10,
    reject_max_x_frac: float = 0.15,
    num_candidates: int = 1,
    top_k: int = 1,
    max_homopolymer: Optional[int] = None,
    max_x_frac: Optional[float] = None,
    max_internal_stops: int = 0,
    summary_path: Optional[str] = None,
    top_k_output_path: Optional[str] = None,
    roundtrip_score: bool = False,
    recon_weight: float = 0.1,
    provenance_inputs: Optional[Dict[str, str]] = None,
    ast_conditioning: Optional[AstConditioningConfig] = None,
) -> str:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")

    seed_info = resolve_seed(seed)
    set_global_seeds(int(seed_info["value"]))

    layout = ensure_run_layout()
    collect_and_write_provenance(
        layout=layout,
        run_kind="generate_protein",
        seed_info=seed_info,
        input_paths=provenance_inputs or {},
    )
    io_cfg = _run_local_io_cfg(io_cfg)
    output_path = path_in_run(layout, "outputs", os.path.basename(output_path))
    if summary_path:
        summary_path = path_in_run(layout, "outputs", os.path.basename(summary_path))
    if top_k_output_path:
        top_k_output_path = path_in_run(layout, "outputs", os.path.basename(top_k_output_path))

    device = get_device()
    tok = "aa"
    seq_len, vocab_size = tokenizer_meta(tok, window_aa)
    assert vocab_size == AA_VOCAB_SIZE
    hidden_dim = train_cfg.hidden_dim
    model_type = train_cfg.model_type
    transformer_d_model = train_cfg.transformer_d_model
    transformer_nhead = train_cfg.transformer_nhead
    transformer_layers = train_cfg.transformer_layers
    transformer_dropout = train_cfg.transformer_dropout
    latent_dim = transformer_d_model if str(model_type).lower() == "transformer" else hidden_dim

    model, optimizer, global_step, ckpt_path = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tok,
        loss_type="ce",
        model_type=model_type,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        beta_kl=train_cfg.beta_kl,
    )
    model.eval()

    if num_windows is not None:
        n_windows = int(num_windows)
        target_aa = n_windows * window_aa
    else:
        n_windows = (length_aa + window_aa - 1) // window_aa
        target_aa = length_aa

    temperature = float(temperature)
    latent_scale = float(latent_scale)
    ast_details = _load_ast_conditioning_payload(ast_conditioning) if ast_conditioning is not None else None
    ast_vocab_mask = _build_ast_guided_mask(vocab_size=vocab_size, conditioning=ast_conditioning, details=ast_details)

    def _sample_once() -> str:
        aa_chars: List[str] = []
        with torch.no_grad():
            for _ in range(n_windows):
                z = torch.randn(1, latent_dim, device=device) * latent_scale
                logits_flat = model.decode(z)
                logits = logits_flat.view(seq_len, vocab_size).cpu().numpy()
                for j in range(seq_len):
                    idx = _sample_from_logits(logits[j] + np.log(np.clip(ast_vocab_mask[: logits[j].shape[0]], 1e-9, None)), temperature)
                    aa_chars.append(IDX_TO_AA[idx])
        return "".join(aa_chars)[:target_aa]

    def _sample_candidate() -> str:
        if not reject:
            return _sample_once()
        tries = max(1, int(reject_tries))
        candidate = ""
        for t in range(tries):
            candidate = _sample_once()
            if _passes_protein_filters(candidate, max_run=int(reject_max_run), max_x_frac=float(reject_max_x_frac)):
                return candidate
            if (t + 1) % 10 == 0:
                logging.info(f"[generate-protein] rejection: {t+1}/{tries} rejected")
        logging.warning("[generate-protein] rejection-sampling exhausted tries; using last sample")
        return candidate if candidate else _sample_once()

    nc = max(1, int(num_candidates))
    top_k = max(1, min(int(top_k), nc))
    max_homopolymer = int(max_homopolymer) if max_homopolymer is not None else int(reject_max_run)
    max_x_frac = float(max_x_frac) if max_x_frac is not None else float(reject_max_x_frac)

    allowed = set(IDX_TO_AA)
    candidates: List[Dict[str, object]] = []
    for i in range(nc):
        cand = _sample_candidate()
        recon = _roundtrip_recon_score(model, cand, tok, seq_len, vocab_size, device) if roundtrip_score else None
        scorecard = build_protein_scorecard(
            cand,
            {
                "sequence_id": f"candidate-{i}",
                "tokenizer": tok,
                "model_type": model_type,
                "max_homopolymer": max_homopolymer,
                "max_x_frac": max_x_frac,
                "max_internal_stops": max_internal_stops,
                "roundtrip_recon": recon,
                "recon_weight": float(recon_weight),
                "allowed": allowed,
            },
        )
        metrics = scorecard["metrics"]
        candidates.append({
            "candidate": i,
            "sequence": cand,
            "length": len(cand),
            "max_homopolymer": int(metrics["max_homopolymer"]),
            "x_fraction": metrics["x_fraction"],
            "invalid_fraction": metrics["invalid_fraction"],
            "stop_count": int(metrics["stop_count"]),
            "roundtrip_recon": recon,
            "recon_weight": float(recon_weight),
            "score": metrics["score"],
            "scorecard_version": scorecard.get("scorecard_version"),
            "risk_flags": scorecard.get("risk_flags", []),
            "summary": scorecard.get("summary", {}),
        })

    ranked = sorted(candidates, key=lambda x: float(x["score"]), reverse=True)
    seq = str(ranked[0]["sequence"])

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

    summary_json, summary_csv = _make_out_paths(output_path, summary_path)
    top_k_output = top_k_output_path if top_k_output_path else f"{output_path}.top{top_k}.fasta"
    _write_top_k_fasta(top_k_output, name, ranked, top_k)
    _write_candidate_summary(
        summary_json,
        summary_csv,
        {
            "mode": "protein",
            "scorecard_version": ranked[0].get("scorecard_version") if ranked else None,
            "num_candidates": nc,
            "top_k": top_k,
            "max_homopolymer": max_homopolymer,
            "max_x_frac": max_x_frac,
            "max_internal_stops": max_internal_stops,
            "recon_weight": float(recon_weight),
            "ast_conditioning": ast_conditioning_metadata(ast_conditioning, ast_details),
            "winner": {k: v for k, v in ranked[0].items() if k != "sequence"},
            "top_candidates": [{k: v for k, v in c.items() if k != "sequence"} for c in ranked[:top_k]],
            "top_k_output_path": top_k_output,
            "output_path": output_path,
        },
        [{k: v for k, v in c.items() if k != "sequence"} for c in ranked],
    )
    logging.info("[generate-protein] wrote candidate summary: %s", summary_json)
    update_run_manifest(
        layout,
        paths={
            "generated": {
                "protein_faa": output_path,
                "protein_top_k_fasta": top_k_output,
                "protein_summary_json": summary_json,
                "protein_summary_csv": summary_csv,
            }
        },
        generated_sequences={
            "protein": {
                "output": output_path,
                "top_k_output": top_k_output,
                "summary_json": summary_json,
                "summary_csv": summary_csv,
                "name": name,
                "tokenizer": tok,
                "length_aa": len(seq),
                "top_k": top_k,
                "num_candidates": nc,
                "seed": int(seed_info["value"]),
            }
        },
        metrics={"generate_protein": {"length_aa": len(seq), "top_k": top_k, "seed": int(seed_info["value"])}},
        provenance={"rng": {"seed": int(seed_info["value"]), "source": str(seed_info["source"])}}
    )

    return seq
