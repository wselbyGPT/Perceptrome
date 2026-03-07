import csv
import json
import logging, os, random
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .config import TrainingConfig, IOConfig
from .model import get_device, load_or_init_model
from .encoding_main import tokenizer_meta, IDX_TO_CODON, CODON_VOCAB_SIZE, GC_COUNT_PER_TOKEN, IDX_TO_AA, AA_VOCAB_SIZE
from .run_layout import ensure_run_layout, path_in_run, update_run_manifest


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
    if not seq:
        return 0
    run = 1
    best = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for ch in seq if ch in ("G", "C"))
    return gc / float(len(seq))


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
    allowed = allowed if allowed is not None else set(IDX_TO_AA)
    run = _max_homopolymer_run(seq)
    x_frac = seq.count("X") / float(max(1, len(seq)))
    invalid_frac = sum(1 for ch in seq if ch not in allowed) / float(max(1, len(seq)))
    stop_count = seq.count("*")
    pen_run = max(0, run - max_homopolymer) / max(1.0, float(max_homopolymer))
    pen_x = max(0.0, x_frac - max_x_frac)
    pen_stop = max(0, stop_count - int(max_internal_stops))
    pen_invalid = invalid_frac * 2.0
    score = -(pen_run + pen_x + pen_stop + pen_invalid) - (float(recon_weight) * float(recon) if recon is not None else 0.0)
    return {
        "score": float(score),
        "max_homopolymer": float(run),
        "x_fraction": float(x_frac),
        "invalid_fraction": float(invalid_frac),
        "stop_count": float(stop_count),
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
) -> str:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    tok = tokenizer.lower()
    if tok not in ("base", "codon"):
        raise ValueError("generate_plasmid_sequence only supports base|codon")

    if tok == "codon" and length_bp % 3 != 0:
        length_bp = (length_bp // 3) * 3

    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    layout = ensure_run_layout()
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
                        idx = _sample_from_logits(np.log(np.clip(w, 1e-9, None)), temperature)
                        seq_parts.append(idx_to_base[idx])
                else:
                    for j in range(seq_len):
                        w = 1.0 / (1.0 + np.exp(-logits[j]))
                        if gc_bias != 1.0:
                            w *= (gc_bias ** GC_COUNT_PER_TOKEN[: w.shape[0]])
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
        gc = _gc_fraction(seq)
        run = _max_homopolymer_run(seq)
        gc_dev = abs(gc - target_gc)
        recon = _roundtrip_recon_score(model, seq, tok, seq_len, vocab_size, device) if roundtrip_score else None
        score, run_pen = _plasmid_candidate_score(
            gc_dev=gc_dev,
            homopolymer_run=run,
            max_homopolymer=max_homopolymer,
            recon=recon,
            recon_weight=recon_weight,
        )
        candidates.append({
            "candidate": i,
            "sequence": seq,
            "length": len(seq),
            "gc_fraction": gc,
            "gc_deviation": gc_dev,
            "max_homopolymer": run,
            "homopolymer_penalty": run_pen,
            "roundtrip_recon": recon,
            "recon_weight": float(recon_weight),
            "score": score,
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
    _write_candidate_summary(
        summary_json,
        summary_csv,
        {
            "mode": "plasmid",
            "tokenizer": tok,
            "num_candidates": nc,
            "top_k": top_k,
            "target_gc": target_gc,
            "max_homopolymer": max_homopolymer,
            "recon_weight": float(recon_weight),
            "winner": {k: v for k, v in winner.items() if k != "sequence"},
            "top_candidates": [{k: v for k, v in c.items() if k != "sequence"} for c in ranked[:top_k]],
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
        metrics={"generate_plasmid": {"length_bp": len(seq), "top_k": top_k}},
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
) -> str:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")

    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    layout = ensure_run_layout()
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

    def _sample_once() -> str:
        aa_chars: List[str] = []
        with torch.no_grad():
            for _ in range(n_windows):
                z = torch.randn(1, latent_dim, device=device) * latent_scale
                logits_flat = model.decode(z)
                logits = logits_flat.view(seq_len, vocab_size).cpu().numpy()
                for j in range(seq_len):
                    idx = _sample_from_logits(logits[j], temperature)
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
        metrics = _protein_candidate_score(
            seq=cand,
            max_homopolymer=max_homopolymer,
            max_x_frac=max_x_frac,
            max_internal_stops=max_internal_stops,
            recon=recon,
            recon_weight=recon_weight,
            allowed=allowed,
        )
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
            "num_candidates": nc,
            "top_k": top_k,
            "max_homopolymer": max_homopolymer,
            "max_x_frac": max_x_frac,
            "max_internal_stops": max_internal_stops,
            "recon_weight": float(recon_weight),
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
        metrics={"generate_protein": {"length_aa": len(seq), "top_k": top_k}},
    )

    return seq
