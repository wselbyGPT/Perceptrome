import csv
import json
import logging, os, random
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .config import TrainingConfig, IOConfig
from .model import get_device, load_or_init_model
from .encoding_main import tokenizer_meta, IDX_TO_CODON, CODON_VOCAB_SIZE, GC_COUNT_PER_TOKEN, IDX_TO_AA, AA_VOCAB_SIZE

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
CODON_TO_IDX = {codon: i for i, codon in enumerate(IDX_TO_CODON)}
AA_TO_IDX = {aa: i for i, aa in enumerate(IDX_TO_AA)}
VALID_AA_CHARS = set("ACDEFGHIKLMNPQRSTVWY")

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
    best = run = 1
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
    gc = sum(1 for c in seq if c in ("G", "C"))
    return gc / float(len(seq))


def _invalid_aa_ratio(seq: str) -> float:
    if not seq:
        return 1.0
    bad = sum(1 for aa in seq if aa not in VALID_AA_CHARS and aa != "*")
    return bad / float(len(seq))


def _stop_penalty(seq: str, stop_policy: str) -> float:
    pol = str(stop_policy or "allow").lower()
    stop_count = seq.count("*")
    if pol == "allow":
        return 0.0
    if pol == "none":
        return float(stop_count)
    if pol == "terminal":
        if stop_count == 0:
            return 0.0
        if stop_count == 1 and seq.endswith("*"):
            return 0.0
        return float(stop_count)
    raise ValueError(f"Unknown stop policy: {stop_policy}")


def _roundtrip_score(
    model: Any,
    seq: str,
    tokenizer: str,
    seq_len: int,
    vocab_size: int,
    device: Any,
    loss_type: str,
) -> Optional[float]:
    if torch is None:
        return None

    tok = tokenizer.lower()
    if tok == "base":
        toks = list(seq)
        idx_map = BASE_TO_IDX
    elif tok == "codon":
        toks = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]
        idx_map = CODON_TO_IDX
    elif tok == "aa":
        toks = list(seq)
        idx_map = AA_TO_IDX
    else:
        return None

    if not toks:
        return None

    n = min(len(toks), seq_len)
    x = torch.zeros((1, seq_len, vocab_size), dtype=torch.float32, device=device)
    valid = 0
    for i in range(n):
        idx = idx_map.get(toks[i])
        if idx is None or idx < 0 or idx >= vocab_size:
            continue
        x[0, i, idx] = 1.0
        valid += 1
    if valid == 0:
        return None

    x_flat = x.view(1, seq_len * vocab_size)
    with torch.no_grad():
        mu, _ = model.encode(x_flat)
        probs = model.decode_probs(mu, seq_len=seq_len, vocab_size=vocab_size, loss_type=loss_type)
        p = 0.0
        for i in range(n):
            idx = idx_map.get(toks[i])
            if idx is None or idx < 0 or idx >= vocab_size:
                continue
            p += float(probs[0, i, idx].item())
    return p / float(valid)


def _write_summary(path: Optional[str], rows: List[Dict[str, Any]], fmt: str) -> None:
    if not path:
        return
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "csv":
        if not rows:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return
        fields = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _write_fasta(path: str, entries: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(f">{e['name']}\n")
            seq = e["sequence"]
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")

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
    roundtrip_score: bool = False,
    summary_csv: Optional[str] = None,
    summary_json: Optional[str] = None,
) -> Dict[str, Any]:
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

    candidates: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i in range(max(1, int(num_candidates))):
            seq_parts: List[str] = []
            for _ in range(n_windows):
                z = torch.randn(1, latent_dim, device=device) * latent_scale
                logits_flat = model.decode(z)   # (1, seq_len*vocab)
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

            seq = "".join(seq_parts)[:target_bp]
            gc = _gc_fraction(seq)
            gc_dev = abs(gc - float(target_gc)) if target_gc is not None else 0.0
            max_run = _max_homopolymer_run(seq)
            hp_penalty = 0.0
            if max_homopolymer is not None and max_homopolymer > 0:
                hp_penalty = max(0, max_run - int(max_homopolymer)) / float(max_homopolymer)
            rt = _roundtrip_score(model, seq, tok, seq_len, vocab_size, device, loss_type="mse") if roundtrip_score else None
            score = -(gc_dev + hp_penalty + (0.0 if rt is None else (1.0 - rt)))
            candidates.append({
                "rank": 0,
                "candidate": i + 1,
                "name": f"{name}_{i+1}",
                "sequence": seq,
                "length": len(seq),
                "gc_fraction": gc,
                "gc_target_deviation": gc_dev,
                "max_homopolymer_run": max_run,
                "homopolymer_penalty": hp_penalty,
                "roundtrip_score": rt,
                "invalid_aa_ratio": 0.0,
                "stop_penalty": 0.0,
                "heuristic_score": score,
            })

    ranked = sorted(candidates, key=lambda x: x["heuristic_score"], reverse=True)
    keep = max(1, min(int(top_k), len(ranked)))
    top = ranked[:keep]
    for r, row in enumerate(top, start=1):
        row["rank"] = r

    _write_fasta(output_path, top)
    _write_summary(summary_csv, [{k: v for k, v in row.items() if k != "sequence"} for row in ranked], fmt="csv")
    _write_summary(summary_json, [{k: v for k, v in row.items() if k != "sequence"} for row in ranked], fmt="json")

    return {"candidates": ranked, "top": top}

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
    max_invalid_aa_ratio: float = 0.15,
    stop_policy: str = "allow",
    roundtrip_score: bool = False,
    summary_csv: Optional[str] = None,
    summary_json: Optional[str] = None,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")

    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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

    candidates: List[Dict[str, Any]] = []
    for i in range(max(1, int(num_candidates))):
        if reject:
            tries = max(1, int(reject_tries))
            for t in range(tries):
                seq = _sample_once()
                if _passes_protein_filters(seq, max_run=int(reject_max_run), max_x_frac=float(reject_max_x_frac)):
                    break
                if (t + 1) % 10 == 0:
                    logging.info(f"[generate-protein] rejection: {t+1}/{tries} rejected")
            else:
                logging.warning("[generate-protein] rejection-sampling exhausted tries; using last sample")
                seq = _sample_once()
        else:
            seq = _sample_once()

        max_run = _max_homopolymer_run(seq)
        hp_limit = int(max_homopolymer) if max_homopolymer is not None else int(reject_max_run)
        hp_penalty = max(0, max_run - hp_limit) / float(max(1, hp_limit))
        invalid_ratio = _invalid_aa_ratio(seq)
        invalid_penalty = max(0.0, invalid_ratio - float(max_invalid_aa_ratio))
        stop_pen = _stop_penalty(seq, stop_policy)
        rt = _roundtrip_score(model, seq, "aa", seq_len, vocab_size, device, loss_type="ce") if roundtrip_score else None
        score = -(hp_penalty + invalid_penalty + stop_pen + (0.0 if rt is None else (1.0 - rt)))
        candidates.append({
            "rank": 0,
            "candidate": i + 1,
            "name": f"{name}_{i+1}",
            "sequence": seq,
            "length": len(seq),
            "gc_fraction": None,
            "gc_target_deviation": None,
            "max_homopolymer_run": max_run,
            "homopolymer_penalty": hp_penalty,
            "roundtrip_score": rt,
            "invalid_aa_ratio": invalid_ratio,
            "stop_penalty": stop_pen,
            "heuristic_score": score,
        })

    ranked = sorted(candidates, key=lambda x: x["heuristic_score"], reverse=True)
    keep = max(1, min(int(top_k), len(ranked)))
    top = ranked[:keep]
    for r, row in enumerate(top, start=1):
        row["rank"] = r

    _write_fasta(output_path, top)
    _write_summary(summary_csv, [{k: v for k, v in row.items() if k != "sequence"} for row in ranked], fmt="csv")
    _write_summary(summary_json, [{k: v for k, v in row.items() if k != "sequence"} for row in ranked], fmt="json")

    return {"candidates": ranked, "top": top}
