import logging, os, random
from typing import List, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .config import TrainingConfig, IOConfig
from .model import get_device, load_or_init_model
from .encoding_main import tokenizer_meta, IDX_TO_CODON, CODON_VOCAB_SIZE, GC_COUNT_PER_TOKEN, IDX_TO_AA, AA_VOCAB_SIZE

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

    seq_parts: List[str] = []

    with torch.no_grad():
        for _ in range(n_windows):
            z = torch.randn(1, latent_dim, device=device) * latent_scale
            logits_flat = model.decode(z)   # (1, seq_len*vocab)
            logits = logits_flat.view(seq_len, vocab_size).cpu().numpy()

            if tok == "base":
                idx_to_base = ["A", "C", "G", "T"]
                for j in range(seq_len):
                    # base/codon models are trained with MSE on sigmoid weights
                    w = 1.0 / (1.0 + np.exp(-logits[j]))
                    # apply gc bias to C/G
                    if gc_bias != 1.0:
                        w[1] *= gc_bias
                        w[2] *= gc_bias
                    # weights, not logits: convert to pseudo-logits via log
                    idx = _sample_from_logits(np.log(np.clip(w, 1e-9, None)), temperature)
                    seq_parts.append(idx_to_base[idx])
            else:
                for j in range(seq_len):
                    w = 1.0 / (1.0 + np.exp(-logits[j]))
                    if gc_bias != 1.0:
                        w *= (gc_bias ** GC_COUNT_PER_TOKEN[: w.shape[0]])
                    idx = _sample_from_logits(np.log(np.clip(w, 1e-9, None)), temperature)
                    seq_parts.append(IDX_TO_CODON[idx])

    seq = "".join(seq_parts)
    seq = seq[:target_bp]

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

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
) -> str:
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

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

    return seq
