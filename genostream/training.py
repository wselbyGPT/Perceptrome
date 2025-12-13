import logging, os
from typing import Dict, Any, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

from .config import IOConfig, TrainingConfig
from .model import get_device, load_or_init_model, save_checkpoint, vae_loss
from .encoding import tokenizer_meta


def _default_loss_type(tokenizer: str) -> str:
    # AA/proteome mode benefits strongly from categorical CE.
    return "ce" if str(tokenizer).lower() == "aa" else "mse"


def _apply_aa_mask(batch_onehot: "torch.Tensor", mask_prob: float) -> "torch.Tensor":
    """Randomly replace some AA positions with X (unknown) in the *input*.

    batch_onehot: (B, L, V) one-hot
    Returns a modified copy.
    """
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    p = float(mask_prob)
    if p <= 0:
        return batch_onehot
    # X is last index in AA vocab (see encoding.AA_VOCAB = 20 + X)
    X_IDX = batch_onehot.size(2) - 1
    x = batch_onehot.clone()
    mask = (torch.rand(x.size(0), x.size(1), device=x.device) < p)
    if mask.any():
        x[mask] = 0.0
        x[mask, X_IDX] = 1.0
    return x



def _apply_aa_span_mask(batch_onehot: "torch.Tensor", span_prob: float, span_len: int) -> "torch.Tensor":
    """Replace a contiguous span with X (unknown) in the *input*.

    This is a simple inpainting-style corruption. We keep the training target clean.
    """
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    p = float(span_prob)
    L = int(span_len)
    if p <= 0 or L <= 0:
        return batch_onehot
    B = batch_onehot.size(0)
    seq_len = batch_onehot.size(1)
    if seq_len <= 0:
        return batch_onehot
    # X is last index in AA vocab
    X_IDX = batch_onehot.size(2) - 1
    x = batch_onehot.clone()
    # for each sample, maybe apply one span
    apply = (torch.rand(B, device=x.device) < p)
    if not bool(apply.any()):
        return x
    # start positions
    max_start = max(0, seq_len - L)
    starts = torch.randint(low=0, high=max_start + 1, size=(B,), device=x.device)
    for b in range(B):
        if not bool(apply[b]):
            continue
        s = int(starts[b].item())
        e = min(seq_len, s + L)
        x[b, s:e, :] = 0.0
        x[b, s:e, X_IDX] = 1.0
    return x

def train_on_encoded(
    accession: str,
    encoded: np.ndarray,
    steps: int,
    batch_size: int,
    state: Dict[str, Any],
    io_cfg: IOConfig,
    train_cfg: TrainingConfig,
    tokenizer: str,
    window_size_bp: int,
    loss_type: Optional[str] = None,
    mask_prob: Optional[float] = None,
    span_mask_prob: Optional[float] = None,
    span_mask_len: Optional[int] = None,
) -> float:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    device = get_device()

    seq_len, vocab_size = tokenizer_meta(tokenizer, window_size_bp)
    hidden_dim = train_cfg.hidden_dim

    lt = _default_loss_type(tokenizer) if loss_type is None else str(loss_type).lower()
    mp = float(mask_prob) if mask_prob is not None else float(getattr(train_cfg, 'aa_mask_prob', 0.05 if str(tokenizer).lower() == 'aa' else 0.0))
    sp = float(span_mask_prob) if span_mask_prob is not None else float(getattr(train_cfg, 'aa_span_mask_prob', 0.0))
    sl = int(span_mask_len) if span_mask_len is not None else int(getattr(train_cfg, 'aa_span_mask_len', 0))

    model, optimizer, global_step, ckpt_path = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tokenizer,
        loss_type=lt,
    )

    windows_tensor = torch.from_numpy(encoded)  # (N, L, V)
    dataset = TensorDataset(windows_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if len(dataloader) == 0:
        logging.warning(f"{accession}: no windows to train on (shape={encoded.shape})")
        return 0.0

    logging.info(
        f"{accession}: train tokenizer={tokenizer} windows={encoded.shape[0]} "
        f"L={encoded.shape[1]} V={encoded.shape[2]} steps={steps} batch={batch_size}"
    )

    step_count = 0
    last_total = 0.0
    it = iter(dataloader)

    while step_count < steps:
        try:
            (batch,) = next(it)
        except StopIteration:
            it = iter(dataloader)
            (batch,) = next(it)

        batch = batch.to(device)         # (B, L, V)
        # Denoising/inpainting (AA only): corrupt *input*, keep target clean.
        x_in = batch
        if str(tokenizer).lower() == "aa":
            if sp > 0 and sl > 0:
                x_in = _apply_aa_span_mask(x_in, sp, sl)
            if mp > 0:
                x_in = _apply_aa_mask(x_in, mp)
        x_flat = x_in.view(x_in.size(0), -1)
        x_target_flat = batch.view(batch.size(0), -1)

        if train_cfg.kl_warmup_steps > 0:
            warm = min(1.0, (global_step + 1) / float(train_cfg.kl_warmup_steps))
            beta = train_cfg.beta_kl * warm
        else:
            beta = train_cfg.beta_kl

        optimizer.zero_grad(set_to_none=True)
        recon_logits, mu, logvar = model(x_flat)
        total, recon, kl = vae_loss(recon_logits, x_target_flat, mu, logvar, beta, lt, seq_len, vocab_size)
        total.backward()

        if train_cfg.max_grad_norm and train_cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

        optimizer.step()

        step_count += 1
        global_step += 1
        last_total = float(total.item())

        if step_count % 10 == 0 or step_count == steps:
            logging.info(
                f"{accession}: step {step_count}/{steps} total={total.item():.6f} recon={recon.item():.6f} kl={kl.item():.6f} beta={beta:.3g} loss={lt} mask={mp:.3g} span={sp:.3g}/{sl}"
            )

    state["total_steps"] = int(state.get("total_steps", 0)) + step_count

    save_checkpoint(
        ckpt_path=ckpt_path,
        model=model,
        optimizer=optimizer,
        global_step=global_step,
        tokenizer=tokenizer,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        loss_type=lt,
    )

    return last_total

def cleanup_accession_files(accession: str, io_cfg: IOConfig, encoded_path: str) -> None:
    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
    gb_path = os.path.join(getattr(io_cfg, 'cache_genbank_dir', 'cache/genbank'), f"{accession}.gb")
    for path in (fasta_path, gb_path, encoded_path):
        if os.path.exists(path):
            try:
                os.remove(path)
                logging.info(f"{accession}: deleted {path}")
            except OSError as e:
                logging.warning(f"{accession}: failed to delete {path}: {e}")

def compute_window_errors(
    accession: str,
    encoded: np.ndarray,
    io_cfg: IOConfig,
    train_cfg: TrainingConfig,
    tokenizer: str,
    window_size_bp: int,
    loss_type: Optional[str] = None,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    device = get_device()

    seq_len, vocab_size = tokenizer_meta(tokenizer, window_size_bp)
    hidden_dim = train_cfg.hidden_dim

    lt = _default_loss_type(tokenizer) if loss_type is None else str(loss_type).lower()

    model, optimizer, global_step, ckpt_path = load_or_init_model(
        io_cfg=io_cfg,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        learning_rate=train_cfg.learning_rate,
        device=device,
        tokenizer=tokenizer,
        loss_type=lt,
    )

    model.eval()
    with torch.no_grad():
        wt = torch.from_numpy(encoded).to(device)
        if wt.numel() == 0:
            return np.zeros((0,), dtype=np.float32)
        N = wt.size(0)
        x_flat = wt.view(N, -1)
        mu, logvar = model.encode(x_flat)
        logits = model.decode(mu)
        if lt == "ce":
            # Negative log-likelihood per window (mean CE over positions)
            if torch is None:
                raise RuntimeError("PyTorch not installed.")
            import torch.nn.functional as F
            logits3 = logits.view(N, int(seq_len), int(vocab_size))
            targets = wt.argmax(dim=2)
            ce = F.cross_entropy(logits3.view(-1, int(vocab_size)), targets.view(-1), reduction="none")
            ce_w = ce.view(N, int(seq_len)).mean(dim=1)
            return ce_w.cpu().numpy().astype(np.float32)
        else:
            recon = torch.sigmoid(logits).view_as(wt)
            mse = (recon - wt).pow(2).mean(dim=(1, 2))
            return mse.cpu().numpy().astype(np.float32)
