import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from .config import IOConfig, TrainingConfig
from .model import get_device, load_or_init_model, save_checkpoint, vae_loss
from .encoding_main import tokenizer_meta


def _default_loss_type(tokenizer: str) -> str:
    # AA/proteome mode benefits strongly from categorical CE.
    return "ce" if str(tokenizer).lower() == "aa" else "mse"


def _apply_aa_mask(batch_onehot: np.ndarray, mask_prob: float) -> np.ndarray:
    """Randomly replace some AA positions with X (unknown) in the *input*.

    batch_onehot: (B, L, V) one-hot
    Returns a modified copy.
    """
    p = float(mask_prob)
    if p <= 0:
        return batch_onehot
    x = batch_onehot.copy()
    X_IDX = x.shape[2] - 1
    mask = np.random.rand(x.shape[0], x.shape[1]) < p
    if mask.any():
        x[mask] = 0.0
        x[mask, X_IDX] = 1.0
    return x


def _apply_aa_span_mask(batch_onehot: np.ndarray, span_prob: float, span_len: int) -> np.ndarray:
    """Replace a contiguous span with X (unknown) in the *input*.

    This is a simple inpainting-style corruption. We keep the training target clean.
    """
    p = float(span_prob)
    L = int(span_len)
    if p <= 0 or L <= 0:
        return batch_onehot
    seq_len = batch_onehot.shape[1]
    if seq_len <= 0:
        return batch_onehot
    x = batch_onehot.copy()
    X_IDX = x.shape[2] - 1
    apply = np.random.rand(x.shape[0]) < p
    if not apply.any():
        return x
    max_start = max(0, seq_len - L)
    starts = np.random.randint(0, max_start + 1, size=(x.shape[0],))
    for b in range(x.shape[0]):
        if not apply[b]:
            continue
        s = int(starts[b])
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
    device = get_device()

    seq_len, vocab_size = tokenizer_meta(tokenizer, window_size_bp)
    hidden_dim = train_cfg.hidden_dim
    model_type = train_cfg.model_type
    transformer_d_model = train_cfg.transformer_d_model
    transformer_nhead = train_cfg.transformer_nhead
    transformer_layers = train_cfg.transformer_layers
    transformer_dropout = train_cfg.transformer_dropout

    lt = _default_loss_type(tokenizer) if loss_type is None else str(loss_type).lower()
    mp = float(mask_prob) if mask_prob is not None else float(
        getattr(train_cfg, "aa_mask_prob", 0.05 if str(tokenizer).lower() == "aa" else 0.0)
    )
    sp = float(span_mask_prob) if span_mask_prob is not None else float(
        getattr(train_cfg, "aa_span_mask_prob", 0.0)
    )
    sl = int(span_mask_len) if span_mask_len is not None else int(
        getattr(train_cfg, "aa_span_mask_len", 0)
    )

    with tf.device(device):
        model, optimizer, global_step, manager = load_or_init_model(
            io_cfg=io_cfg,
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            learning_rate=train_cfg.learning_rate,
            device=device,
            tokenizer=tokenizer,
            loss_type=lt,
            model_type=model_type,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            transformer_dropout=transformer_dropout,
        )

        if encoded.shape[0] == 0:
            logging.warning(f"{accession}: no windows to train on (shape={encoded.shape})")
            return 0.0

        dataset = (
            tf.data.Dataset.from_tensor_slices(encoded.astype(np.float32))
            .shuffle(max(1, encoded.shape[0]))
            .batch(batch_size, drop_remainder=False)
            .repeat()
        )
        dataloader = iter(dataset)

        logging.info(
            f"{accession}: train tokenizer={tokenizer} windows={encoded.shape[0]} "
            f"L={encoded.shape[1]} V={encoded.shape[2]} steps={steps} batch={batch_size}"
        )

        step_count = 0
        last_total = 0.0

        while step_count < steps:
            batch = next(dataloader)
            batch_np = batch.numpy()

            x_in = batch_np
            if str(tokenizer).lower() == "aa":
                if sp > 0 and sl > 0:
                    x_in = _apply_aa_span_mask(x_in, sp, sl)
                if mp > 0:
                    x_in = _apply_aa_mask(x_in, mp)

            x_in_tf = tf.convert_to_tensor(x_in, dtype=tf.float32)
            x_target_tf = tf.convert_to_tensor(batch_np, dtype=tf.float32)

            x_flat = tf.reshape(x_in_tf, (tf.shape(x_in_tf)[0], -1))
            x_target_flat = tf.reshape(x_target_tf, (tf.shape(x_target_tf)[0], -1))

            if train_cfg.kl_warmup_steps > 0:
                warm = min(1.0, (global_step + 1) / float(train_cfg.kl_warmup_steps))
                beta = train_cfg.beta_kl * warm
            else:
                beta = train_cfg.beta_kl

            with tf.GradientTape() as tape:
                recon_logits, mu, logvar = model(x_flat, training=True)
                total, recon, kl = vae_loss(
                    recon_logits,
                    x_target_flat,
                    mu,
                    logvar,
                    beta,
                    lt,
                    seq_len,
                    vocab_size,
                )

            grads = tape.gradient(total, model.trainable_variables)
            if train_cfg.max_grad_norm and train_cfg.max_grad_norm > 0:
                grads = [
                    tf.clip_by_norm(g, train_cfg.max_grad_norm) if g is not None else None
                    for g in grads
                ]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            step_count += 1
            global_step += 1
            last_total = float(total.numpy())

            if step_count % 10 == 0 or step_count == steps:
                logging.info(
                    f"{accession}: step {step_count}/{steps} total={total.numpy():.6f} "
                    f"recon={recon.numpy():.6f} kl={kl.numpy():.6f} beta={beta:.3g} "
                    f"loss={lt} mask={mp:.3g} span={sp:.3g}/{sl}"
                )

        state["total_steps"] = int(state.get("total_steps", 0)) + step_count

        save_checkpoint(
            manager=manager,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            tokenizer=tokenizer,
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            loss_type=lt,
            model_type=model_type,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            transformer_dropout=transformer_dropout,
        )

        return last_total


def cleanup_accession_files(accession: str, io_cfg: IOConfig, encoded_path: str) -> None:
    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
    gb_path = os.path.join(getattr(io_cfg, "cache_genbank_dir", "cache/genbank"), f"{accession}.gb")
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
    device = get_device()

    seq_len, vocab_size = tokenizer_meta(tokenizer, window_size_bp)
    hidden_dim = train_cfg.hidden_dim
    model_type = train_cfg.model_type
    transformer_d_model = train_cfg.transformer_d_model
    transformer_nhead = train_cfg.transformer_nhead
    transformer_layers = train_cfg.transformer_layers
    transformer_dropout = train_cfg.transformer_dropout

    lt = _default_loss_type(tokenizer) if loss_type is None else str(loss_type).lower()

    with tf.device(device):
        model, optimizer, global_step, manager = load_or_init_model(
            io_cfg=io_cfg,
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            learning_rate=train_cfg.learning_rate,
            device=device,
            tokenizer=tokenizer,
            loss_type=lt,
            model_type=model_type,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            transformer_dropout=transformer_dropout,
        )

        if encoded.size == 0:
            return np.zeros((0,), dtype=np.float32)

        wt = tf.convert_to_tensor(encoded.astype(np.float32))
        N = wt.shape[0]
        x_flat = tf.reshape(wt, (N, -1))
        mu, logvar = model.encode(x_flat, training=False)
        logits = model.decode(mu, training=False)

        if lt == "ce":
            logits3 = tf.reshape(logits, (N, int(seq_len), int(vocab_size)))
            targets = tf.argmax(wt, axis=2)
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits3)
            ce_w = tf.reduce_mean(tf.reshape(ce, (N, int(seq_len))), axis=1)
            return ce_w.numpy().astype(np.float32)

        recon = tf.reshape(tf.nn.sigmoid(logits), tf.shape(wt))
        mse = tf.reduce_mean(tf.square(recon - wt), axis=(1, 2))
        return mse.numpy().astype(np.float32)
