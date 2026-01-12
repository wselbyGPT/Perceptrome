import json
import logging
import os
from typing import Dict, Tuple

import tensorflow as tf

from .config import IOConfig


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if int(nhead) <= 0:
            raise ValueError(f"nhead must be > 0 (got {nhead})")
        key_dim = max(1, int(d_model) // int(nhead))
        if int(d_model) % int(nhead) != 0:
            logging.warning(
                "Transformer d_model (%s) is not divisible by nhead (%s); using key_dim=%s",
                d_model,
                nhead,
                key_dim,
            )
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=int(nhead), key_dim=key_dim, dropout=float(dropout)
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(int(d_model) * 4, activation="relu"),
                tf.keras.layers.Dense(int(d_model)),
            ]
        )
        self.dropout1 = tf.keras.layers.Dropout(float(dropout))
        self.dropout2 = tf.keras.layers.Dropout(float(dropout))
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.dropout1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + self.dropout2(ffn_out, training=training))


class TransformerVAE(tf.keras.Model):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)

        self.input_proj = tf.keras.layers.Dense(self.d_model)
        self.pos_embed = self.add_weight(
            shape=(1, self.seq_len, self.d_model), initializer="zeros", trainable=True
        )
        self.encoder_layers = [
            TransformerBlock(self.d_model, int(nhead), float(dropout))
            for _ in range(int(num_layers))
        ]

        self.fc_mu = tf.keras.layers.Dense(self.d_model)
        self.fc_logvar = tf.keras.layers.Dense(self.d_model)

        self.z_to_seq = tf.keras.layers.Dense(self.seq_len * self.d_model)
        self.decoder_layers = [
            TransformerBlock(self.d_model, int(nhead), float(dropout))
            for _ in range(int(num_layers))
        ]
        self.out_proj = tf.keras.layers.Dense(self.vocab_size)

    def _ensure_seq(self, x: tf.Tensor) -> tf.Tensor:
        if len(x.shape) == 2:
            return tf.reshape(x, (-1, self.seq_len, self.vocab_size))
        return x

    def encode(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        x_seq = self._ensure_seq(x)
        h = self.input_proj(x_seq) + self.pos_embed
        for layer in self.encoder_layers:
            h = layer(h, training=training)
        pooled = tf.reduce_mean(h, axis=1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(tf.shape(std))
        return mu + eps * std

    def decode(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = self.z_to_seq(z)
        h = tf.reshape(h, (-1, self.seq_len, self.d_model))
        h = h + self.pos_embed
        for layer in self.decoder_layers:
            h = layer(h, training=training)
        logits = self.out_proj(h)
        return tf.reshape(logits, (-1, self.seq_len * self.vocab_size))

    def decode_probs(self, z: tf.Tensor, loss_type: str) -> tf.Tensor:
        logits = tf.reshape(self.decode(z), (-1, self.seq_len, self.vocab_size))
        if str(loss_type).lower() == "ce":
            return tf.nn.softmax(logits, axis=-1)
        return tf.nn.sigmoid(logits)

    def call(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        mu, logvar = self.encode(x, training=training)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z, training=training)
        return recon_logits, mu, logvar


class PlasmidVAE(tf.keras.Model):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.fc1 = tf.keras.layers.Dense(self.hidden_dim, activation="relu")
        self.fc_mu = tf.keras.layers.Dense(self.hidden_dim)
        self.fc_logvar = tf.keras.layers.Dense(self.hidden_dim)
        self.fc2 = tf.keras.layers.Dense(self.hidden_dim, activation="relu")
        self.fc_out = tf.keras.layers.Dense(int(input_dim))

    def encode(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        h = self.fc1(x, training=training)
        return self.fc_mu(h, training=training), self.fc_logvar(h, training=training)

    def reparameterize(self, mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(tf.shape(std))
        return mu + eps * std

    def decode(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = self.fc2(z, training=training)
        return self.fc_out(h, training=training)

    def decode_probs(self, z: tf.Tensor, seq_len: int, vocab_size: int, loss_type: str) -> tf.Tensor:
        logits = tf.reshape(self.decode(z), (-1, int(seq_len), int(vocab_size)))
        if str(loss_type).lower() == "ce":
            return tf.nn.softmax(logits, axis=-1)
        return tf.nn.sigmoid(logits)

    def call(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        mu, logvar = self.encode(x, training=training)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z, training=training)
        return recon_logits, mu, logvar


def get_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    return "/GPU:0" if gpus else "/CPU:0"


def _checkpoint_manager(
    model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, io_cfg: IOConfig
) -> tf.train.CheckpointManager:
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    return tf.train.CheckpointManager(
        ckpt,
        io_cfg.checkpoints_dir,
        max_to_keep=1,
        checkpoint_name="keras_latest",
    )


def _meta_path(io_cfg: IOConfig) -> str:
    return os.path.join(io_cfg.checkpoints_dir, "keras_latest.json")


def load_or_init_model(
    io_cfg: IOConfig,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
    learning_rate: float,
    device: str,
    tokenizer: str,
    loss_type: str,
    model_type: str,
    transformer_d_model: int,
    transformer_nhead: int,
    transformer_layers: int,
    transformer_dropout: float,
) -> Tuple[tf.keras.Model, tf.keras.optimizers.Optimizer, int, tf.train.CheckpointManager]:
    """
    seq_len: number of positions (bp or codons)
    vocab_size: 4 for base, 65 for codon
    """
    mt = str(model_type).lower()
    input_dim = int(seq_len) * int(vocab_size)
    if mt == "transformer":
        model = TransformerVAE(
            seq_len=seq_len,
            vocab_size=vocab_size,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
        )
    else:
        model = PlasmidVAE(input_dim=input_dim, hidden_dim=hidden_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))

    dummy = tf.zeros((1, input_dim), dtype=tf.float32)
    _ = model(dummy, training=False)

    manager = _checkpoint_manager(model, optimizer, io_cfg)
    meta_path = _meta_path(io_cfg)
    global_step = 0

    if os.path.exists(meta_path) and manager.latest_checkpoint:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta: Dict[str, object] = json.load(f)
        ck_tok = str(meta.get("tokenizer", "base")).lower()
        ck_seq = int(meta.get("seq_len", seq_len))
        ck_vocab = int(meta.get("vocab_size", vocab_size))
        ck_hidden = int(meta.get("hidden_dim", hidden_dim))
        ck_loss = str(meta.get("loss_type", "mse")).lower()
        ck_model_type = str(meta.get("model_type", "mlp")).lower()
        ck_d_model = int(meta.get("transformer_d_model", transformer_d_model))
        ck_nhead = int(meta.get("transformer_nhead", transformer_nhead))
        ck_layers = int(meta.get("transformer_layers", transformer_layers))
        ck_dropout = float(meta.get("transformer_dropout", transformer_dropout))
        global_step = int(meta.get("global_step", 0))

        if ck_tok != tokenizer.lower():
            raise ValueError(
                f"Checkpoint tokenizer={ck_tok} but requested tokenizer={tokenizer}. "
                f"Delete {meta_path} or match settings."
            )
        if ck_seq != seq_len:
            raise ValueError(
                f"Checkpoint seq_len={ck_seq} but requested seq_len={seq_len}. "
                f"Delete {meta_path} or match settings."
            )
        if ck_vocab != vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size={ck_vocab} but requested vocab_size={vocab_size}. "
                f"Delete {meta_path} or match settings."
            )
        if ck_hidden != hidden_dim and mt != "transformer":
            raise ValueError(
                f"Checkpoint hidden_dim={ck_hidden} but requested hidden_dim={hidden_dim}. "
                f"Delete {meta_path} or match settings."
            )
        if ck_loss != str(loss_type).lower():
            raise ValueError(
                f"Checkpoint loss_type={ck_loss} but requested loss_type={loss_type}. "
                f"Delete {meta_path} or match settings."
            )
        if ck_model_type != mt:
            raise ValueError(
                f"Checkpoint model_type={ck_model_type} but requested model_type={mt}. "
                f"Delete {meta_path} or match settings."
            )
        if mt == "transformer":
            if ck_d_model != transformer_d_model:
                raise ValueError(
                    f"Checkpoint transformer_d_model={ck_d_model} but requested {transformer_d_model}. "
                    f"Delete {meta_path} or match settings."
                )
            if ck_nhead != transformer_nhead:
                raise ValueError(
                    f"Checkpoint transformer_nhead={ck_nhead} but requested {transformer_nhead}. "
                    f"Delete {meta_path} or match settings."
                )
            if ck_layers != transformer_layers:
                raise ValueError(
                    f"Checkpoint transformer_layers={ck_layers} but requested {transformer_layers}. "
                    f"Delete {meta_path} or match settings."
                )
            if abs(ck_dropout - float(transformer_dropout)) > 1e-8:
                raise ValueError(
                    f"Checkpoint transformer_dropout={ck_dropout} but requested {transformer_dropout}. "
                    f"Delete {meta_path} or match settings."
                )

        manager.checkpoint.restore(manager.latest_checkpoint).expect_partial()
        logging.info(
            "Loaded checkpoint %s (tokenizer=%s, seq_len=%s, vocab=%s, hidden=%s, model=%s, step=%s)",
            manager.latest_checkpoint,
            ck_tok,
            ck_seq,
            ck_vocab,
            ck_hidden,
            ck_model_type,
            global_step,
        )
    else:
        logging.info(
            "Initializing new VAE (tokenizer=%s, loss_type=%s, model=%s, "
            "seq_len=%s, vocab=%s, input_dim=%s, hidden=%s, d_model=%s, nhead=%s, layers=%s, dropout=%s, lr=%s)",
            tokenizer,
            loss_type,
            mt,
            seq_len,
            vocab_size,
            input_dim,
            hidden_dim,
            transformer_d_model,
            transformer_nhead,
            transformer_layers,
            transformer_dropout,
            learning_rate,
        )

    return model, optimizer, global_step, manager


def save_checkpoint(
    manager: tf.train.CheckpointManager,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    global_step: int,
    tokenizer: str,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
    loss_type: str,
    model_type: str,
    transformer_d_model: int,
    transformer_nhead: int,
    transformer_layers: int,
    transformer_dropout: float,
) -> None:
    meta = {
        "global_step": int(global_step),
        "tokenizer": str(tokenizer).lower(),
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
        "hidden_dim": int(hidden_dim),
        "loss_type": str(loss_type).lower(),
        "model_type": str(model_type).lower(),
        "transformer_d_model": int(transformer_d_model),
        "transformer_nhead": int(transformer_nhead),
        "transformer_layers": int(transformer_layers),
        "transformer_dropout": float(transformer_dropout),
    }
    meta_path = os.path.join(manager.directory, "keras_latest.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    manager.save(checkpoint_number=global_step)
    logging.info("Saved checkpoint step=%s -> %s", global_step, manager.directory)


def vae_loss(
    recon_logits: tf.Tensor,
    x: tf.Tensor,
    mu: tf.Tensor,
    logvar: tf.Tensor,
    beta_kl: float,
    loss_type: str,
    seq_len: int,
    vocab_size: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    lt = str(loss_type).lower()
    if lt == "ce":
        logits = tf.reshape(recon_logits, (-1, int(seq_len), int(vocab_size)))
        targets = tf.argmax(tf.reshape(x, (-1, int(seq_len), int(vocab_size))), axis=-1)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        recon_term = tf.reduce_mean(ce)
    else:
        recon = tf.nn.sigmoid(recon_logits)
        recon_term = tf.reduce_mean(tf.square(recon - x))

    kl = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
    total = recon_term + float(beta_kl) * kl
    return total, recon_term, kl
