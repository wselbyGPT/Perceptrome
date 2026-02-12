import logging
import os
from typing import Dict, Tuple

try:
    import torch
    from torch import nn, optim
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    F = None  # type: ignore

from .config import IOConfig


class TransformerVAE(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for TransformerVAE.")
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(self.vocab_size, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(self.d_model * 4),
            dropout=float(dropout),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.fc_mu = nn.Linear(self.d_model, self.d_model)
        self.fc_logvar = nn.Linear(self.d_model, self.d_model)

        self.z_to_seq = nn.Linear(self.d_model, self.seq_len * self.d_model)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(self.d_model * 4),
            dropout=float(dropout),
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=int(num_layers))
        self.out_proj = nn.Linear(self.d_model, self.vocab_size)

    def _ensure_seq(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            return x.view(x.size(0), self.seq_len, self.vocab_size)
        return x

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        x_seq = self._ensure_seq(x)
        h = self.input_proj(x_seq) + self.pos_embed
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h = self.z_to_seq(z).view(z.size(0), self.seq_len, self.d_model)
        h = self.decoder(h + self.pos_embed)
        logits = self.out_proj(h)
        return logits.view(z.size(0), self.seq_len * self.vocab_size)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        return F.softmax(logits, dim=-1) if str(loss_type).lower() == "ce" else torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CNNVAE(nn.Module):  # type: ignore[misc]
    def __init__(self, seq_len: int, vocab_size: int, hidden_dim: int, dropout: float = 0.1):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for CNNVAE.")
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)

        ch1 = max(64, min(512, self.hidden_dim // 2))
        ch2 = max(128, min(1024, self.hidden_dim))

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(self.vocab_size, ch1, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(ch1),
            nn.Conv1d(ch1, ch2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(ch2),
            nn.Conv1d(ch2, ch2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_mu = nn.Linear(ch2, self.hidden_dim)
        self.fc_logvar = nn.Linear(ch2, self.hidden_dim)

        self.z_to_seq = nn.Linear(self.hidden_dim, ch2 * self.seq_len)
        self.decoder_cnn = nn.Sequential(
            nn.Conv1d(ch2, ch2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(ch2, ch1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(ch1, self.vocab_size, kernel_size=7, padding=3),
        )

    def _ensure_seq(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            return x.view(x.size(0), self.seq_len, self.vocab_size)
        return x

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        x_seq = self._ensure_seq(x).transpose(1, 2)
        h = self.encoder_cnn(x_seq)
        pooled = self.pool(h).squeeze(-1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h = self.z_to_seq(z).view(z.size(0), -1, self.seq_len)
        logits = self.decoder_cnn(h).transpose(1, 2)
        return logits.reshape(z.size(0), self.seq_len * self.vocab_size)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        return F.softmax(logits, dim=-1) if str(loss_type).lower() == "ce" else torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class DNABertVAE(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        kmer_size: int,
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for DNABertVAE.")
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.kmer_size = max(1, int(kmer_size))

        # DNABERT-style k-mer tokenization with explicit CLS token.
        self.kmer_vocab_size = int(self.vocab_size ** self.kmer_size)
        self.cls_token_id = self.kmer_vocab_size
        self.total_vocab = self.kmer_vocab_size + 1

        self.tokens_len = max(1, self.seq_len - self.kmer_size + 1)
        self.token_embed = nn.Embedding(self.total_vocab, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokens_len + 1, self.d_model))
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(self.d_model * 4),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.fc_mu = nn.Linear(self.d_model, self.d_model)
        self.fc_logvar = nn.Linear(self.d_model, self.d_model)

        self.z_to_seq = nn.Linear(self.d_model, self.seq_len * self.d_model)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(self.d_model * 4),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=int(num_layers))
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)
        self.out_proj = nn.Linear(self.d_model, self.vocab_size)

    def _ensure_seq(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            return x.view(x.size(0), self.seq_len, self.vocab_size)
        return x

    def _kmer_ids(self, token_ids: "torch.Tensor") -> "torch.Tensor":
        # token_ids: (B, L) with vocab values in [0, vocab_size)
        B, L = token_ids.shape
        if L < self.kmer_size:
            return token_ids[:, :1].clamp(0, self.kmer_vocab_size - 1)

        bases = token_ids.new_tensor(
            [self.vocab_size ** i for i in range(self.kmer_size - 1, -1, -1)],
            dtype=token_ids.dtype,
        )
        kmers = []
        for start in range(L - self.kmer_size + 1):
            seg = token_ids[:, start : start + self.kmer_size]
            k_ids = (seg * bases).sum(dim=1)
            kmers.append(k_ids)
        return torch.stack(kmers, dim=1)

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        x_seq = self._ensure_seq(x)
        token_ids = x_seq.argmax(dim=2)
        kmer_ids = self._kmer_ids(token_ids)

        tok = self.token_embed(kmer_ids)
        cls = self.cls_embed.expand(tok.size(0), -1, -1)
        h = torch.cat([cls, tok], dim=1)
        h = h + self.pos_embed[:, : h.size(1), :]
        h = self.encoder(h)
        cls_out = h[:, 0, :]
        return self.fc_mu(cls_out), self.fc_logvar(cls_out)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h = self.z_to_seq(z).view(z.size(0), self.seq_len, self.d_model)
        h = self.decoder(h + self.decoder_pos)
        logits = self.out_proj(h)
        return logits.view(z.size(0), self.seq_len * self.vocab_size)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        return F.softmax(logits, dim=-1) if str(loss_type).lower() == "ce" else torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class PlasmidVAE(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for PlasmidVAE.")
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, input_dim)
        self.act = nn.ReLU()

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        h = self.act(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h = self.act(self.fc2(z))
        return self.fc_out(h)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        return F.softmax(logits, dim=-1) if str(loss_type).lower() == "ce" else torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def get_device() -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _validate_checkpoint_meta(meta: Dict[str, object], tokenizer: str, seq_len: int, vocab_size: int, hidden_dim: int, loss_type: str, model_type: str, transformer_d_model: int, transformer_nhead: int, transformer_layers: int, transformer_dropout: float, dnabert_kmer: int) -> None:
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
    ck_kmer = int(meta.get("dnabert_kmer", dnabert_kmer))

    if ck_tok != tokenizer.lower():
        raise ValueError(f"Checkpoint tokenizer={ck_tok} but requested tokenizer={tokenizer}. Delete checkpoint or match settings.")
    if ck_seq != seq_len:
        raise ValueError(f"Checkpoint seq_len={ck_seq} but requested seq_len={seq_len}. Delete checkpoint or match settings.")
    if ck_vocab != vocab_size:
        raise ValueError(f"Checkpoint vocab_size={ck_vocab} but requested vocab_size={vocab_size}. Delete checkpoint or match settings.")
    if ck_loss != str(loss_type).lower():
        raise ValueError(f"Checkpoint loss_type={ck_loss} but requested loss_type={loss_type}. Delete checkpoint or match settings.")
    if ck_model_type != str(model_type).lower():
        raise ValueError(f"Checkpoint model_type={ck_model_type} but requested model_type={model_type}. Delete checkpoint or match settings.")

    mt = str(model_type).lower()
    if mt == "mlp" and ck_hidden != int(hidden_dim):
        raise ValueError(f"Checkpoint hidden_dim={ck_hidden} but requested hidden_dim={hidden_dim}. Delete checkpoint or match settings.")
    if mt in ("transformer", "dnabert"):
        if ck_d_model != int(transformer_d_model):
            raise ValueError(f"Checkpoint transformer_d_model={ck_d_model} but requested {transformer_d_model}. Delete checkpoint or match settings.")
        if ck_nhead != int(transformer_nhead):
            raise ValueError(f"Checkpoint transformer_nhead={ck_nhead} but requested {transformer_nhead}. Delete checkpoint or match settings.")
        if ck_layers != int(transformer_layers):
            raise ValueError(f"Checkpoint transformer_layers={ck_layers} but requested {transformer_layers}. Delete checkpoint or match settings.")
        if abs(ck_dropout - float(transformer_dropout)) > 1e-8:
            raise ValueError(f"Checkpoint transformer_dropout={ck_dropout} but requested {transformer_dropout}. Delete checkpoint or match settings.")
    if mt == "dnabert" and ck_kmer != int(dnabert_kmer):
        raise ValueError(f"Checkpoint dnabert_kmer={ck_kmer} but requested {dnabert_kmer}. Delete checkpoint or match settings.")


def _build_model(seq_len: int, vocab_size: int, hidden_dim: int, model_type: str, transformer_d_model: int, transformer_nhead: int, transformer_layers: int, transformer_dropout: float, dnabert_kmer: int) -> "nn.Module":
    mt = str(model_type).lower()
    input_dim = int(seq_len) * int(vocab_size)
    if mt == "transformer":
        return TransformerVAE(seq_len=seq_len, vocab_size=vocab_size, d_model=transformer_d_model, nhead=transformer_nhead, num_layers=transformer_layers, dropout=transformer_dropout)
    if mt == "cnn":
        return CNNVAE(seq_len=seq_len, vocab_size=vocab_size, hidden_dim=hidden_dim, dropout=transformer_dropout)
    if mt == "dnabert":
        return DNABertVAE(seq_len=seq_len, vocab_size=vocab_size, d_model=transformer_d_model, nhead=transformer_nhead, num_layers=transformer_layers, dropout=transformer_dropout, kmer_size=dnabert_kmer)
    return PlasmidVAE(input_dim=input_dim, hidden_dim=hidden_dim)


def load_or_init_model(
    io_cfg: IOConfig,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
    learning_rate: float,
    device: "torch.device",
    tokenizer: str,
    loss_type: str,
    model_type: str,
    transformer_d_model: int,
    transformer_nhead: int,
    transformer_layers: int,
    transformer_dropout: float,
    dnabert_kmer: int,
) -> Tuple["nn.Module", "optim.Optimizer", int, str]:
    if torch is None or nn is None or optim is None:
        raise RuntimeError("PyTorch is required.")

    ckpt_path = os.path.join(io_cfg.checkpoints_dir, "latest.pt")
    mt = str(model_type).lower()

    model = _build_model(
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        model_type=mt,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        dnabert_kmer=dnabert_kmer,
    ).to(device)
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))
    global_step = 0

    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location=device)
        meta: Dict[str, object] = data.get("meta", {})
        _validate_checkpoint_meta(
            meta=meta,
            tokenizer=tokenizer,
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            loss_type=loss_type,
            model_type=mt,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            transformer_dropout=transformer_dropout,
            dnabert_kmer=dnabert_kmer,
        )
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optim"])
        global_step = int(meta.get("global_step", 0))
        logging.info("Loaded checkpoint %s (model=%s, tokenizer=%s, step=%s)", ckpt_path, mt, tokenizer, global_step)
    else:
        logging.info(
            "Initializing new VAE (model=%s tokenizer=%s loss=%s seq_len=%s vocab=%s hidden=%s d_model=%s nhead=%s layers=%s dropout=%s dnabert_kmer=%s lr=%s)",
            mt,
            tokenizer,
            loss_type,
            seq_len,
            vocab_size,
            hidden_dim,
            transformer_d_model,
            transformer_nhead,
            transformer_layers,
            transformer_dropout,
            dnabert_kmer,
            learning_rate,
        )

    return model, optimizer, global_step, ckpt_path


def save_checkpoint(
    ckpt_path: str,
    model: "nn.Module",
    optimizer: "optim.Optimizer",
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
    dnabert_kmer: int,
) -> None:
    if torch is None:
        return
    payload = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "meta": {
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
            "dnabert_kmer": int(dnabert_kmer),
        },
    }
    tmp = ckpt_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, ckpt_path)
    logging.info("Saved checkpoint step=%s -> %s", global_step, ckpt_path)


def vae_loss(
    recon_logits: "torch.Tensor",
    x: "torch.Tensor",
    mu: "torch.Tensor",
    logvar: "torch.Tensor",
    beta_kl: float,
    loss_type: str,
    seq_len: int,
    vocab_size: int,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch required.")

    lt = str(loss_type).lower()
    if lt == "ce":
        if F is None:
            raise RuntimeError("PyTorch required.")
        logits = recon_logits.view(recon_logits.size(0), int(seq_len), int(vocab_size))
        targets = x.view(x.size(0), int(seq_len), int(vocab_size)).argmax(dim=2)
        recon_term = F.cross_entropy(logits.view(-1, int(vocab_size)), targets.view(-1), reduction="mean")
    else:
        recon = torch.sigmoid(recon_logits)
        recon_term = nn.MSELoss(reduction="mean")(recon, x)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_term + float(beta_kl) * kl
    return total, recon_term, kl
