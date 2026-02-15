import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

try:
    import torch
    from torch import nn, optim
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    optim = None  # type: ignore
    F = None      # type: ignore

from .config import IOConfig


def _checkpoint_meta_path(ckpt_path: str) -> str:
    return f"{ckpt_path}.meta.json"


def _load_sidecar_metadata(ckpt_path: str) -> Dict[str, Any]:
    meta_path = _checkpoint_meta_path(ckpt_path)
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint metadata at {meta_path} is not a JSON object.")
    return data


def _build_checkpoint_metadata(
    *,
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
    source_modality: str,
    proteome_flags: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "global_step": int(global_step),
        "training_step": int(global_step),
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
        "source_modality": str(source_modality).lower(),
        "proteome_flags": dict(proteome_flags),
    }


def _require_match(field: str, expected: Any, actual: Any, ckpt_path: str) -> None:
    if expected != actual:
        raise ValueError(
            f"Checkpoint metadata mismatch for {field}: checkpoint={actual!r}, current={expected!r}. "
            f"Delete {ckpt_path} (and sidecar metadata) or match settings."
        )


def _validate_checkpoint_metadata(
    ckpt_path: str,
    current: Dict[str, Any],
    checkpoint_meta: Dict[str, Any],
) -> None:
    _require_match("tokenizer", str(current["tokenizer"]).lower(), str(checkpoint_meta.get("tokenizer", "base")).lower(), ckpt_path)
    _require_match("seq_len", int(current["seq_len"]), int(checkpoint_meta.get("seq_len", current["seq_len"])), ckpt_path)
    _require_match("vocab_size", int(current["vocab_size"]), int(checkpoint_meta.get("vocab_size", current["vocab_size"])), ckpt_path)
    _require_match("loss_type", str(current["loss_type"]).lower(), str(checkpoint_meta.get("loss_type", "mse")).lower(), ckpt_path)
    _require_match("model_type", str(current["model_type"]).lower(), str(checkpoint_meta.get("model_type", "mlp")).lower(), ckpt_path)
    _require_match("source_modality", str(current["source_modality"]).lower(), str(checkpoint_meta.get("source_modality", "fasta")).lower(), ckpt_path)

    if str(current["model_type"]).lower() != "transformer":
        _require_match("hidden_dim", int(current["hidden_dim"]), int(checkpoint_meta.get("hidden_dim", current["hidden_dim"])), ckpt_path)
    else:
        _require_match("transformer_d_model", int(current["transformer_d_model"]), int(checkpoint_meta.get("transformer_d_model", current["transformer_d_model"])), ckpt_path)
        _require_match("transformer_nhead", int(current["transformer_nhead"]), int(checkpoint_meta.get("transformer_nhead", current["transformer_nhead"])), ckpt_path)
        _require_match("transformer_layers", int(current["transformer_layers"]), int(checkpoint_meta.get("transformer_layers", current["transformer_layers"])), ckpt_path)
        _require_match("transformer_dropout", float(current["transformer_dropout"]), float(checkpoint_meta.get("transformer_dropout", current["transformer_dropout"])), ckpt_path)

    ck_flags = checkpoint_meta.get("proteome_flags", {})
    if not isinstance(ck_flags, dict):
        raise ValueError(f"Checkpoint metadata mismatch: proteome_flags is invalid in {ckpt_path} sidecar.")
    _require_match("proteome_flags", dict(current["proteome_flags"]), ck_flags, ckpt_path)

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
        lt = str(loss_type).lower()
        if lt == "ce":
            return F.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar

class PlasmidVAE(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for PlasmidVAE.")
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        h = self.act(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        """Return *logits* (not probabilities).

        - For MSE-based training, we apply sigmoid to these logits in the loss.
        - For categorical training (CE), these logits are fed directly to softmax/CE.
        """
        h = self.act(self.fc2(z))
        return self.fc_out(h)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        """Return probabilities shaped (B, seq_len, vocab_size)."""
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        lt = str(loss_type).lower()
        if lt == "ce":
            return F.softmax(logits, dim=-1)
        # mse (legacy): independent sigmoid weights
        return torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar

def get_device() -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    source_modality: str = "fasta",
    proteome_flags: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, "optim.Optimizer", int, str]:
    """
    seq_len: number of positions (bp or codons)
    vocab_size: 4 for base, 65 for codon
    """
    if torch is None or nn is None or optim is None:
        raise RuntimeError("PyTorch is required.")

    ckpt_path = os.path.join(io_cfg.checkpoints_dir, "latest.pt")

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
        ).to(device)
    else:
        model = PlasmidVAE(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    global_step = 0

    current_meta = {
        "tokenizer": tokenizer,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "loss_type": loss_type,
        "model_type": mt,
        "transformer_d_model": transformer_d_model,
        "transformer_nhead": transformer_nhead,
        "transformer_layers": transformer_layers,
        "transformer_dropout": transformer_dropout,
        "source_modality": source_modality,
        "proteome_flags": dict(proteome_flags or {}),
    }

    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location=device)
        meta: Dict[str, object] = data.get("meta", {})
        sidecar_meta = _load_sidecar_metadata(ckpt_path)
        checkpoint_meta = dict(meta)
        checkpoint_meta.update(sidecar_meta)
        _validate_checkpoint_metadata(ckpt_path, current_meta, checkpoint_meta)

        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optim"])
        global_step = int(meta.get("global_step", 0))

        logging.info(
            "Loaded checkpoint %s (tokenizer=%s, seq_len=%s, vocab=%s, hidden=%s, model=%s, step=%s)",
            ckpt_path,
            checkpoint_meta.get("tokenizer"),
            checkpoint_meta.get("seq_len"),
            checkpoint_meta.get("vocab_size"),
            checkpoint_meta.get("hidden_dim"),
            checkpoint_meta.get("model_type"),
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

    return model, optimizer, global_step, ckpt_path

def save_checkpoint(
    ckpt_path: str,
    model: PlasmidVAE,
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
    source_modality: str = "fasta",
    proteome_flags: Optional[Dict[str, Any]] = None,
) -> None:
    if torch is None:
        return
    metadata = _build_checkpoint_metadata(
        global_step=global_step,
        tokenizer=tokenizer,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        loss_type=loss_type,
        model_type=model_type,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        source_modality=source_modality,
        proteome_flags=dict(proteome_flags or {}),
    )
    payload = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "meta": metadata,
    }
    tmp = ckpt_path + ".tmp"
    meta_tmp = _checkpoint_meta_path(ckpt_path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, ckpt_path)
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(meta_tmp, _checkpoint_meta_path(ckpt_path))
    logging.info(f"Saved checkpoint step={global_step} -> {ckpt_path}")

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
        # logits: (B, L*V) -> (B, L, V)
        logits = recon_logits.view(recon_logits.size(0), int(seq_len), int(vocab_size))
        targets = x.view(x.size(0), int(seq_len), int(vocab_size)).argmax(dim=2)  # (B, L)
        # per-position CE, mean over positions and batch
        ce = F.cross_entropy(logits.view(-1, int(vocab_size)), targets.view(-1), reduction="mean")
        recon_term = ce
    else:
        # legacy: regression on one-hot using sigmoid weights
        recon = torch.sigmoid(recon_logits)
        recon_term = nn.MSELoss(reduction="mean")(recon, x)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_term + float(beta_kl) * kl
    return total, recon_term, kl
