import logging, os
from typing import Any, Callable, Dict, Mapping, Protocol, Tuple

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

class VAEModelProtocol(Protocol):
    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        ...

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        ...

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        ...

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        ...

    def parameters(self) -> Any:
        ...

    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> Any:
        ...


ModelFactory = Callable[[int, int, Mapping[str, Any], "torch.device"], VAEModelProtocol]
ModelConfig = Dict[str, Any]

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

def _get_required(model_config: Mapping[str, Any], key: str) -> Any:
    if key not in model_config or model_config[key] is None:
        raise KeyError(f"model_config missing required key '{key}'")
    return model_config[key]

def _normalize_model_config(model_type: str, model_config: Mapping[str, Any]) -> ModelConfig:
    mt = str(model_type).lower()
    cfg: ModelConfig = dict(model_config)
    if mt == "transformer":
        return {
            "d_model": int(_get_required(cfg, "d_model")),
            "nhead": int(_get_required(cfg, "nhead")),
            "num_layers": int(_get_required(cfg, "num_layers")),
            "dropout": float(_get_required(cfg, "dropout")),
        }
    if mt == "mlp":
        return {
            "hidden_dim": int(_get_required(cfg, "hidden_dim")),
        }
    return dict(cfg)

def _build_plasmid_vae(
    seq_len: int,
    vocab_size: int,
    model_config: Mapping[str, Any],
    device: "torch.device",
) -> PlasmidVAE:
    input_dim = int(seq_len) * int(vocab_size)
    hidden_dim = int(_get_required(model_config, "hidden_dim"))
    return PlasmidVAE(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

def _build_transformer_vae(
    seq_len: int,
    vocab_size: int,
    model_config: Mapping[str, Any],
    device: "torch.device",
) -> TransformerVAE:
    return TransformerVAE(
        seq_len=seq_len,
        vocab_size=vocab_size,
        d_model=int(_get_required(model_config, "d_model")),
        nhead=int(_get_required(model_config, "nhead")),
        num_layers=int(_get_required(model_config, "num_layers")),
        dropout=float(_get_required(model_config, "dropout")),
    ).to(device)

MODEL_REGISTRY: Dict[str, ModelFactory] = {
    "mlp": _build_plasmid_vae,
    "transformer": _build_transformer_vae,
}

def resolve_latent_dim(model_type: str, model_config: Mapping[str, Any]) -> int:
    cfg = _normalize_model_config(model_type, model_config)
    if str(model_type).lower() == "transformer":
        return int(cfg["d_model"])
    if str(model_type).lower() == "mlp":
        return int(cfg["hidden_dim"])
    raise ValueError(f"Latent dimension not defined for model_type '{model_type}'")

def build_model(
    model_type: str,
    model_config: Mapping[str, Any],
    seq_len: int,
    vocab_size: int,
    device: "torch.device",
) -> VAEModelProtocol:
    mt = str(model_type).lower()
    factory = MODEL_REGISTRY.get(mt)
    if factory is None:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {sorted(MODEL_REGISTRY)}")
    cfg = _normalize_model_config(mt, model_config)
    return factory(seq_len, vocab_size, cfg, device)

def get_device() -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_or_init_model(
    io_cfg: IOConfig,
    seq_len: int,
    vocab_size: int,
    learning_rate: float,
    device: "torch.device",
    tokenizer: str,
    loss_type: str,
    model_type: str,
    model_config: Mapping[str, Any],
) -> Tuple[VAEModelProtocol, "optim.Optimizer", int, str]:
    """
    seq_len: number of positions (bp or codons)
    vocab_size: 4 for base, 65 for codon
    """
    if torch is None or nn is None or optim is None:
        raise RuntimeError("PyTorch is required.")

    ckpt_path = os.path.join(io_cfg.checkpoints_dir, "latest.pt")

    mt = str(model_type).lower()
    normalized_config = _normalize_model_config(mt, model_config)
    model = build_model(mt, normalized_config, seq_len, vocab_size, device)
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    global_step = 0

    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location=device)
        meta: Dict[str, object] = data.get("meta", {})
        ck_tok = str(meta.get("tokenizer", "base")).lower()
        ck_seq = int(meta.get("seq_len", seq_len))
        ck_vocab = int(meta.get("vocab_size", vocab_size))
        ck_model_config_raw = meta.get("model_config", {})
        if not isinstance(ck_model_config_raw, dict):
            ck_model_config_raw = {}
        if mt == "transformer" and not ck_model_config_raw:
            ck_model_config_raw = {
                "d_model": meta.get("transformer_d_model"),
                "nhead": meta.get("transformer_nhead"),
                "num_layers": meta.get("transformer_layers"),
                "dropout": meta.get("transformer_dropout"),
            }
        if mt == "mlp" and not ck_model_config_raw:
            ck_model_config_raw = {
                "hidden_dim": meta.get("hidden_dim"),
            }
        ck_model_config_raw = {k: v for k, v in ck_model_config_raw.items() if v is not None}
        ck_model_config = _normalize_model_config(mt, {**normalized_config, **ck_model_config_raw})
        ck_loss = str(meta.get("loss_type", "mse")).lower()
        ck_model_type = str(meta.get("model_type", "mlp")).lower()
        
        if ck_tok != tokenizer.lower():
            raise ValueError(f"Checkpoint tokenizer={ck_tok} but requested tokenizer={tokenizer}. Delete {ckpt_path} or match settings.")
        if ck_seq != seq_len:
            raise ValueError(f"Checkpoint seq_len={ck_seq} but requested seq_len={seq_len}. Delete {ckpt_path} or match settings.")
        if ck_vocab != vocab_size:
            raise ValueError(f"Checkpoint vocab_size={ck_vocab} but requested vocab_size={vocab_size}. Delete {ckpt_path} or match settings.")
        if ck_loss != str(loss_type).lower():
            raise ValueError(
                f"Checkpoint loss_type={ck_loss} but requested loss_type={loss_type}. "
                f"Delete {ckpt_path} or match settings."
            )
        if ck_model_type != mt:
            raise ValueError(
                f"Checkpoint model_type={ck_model_type} but requested model_type={mt}. "
                f"Delete {ckpt_path} or match settings."
            )
        for key, value in normalized_config.items():
            ck_value = ck_model_config.get(key)
            if isinstance(value, float):
                if ck_value is None or abs(float(ck_value) - float(value)) > 1e-8:
                    raise ValueError(
                        f"Checkpoint model_config[{key}]={ck_value} but requested {value}. "
                        f"Delete {ckpt_path} or match settings."
                    )
            else:
                if ck_value is None or ck_value != value:
                    raise ValueError(
                        f"Checkpoint model_config[{key}]={ck_value} but requested {value}. "
                        f"Delete {ckpt_path} or match settings."
                    )

        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optim"])
        global_step = int(meta.get("global_step", 0))

        logging.info(
            "Loaded checkpoint %s (tokenizer=%s, seq_len=%s, vocab=%s, hidden=%s, model=%s, step=%s)",
            ckpt_path, ck_tok, ck_seq, ck_vocab, ck_model_config.get("hidden_dim"), ck_model_type, global_step
        )
        logging.info(
            "Initializing new VAE (tokenizer=%s, loss_type=%s, model=%s, "
            "seq_len=%s, vocab=%s, model_config=%s, lr=%s)",
            tokenizer,
            loss_type,
            mt,
            seq_len,
            vocab_size,
            normalized_config,
            learning_rate,
        )

    return model, optimizer, global_step, ckpt_path

def save_checkpoint(
    ckpt_path: str,
    model: VAEModelProtocol,
    optimizer: "optim.Optimizer",
    global_step: int,
    tokenizer: str,
    seq_len: int,
    vocab_size: int,
    loss_type: str,
    model_type: str,
    model_config: Mapping[str, Any],
) -> None:
    if torch is None:
        return
    normalized_config = _normalize_model_config(model_type, model_config)
    payload = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "meta": {
            "global_step": int(global_step),
            "tokenizer": str(tokenizer).lower(),
            "seq_len": int(seq_len),
            "vocab_size": int(vocab_size),
            "loss_type": str(loss_type).lower(),
            "model_type": str(model_type).lower(),
            "model_config": normalized_config,
        },
    }
    tmp = ckpt_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, ckpt_path)
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
