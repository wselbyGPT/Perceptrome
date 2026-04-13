"""Latent space manipulation strategies for genome generation.

These operate on latent vectors *before* they are decoded into sequences,
enabling structured exploration of the model's learned representation.

Strategies
----------
- ``random``         : standard i.i.d. Gaussian sampling (baseline)
- ``interpolate``    : SLERP between two encoded reference sequences
- ``arithmetic``     : vector arithmetic for property transfer (A - B + C)
- ``gradient``       : gradient-guided optimization toward target properties
- ``walk``           : systematic walk along principal latent directions
- ``inpaint``        : fix known motif positions and regenerate the rest
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatentConfig:
    """Configuration for latent space generation strategies."""

    strategy: str = "random"              # random | interpolate | arithmetic | gradient | walk | inpaint
    latent_scale: float = 1.0

    # --- interpolation ---
    interpolate_steps: int = 10           # number of interpolation points
    interpolate_method: str = "slerp"     # slerp | lerp

    # --- arithmetic ---
    # Vectors A, B, C are provided at call time; result = A - B + C

    # --- gradient-guided ---
    optimize_steps: int = 50
    optimize_lr: float = 0.01
    optimize_property: str = "gc"         # gc | orf_count | reconstruction
    optimize_target: float = 0.5          # target value for the property

    # --- walk ---
    walk_steps: int = 10
    walk_dim: int = 0                     # which latent dimension to walk along
    walk_range: float = 3.0               # ±walk_range standard deviations

    # --- inpaint ---
    inpaint_mask_ratio: float = 0.3       # fraction of positions to regenerate


def default_latent_config() -> LatentConfig:
    return LatentConfig()


# ---------------------------------------------------------------------------
# Core latent operations
# ---------------------------------------------------------------------------

def random_latent(
    latent_dim: int,
    device: "torch.device",
    scale: float = 1.0,
    batch_size: int = 1,
) -> "torch.Tensor":
    """Standard Gaussian latent sampling (baseline)."""
    return torch.randn(batch_size, latent_dim, device=device) * scale


def encode_sequence(
    model: nn.Module,
    sequence_onehot: "torch.Tensor",
    device: "torch.device",
) -> "torch.Tensor":
    """Encode a one-hot sequence tensor into its latent mean vector.

    Parameters
    ----------
    model : VAE model with .encode() method
    sequence_onehot : (1, seq_len, vocab_size) or (1, seq_len * vocab_size)
    device : torch device

    Returns
    -------
    mu : (1, latent_dim)
    """
    x = sequence_onehot.to(device)
    if x.dim() == 3:
        x = x.view(1, -1)
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu


# ---------------------------------------------------------------------------
# SLERP / LERP interpolation
# ---------------------------------------------------------------------------

def _slerp(z1: "torch.Tensor", z2: "torch.Tensor", t: float) -> "torch.Tensor":
    """Spherical linear interpolation between two latent vectors.

    SLERP preserves the norm of latent vectors during interpolation,
    which is important for VAEs where the prior is a unit Gaussian —
    linear interpolation tends to pass through low-density regions.
    """
    z1_flat = z1.view(-1).float()
    z2_flat = z2.view(-1).float()
    norm1 = torch.norm(z1_flat)
    norm2 = torch.norm(z2_flat)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return (1.0 - t) * z1 + t * z2
    z1n = z1_flat / norm1
    z2n = z2_flat / norm2
    omega = torch.acos(torch.clamp(torch.dot(z1n, z2n), -1.0, 1.0))
    sin_omega = torch.sin(omega)
    if sin_omega.abs() < 1e-8:
        return (1.0 - t) * z1 + t * z2
    coeff1 = torch.sin((1.0 - t) * omega) / sin_omega
    coeff2 = torch.sin(t * omega) / sin_omega
    result = coeff1 * z1_flat + coeff2 * z2_flat
    # Scale to interpolated norm
    target_norm = (1.0 - t) * norm1 + t * norm2
    result = result * (target_norm / torch.norm(result).clamp(min=1e-8))
    return result.view_as(z1)


def _lerp(z1: "torch.Tensor", z2: "torch.Tensor", t: float) -> "torch.Tensor":
    """Linear interpolation between two latent vectors."""
    return (1.0 - t) * z1 + t * z2


def interpolate_latents(
    z_start: "torch.Tensor",
    z_end: "torch.Tensor",
    steps: int,
    method: str = "slerp",
) -> List["torch.Tensor"]:
    """Generate a sequence of interpolated latent vectors.

    Parameters
    ----------
    z_start, z_end : (1, latent_dim) tensors
    steps : number of interpolation points (including endpoints)
    method : "slerp" or "lerp"

    Returns
    -------
    List of (1, latent_dim) tensors
    """
    steps = max(2, int(steps))
    interp_fn = _slerp if method.lower() == "slerp" else _lerp
    return [interp_fn(z_start, z_end, t / (steps - 1)) for t in range(steps)]


# ---------------------------------------------------------------------------
# Latent arithmetic
# ---------------------------------------------------------------------------

def latent_arithmetic(
    z_a: "torch.Tensor",
    z_b: "torch.Tensor",
    z_c: "torch.Tensor",
    scale: float = 1.0,
) -> "torch.Tensor":
    """Compute z_a - z_b + z_c (property transfer in latent space).

    This is analogous to word2vec arithmetic: if z_a encodes a high-GC
    plasmid, z_b encodes a low-GC plasmid, and z_c encodes a target
    template, the result should be a version of z_c shifted toward
    higher GC content.

    Parameters
    ----------
    z_a, z_b, z_c : (1, latent_dim) encoded reference vectors
    scale : scaling factor for the transferred delta

    Returns
    -------
    (1, latent_dim) result vector
    """
    delta = z_a - z_b
    return z_c + scale * delta


# ---------------------------------------------------------------------------
# Gradient-guided latent optimization
# ---------------------------------------------------------------------------

def _gc_fraction_from_logits(logits: "torch.Tensor", vocab_size: int) -> "torch.Tensor":
    """Differentiable GC fraction estimate from decoder logits.

    Parameters
    ----------
    logits : (1, seq_len, vocab_size)
    vocab_size : token vocabulary size

    Returns
    -------
    Scalar tensor with estimated GC fraction.
    """
    import torch.nn.functional as F
    probs = F.softmax(logits, dim=-1)  # (1, L, V)
    if vocab_size == 4:
        # Base tokenizer: C=1, G=2
        gc_prob = probs[:, :, 1] + probs[:, :, 2]
    elif vocab_size >= 64:
        # Codon tokenizer: use pre-computed GC counts
        from .encoding_main import GC_COUNT_PER_TOKEN
        gc_weights = torch.tensor(
            [float(GC_COUNT_PER_TOKEN[i]) / 3.0 if i < len(GC_COUNT_PER_TOKEN) else 0.0
             for i in range(vocab_size)],
            dtype=logits.dtype, device=logits.device,
        )
        gc_prob = (probs * gc_weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    else:
        # AA or unknown — return 0.5 (neutral)
        return torch.tensor(0.5, dtype=logits.dtype, device=logits.device)
    return gc_prob.mean()


def _reconstruction_loss_from_z(model: nn.Module, z: "torch.Tensor",
                                 seq_len: int, vocab_size: int) -> "torch.Tensor":
    """Differentiable reconstruction quality: lower = more plausible latent."""
    logits = model.decode(z)
    logits_3d = logits.view(1, seq_len, vocab_size)
    import torch.nn.functional as F
    # Entropy of the softmax distribution — low entropy = confident reconstruction
    probs = F.softmax(logits_3d, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    return entropy


def gradient_optimize_latent(
    model: nn.Module,
    z_init: "torch.Tensor",
    seq_len: int,
    vocab_size: int,
    target_property: str,
    target_value: float,
    steps: int = 50,
    lr: float = 0.01,
) -> "torch.Tensor":
    """Optimize a latent vector toward a target property using gradients.

    The model stays frozen; only the latent vector is updated. This finds
    the nearest point in latent space that decodes to a sequence with the
    desired property.

    Parameters
    ----------
    model : VAE model (frozen)
    z_init : (1, latent_dim) starting point (e.g., from random or encode)
    seq_len, vocab_size : sequence dimensions
    target_property : "gc" | "reconstruction"
    target_value : target for the property (e.g., 0.5 for GC)
    steps : optimization steps
    lr : learning rate

    Returns
    -------
    Optimized (1, latent_dim) tensor (detached).
    """
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for step in range(steps):
        optimizer.zero_grad()
        logits = model.decode(z).view(1, seq_len, vocab_size)

        if target_property == "gc":
            prop_val = _gc_fraction_from_logits(logits, vocab_size)
            loss = (prop_val - target_value) ** 2
        elif target_property == "reconstruction":
            loss = _reconstruction_loss_from_z(model, z, seq_len, vocab_size)
        else:
            loss = _reconstruction_loss_from_z(model, z, seq_len, vocab_size)

        # Regularization: keep z near the prior
        prior_loss = 0.01 * z.pow(2).mean()
        total = loss + prior_loss
        total.backward()
        optimizer.step()

    # Restore model gradients
    for p in model.parameters():
        p.requires_grad_(True)

    return z.detach()


# ---------------------------------------------------------------------------
# Latent walk
# ---------------------------------------------------------------------------

def latent_walk(
    z_center: "torch.Tensor",
    dim: int,
    steps: int,
    walk_range: float = 3.0,
) -> List["torch.Tensor"]:
    """Walk along a single latent dimension from -range to +range.

    Useful for understanding what each latent dimension controls and for
    systematically exploring the space around a reference encoding.

    Parameters
    ----------
    z_center : (1, latent_dim) center point
    dim : which latent dimension to vary
    steps : number of points along the walk
    walk_range : walk from -walk_range to +walk_range σ

    Returns
    -------
    List of (1, latent_dim) tensors
    """
    steps = max(2, int(steps))
    latent_dim = z_center.size(-1)
    dim = int(dim) % latent_dim
    results = []
    for i in range(steps):
        t = -walk_range + 2.0 * walk_range * (i / (steps - 1))
        z = z_center.clone()
        z[0, dim] = float(t)
        results.append(z)
    return results


def latent_walk_pca(
    model: nn.Module,
    encoded_latents: "torch.Tensor",
    component: int = 0,
    steps: int = 10,
    walk_range: float = 3.0,
) -> List["torch.Tensor"]:
    """Walk along principal components of the encoded latent distribution.

    Given a batch of encoded latent vectors from training data, compute PCA
    and walk along the specified principal component. This explores the
    *most meaningful* directions of variation in the learned space.

    Parameters
    ----------
    model : unused (for API consistency)
    encoded_latents : (N, latent_dim) batch of mu vectors from encoded data
    component : which principal component to walk (0 = highest variance)
    steps : number of walk points
    walk_range : standard deviations to walk

    Returns
    -------
    List of (1, latent_dim) tensors
    """
    steps = max(2, int(steps))
    mean = encoded_latents.mean(dim=0, keepdim=True)
    centered = encoded_latents - mean
    # SVD for PCA
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    pc = Vh[min(component, Vh.size(0) - 1)]  # (latent_dim,)
    pc_std = S[min(component, S.size(0) - 1)] / (encoded_latents.size(0) ** 0.5)

    results = []
    for i in range(steps):
        t = -walk_range + 2.0 * walk_range * (i / (steps - 1))
        z = mean + t * float(pc_std) * pc.unsqueeze(0)
        results.append(z)
    return results


# ---------------------------------------------------------------------------
# Motif-constrained inpainting
# ---------------------------------------------------------------------------

def inpaint_latent(
    model: nn.Module,
    original_onehot: "torch.Tensor",
    mask: "torch.Tensor",
    device: "torch.device",
    latent_scale: float = 1.0,
    num_iterations: int = 5,
) -> "torch.Tensor":
    """Regenerate masked positions while preserving unmasked (motif) positions.

    This encodes the original, adds noise to the latent, decodes, then
    splices the original tokens back at unmasked positions. Repeating this
    iteratively lets the model harmonize the regenerated regions with the
    fixed motifs.

    Parameters
    ----------
    model : VAE model
    original_onehot : (1, seq_len, vocab_size) original sequence
    mask : (seq_len,) boolean tensor — True = regenerate, False = keep original
    device : torch device
    latent_scale : noise scale for the initial latent perturbation
    num_iterations : number of encode→decode→splice iterations

    Returns
    -------
    (1, seq_len * vocab_size) flat logits for the inpainted sequence
    """
    x = original_onehot.to(device)
    if x.dim() == 3:
        seq_len, vocab_size = x.size(1), x.size(2)
        x_flat = x.view(1, -1)
    else:
        raise ValueError("original_onehot must be (1, seq_len, vocab_size)")

    mask = mask.to(device).bool()
    current = x_flat.clone()

    for iteration in range(num_iterations):
        mu, logvar = model.encode(current)
        # Add decreasing noise to encourage convergence
        noise_scale = latent_scale * (1.0 - iteration / max(1, num_iterations))
        z = mu + torch.randn_like(mu) * noise_scale
        logits_flat = model.decode(z)
        logits_3d = logits_flat.view(1, seq_len, vocab_size)
        # Splice: keep original at unmasked, use generated at masked
        original_3d = x.view(1, seq_len, vocab_size)
        result_3d = torch.where(mask.unsqueeze(0).unsqueeze(-1), logits_3d, original_3d)
        current = result_3d.view(1, -1)

    return current


def create_inpaint_mask(
    seq_len: int,
    motif_positions: Optional[List[Tuple[int, int]]] = None,
    mask_ratio: float = 0.3,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """Create an inpainting mask.

    Parameters
    ----------
    seq_len : sequence length
    motif_positions : list of (start, end) ranges to KEEP (not regenerate).
                      If None, randomly mask mask_ratio of positions.
    mask_ratio : fraction of positions to regenerate (if motif_positions is None)
    device : torch device

    Returns
    -------
    (seq_len,) boolean tensor — True = regenerate, False = keep
    """
    if motif_positions is not None:
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        for start, end in motif_positions:
            mask[max(0, start):min(seq_len, end)] = False
        return mask
    else:
        mask = torch.rand(seq_len, device=device) < mask_ratio
        return mask


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def generate_latent_vectors(
    cfg: LatentConfig,
    model: nn.Module,
    latent_dim: int,
    device: "torch.device",
    reference_sequences: Optional[List["torch.Tensor"]] = None,
) -> List["torch.Tensor"]:
    """Generate one or more latent vectors according to the configured strategy.

    Parameters
    ----------
    cfg : LatentConfig
    model : VAE model
    latent_dim : latent dimensionality
    device : torch device
    reference_sequences : optional list of (1, seq_len * vocab_size) encoded
                          reference tensors (needed for interpolate, arithmetic)

    Returns
    -------
    List of (1, latent_dim) latent vectors ready for decoding.
    """
    refs = reference_sequences or []
    strategy = cfg.strategy.lower()

    if strategy == "interpolate":
        if len(refs) < 2:
            logging.warning("Interpolation needs >=2 reference sequences; falling back to random")
            return [random_latent(latent_dim, device, cfg.latent_scale)]
        z_start = encode_sequence(model, refs[0], device)
        z_end = encode_sequence(model, refs[1], device)
        return interpolate_latents(z_start, z_end, cfg.interpolate_steps, cfg.interpolate_method)

    elif strategy == "arithmetic":
        if len(refs) < 3:
            logging.warning("Arithmetic needs >=3 reference sequences; falling back to random")
            return [random_latent(latent_dim, device, cfg.latent_scale)]
        z_a = encode_sequence(model, refs[0], device)
        z_b = encode_sequence(model, refs[1], device)
        z_c = encode_sequence(model, refs[2], device)
        return [latent_arithmetic(z_a, z_b, z_c, cfg.latent_scale)]

    elif strategy == "walk":
        z_center = random_latent(latent_dim, device, cfg.latent_scale)
        if refs:
            z_center = encode_sequence(model, refs[0], device)
        return latent_walk(z_center, cfg.walk_dim, cfg.walk_steps, cfg.walk_range)

    elif strategy == "gradient":
        z_init = random_latent(latent_dim, device, cfg.latent_scale)
        if refs:
            z_init = encode_sequence(model, refs[0], device)
        # seq_len and vocab_size inferred from model output
        test_out = model.decode(z_init)
        total_out = test_out.size(-1)
        # Estimate seq_len from model attributes if available
        seq_len = getattr(model, 'seq_len', None)
        vocab_size = getattr(model, 'vocab_size', None)
        if seq_len is None or vocab_size is None:
            # Fallback: assume DNA
            vocab_size = 4
            seq_len = total_out // vocab_size
        z_opt = gradient_optimize_latent(
            model, z_init, seq_len, vocab_size,
            cfg.optimize_property, cfg.optimize_target,
            cfg.optimize_steps, cfg.optimize_lr,
        )
        return [z_opt]

    else:
        # Default: random
        return [random_latent(latent_dim, device, cfg.latent_scale)]
