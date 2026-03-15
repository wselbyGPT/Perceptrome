from typing import Dict, Optional, Tuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


def hierarchical_loss_bundle(
    recon_total: "torch.Tensor",
    recon_term: "torch.Tensor",
    kl_term: "torch.Tensor",
    critic_outputs: Optional[Dict[str, "torch.Tensor"]] = None,
    property_targets: Optional[Dict[str, "torch.Tensor"]] = None,
    property_weight: float = 0.2,
) -> Tuple["torch.Tensor", Dict[str, float]]:
    total = recon_total
    metrics = {
        "recon": float(recon_term.item()),
        "kl": float(kl_term.item()),
        "property": 0.0,
    }
    if critic_outputs and property_targets and nn is not None:
        mse = nn.MSELoss(reduction="mean")
        prop = 0.0
        for key, pred in critic_outputs.items():
            target = property_targets.get(key)
            if target is not None:
                prop = prop + mse(pred, target.to(dtype=pred.dtype))
        total = total + float(property_weight) * prop
        metrics["property"] = float(prop.item()) if hasattr(prop, "item") else float(prop)
    return total, metrics
