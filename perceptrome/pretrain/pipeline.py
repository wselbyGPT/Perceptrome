from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

try:
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    DataLoader = None  # type: ignore

from .datasets import DatasetSpec, NPZPretrainDataset, pretrain_collate
from perceptrome.jobs.provenance import collect_and_write_provenance, resolve_seed, set_global_seeds
from perceptrome.run_layout import ensure_run_layout
from .models import BackboneConfig, SequenceBackbone
from .objectives import ContrastiveObjective, MaskedSMEObjective, MaskedTokenObjective, ObjectiveWeights
from .runner import PretrainRunner, RunnerConfig
from .transforms import ContrastivePairTransform, MaskSMETransform, MaskedLanguageModelTransform


@dataclass
class PretrainPipelineConfig:
    dataset_path: str
    vocab_size: int
    batch_size: int = 16
    epochs: int = 1
    hidden_size: int = 256
    lr: float = 1e-4
    output_dir: str = "model/pretrain"
    enable_mlm: bool = True
    enable_sme: bool = True
    enable_contrastive: bool = True
    seed: int | None = None
    provenance_inputs: Dict[str, str] | None = None


def _build_row_transforms(cfg: PretrainPipelineConfig) -> List[Any]:
    transforms: List[Any] = []

    mlm = MaskedLanguageModelTransform(vocab_size=int(cfg.vocab_size))
    sme = MaskSMETransform()
    con = ContrastivePairTransform()

    def _row_transform(row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        input_ids = row.get("input_ids")
        if cfg.enable_mlm and input_ids is not None:
            out.update(mlm(input_ids))
        if cfg.enable_sme and all(k in row for k in ("sme_s", "sme_m", "sme_e")):
            out.update(sme(row["sme_s"], row["sme_m"], row["sme_e"]))
        if cfg.enable_contrastive and input_ids is not None:
            out.update(con(input_ids))
        return out

    transforms.append(_row_transform)
    return transforms


def run_pretraining(cfg: PretrainPipelineConfig) -> Dict[str, float]:
    seed_info = resolve_seed(cfg.seed)
    set_global_seeds(int(seed_info["value"]))
    layout = ensure_run_layout()
    paths = {"dataset": str(cfg.dataset_path)}
    if cfg.provenance_inputs:
        paths.update({str(k): str(v) for k, v in cfg.provenance_inputs.items()})
    collect_and_write_provenance(
        layout=layout,
        run_kind="pretrain",
        seed_info=seed_info,
        input_paths=paths,
        extra={"vocab_size": int(cfg.vocab_size), "batch_size": int(cfg.batch_size), "epochs": int(cfg.epochs)},
    )

    dataset = NPZPretrainDataset(DatasetSpec(path=cfg.dataset_path), transforms=_build_row_transforms(cfg))
    loader = DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=pretrain_collate)

    backbone = SequenceBackbone(BackboneConfig(vocab_size=int(cfg.vocab_size), hidden_size=int(cfg.hidden_size)))
    objectives = {}
    if cfg.enable_mlm:
        objectives["masked_token"] = MaskedTokenObjective(hidden_size=backbone.get_hidden_size(), vocab_size=int(cfg.vocab_size))
    if cfg.enable_sme:
        objectives["masked_sme"] = MaskedSMEObjective(hidden_size=backbone.get_hidden_size())
    if cfg.enable_contrastive:
        objectives["contrastive"] = ContrastiveObjective(hidden_size=backbone.get_hidden_size())

    runner = PretrainRunner(
        backbone=backbone,
        objectives=objectives,
        objective_weights=ObjectiveWeights(),
        cfg=RunnerConfig(epochs=int(cfg.epochs), lr=float(cfg.lr), output_dir=str(cfg.output_dir)),
    )
    metrics = runner.train(loader)
    metrics["seed"] = float(seed_info["value"])
    return metrics
