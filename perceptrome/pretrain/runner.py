from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore

from .interfaces import PretrainBackbone
from .objectives import ObjectiveWeights
from perceptrome.jobs.artifact_index import build_artifact_entry
from perceptrome.run_layout import ensure_run_layout, path_in_run, update_run_manifest


@dataclass
class RunnerConfig:
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    log_every: int = 20
    checkpoint_every: int = 200
    output_dir: str = "model/pretrain"


class PretrainRunner:
    def __init__(
        self,
        backbone: PretrainBackbone,
        objectives: Dict[str, nn.Module],
        objective_weights: Optional[ObjectiveWeights] = None,
        cfg: Optional[RunnerConfig] = None,
        device: Optional[str] = None,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for pretraining.")
        self.backbone = backbone
        self.objectives = objectives
        self.objective_weights = objective_weights or ObjectiveWeights()
        self.cfg = cfg or RunnerConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.backbone.to(self.device)
        for mod in self.objectives.values():
            mod.to(self.device)

        params = list(self.backbone.parameters())
        for m in self.objectives.values():
            params.extend(list(m.parameters()))
        self.optimizer = torch.optim.AdamW(params, lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay))
        self.global_step = 0
        self.checkpoint_paths: list[str] = []

    def _to_device(self, batch: Dict[str, object]) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for k, v in batch.items():
            if hasattr(v, "to"):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _total_loss(self, losses: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        w = self.objective_weights
        total = losses.get("masked_token", 0.0) * float(w.mlm)
        total = total + losses.get("masked_sme", 0.0) * float(w.sme)
        total = total + losses.get("contrastive", 0.0) * float(w.contrastive)
        return total

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        layout = ensure_run_layout()
        self.cfg.output_dir = path_in_run(layout, "artifacts", "pretrain")
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        metrics_json_path = path_in_run(layout, "metrics", "pretraining_final_metrics.json")
        checkpoint_index_path = path_in_run(layout, "artifacts", "pretraining_checkpoint_index.json")
        update_run_manifest(
            layout,
            paths={"pretrain": {"output_dir": self.cfg.output_dir, "final_metrics_json": metrics_json_path, "checkpoint_index_json": checkpoint_index_path}},
            checkpoints={"pretrain": {"output_dir": self.cfg.output_dir, "index_json": checkpoint_index_path}},
            artifacts=[build_artifact_entry(artifact_id="pretrain_output_dir", role="pretrain.output_dir", path=self.cfg.output_dir, artifact_type="directory")],
        )
        for epoch in range(int(self.cfg.epochs)):
            self.backbone.train()
            for obj in self.objectives.values():
                obj.train()

            for batch in train_loader:
                self.global_step += 1
                batch = self._to_device(batch)

                enc = self.backbone.encode(batch)
                if "contrastive" in self.objectives and "view_b" in batch:
                    view_b = {"input_ids": batch["view_b"], "input_ids_mask": batch.get("view_b_mask")}
                    enc_pos = self.backbone.encode(view_b)
                    batch["contrastive_pos"] = enc_pos.pooled_embedding.detach()

                losses: Dict[str, torch.Tensor] = {}
                log_row: Dict[str, float] = {}
                for name, obj in self.objectives.items():
                    loss, obj_metrics = obj(enc, batch)
                    losses[name] = loss
                    log_row[f"loss/{name}"] = float(loss.detach())
                    for mk, mv in obj_metrics.items():
                        log_row[f"{name}/{mk}"] = float(mv)

                total = self._total_loss(losses)
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), float(self.cfg.grad_clip_norm))
                self.optimizer.step()

                log_row["loss/total"] = float(total.detach())
                metrics = log_row

                if self.global_step % int(self.cfg.log_every) == 0:
                    print(json.dumps({"step": self.global_step, **log_row}, sort_keys=True))
                if self.global_step % int(self.cfg.checkpoint_every) == 0:
                    ckpt_path = os.path.join(self.cfg.output_dir, f"step_{self.global_step}.pt")
                    self.save_checkpoint(ckpt_path)
                    self.checkpoint_paths.append(ckpt_path)
                    update_run_manifest(
                        layout,
                        paths={"pretrain": {"latest_step_checkpoint": ckpt_path}},
                        checkpoints={"pretrain": {"latest_step_checkpoint": ckpt_path, "count": len(self.checkpoint_paths)}},
                        artifacts=[build_artifact_entry(artifact_id=f"step_checkpoint_{self.global_step}", role="pretrain.checkpoint", path=ckpt_path, artifact_type="pytorch_checkpoint")],
                    )

            epoch_ckpt = os.path.join(self.cfg.output_dir, f"epoch_{epoch + 1}.pt")
            self.save_checkpoint(epoch_ckpt)
            self.checkpoint_paths.append(epoch_ckpt)
            update_run_manifest(
                layout,
                paths={"pretrain": {"latest_epoch_checkpoint": epoch_ckpt}},
                checkpoints={"pretrain": {"latest_epoch_checkpoint": epoch_ckpt, "count": len(self.checkpoint_paths)}},
                artifacts=[build_artifact_entry(artifact_id=f"epoch_checkpoint_{epoch + 1}", role="pretrain.checkpoint", path=epoch_ckpt, artifact_type="pytorch_checkpoint")],
            )
            if val_loader is not None:
                self.validate(val_loader)

        checkpoint_index = {
            "count": len(self.checkpoint_paths),
            "checkpoints": list(self.checkpoint_paths),
            "latest": self.checkpoint_paths[-1] if self.checkpoint_paths else None,
        }
        with open(checkpoint_index_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_index, f, indent=2, sort_keys=True)
            f.write("\n")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")
        update_run_manifest(
            layout,
            pretraining_metrics={"pretrain": dict(metrics)},
            checkpoints={"pretrain": {**checkpoint_index, "index_json": checkpoint_index_path}},
            paths={"pretrain": {"final_metrics_json": metrics_json_path, "checkpoint_index_json": checkpoint_index_path}},
            artifacts=[
                build_artifact_entry(artifact_id="pretrain_checkpoint_index", role="pretrain.checkpoint_index", path=checkpoint_index_path, artifact_type="json"),
                build_artifact_entry(artifact_id="pretrain_final_metrics", role="pretrain.metrics", path=metrics_json_path, artifact_type="json"),
            ],
        )
        metrics["checkpoint_index_json"] = checkpoint_index_path
        metrics["final_metrics_json"] = metrics_json_path
        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.backbone.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                enc = self.backbone.encode(batch)
                total = 0.0
                for name, obj in self.objectives.items():
                    loss, _ = obj(enc, batch)
                    total += float(loss.detach())
                vals.append(total)
        score = float(sum(vals) / max(1, len(vals)))
        print(json.dumps({"val/loss_total": score}, sort_keys=True))
        layout = ensure_run_layout()
        update_run_manifest(layout, metrics={"pretrain": {"val/loss_total": score}}, pretraining_metrics={"pretrain": {"val/loss_total": score}})
        return {"val/loss_total": score}

    def save_checkpoint(self, path: str) -> None:
        payload = {
            "global_step": int(self.global_step),
            "runner_config": asdict(self.cfg),
            "objective_weights": asdict(self.objective_weights),
            "backbone_state": self.backbone.state_dict(),
            "objective_states": {k: v.state_dict() for k, v in self.objectives.items()},
            "optimizer_state": self.optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.global_step = int(payload.get("global_step", 0))
        self.backbone.load_state_dict(payload["backbone_state"])
        for k, v in payload.get("objective_states", {}).items():
            if k in self.objectives:
                self.objectives[k].load_state_dict(v)
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
