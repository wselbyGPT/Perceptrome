from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    Dataset = object  # type: ignore


@dataclass
class DatasetSpec:
    path: str
    input_key: str = "input_ids"
    sample_id_key: str = "sample_ids"


class NPZPretrainDataset(Dataset):  # type: ignore[misc]
    """Loads tensorized pretraining arrays from a .npz archive."""

    def __init__(
        self,
        spec: DatasetSpec,
        transforms: Optional[List[Callable[[Mapping[str, Any]], Dict[str, Any]]]] = None,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for NPZPretrainDataset.")
        self.spec = spec
        self.payload = np.load(spec.path, allow_pickle=True)
        if spec.input_key not in self.payload:
            raise KeyError(f"Missing key {spec.input_key!r} in {spec.path}")
        self.transforms = transforms or []

    def __len__(self) -> int:
        return int(self.payload[self.spec.input_key].shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for key in self.payload.files:
            arr = self.payload[key]
            row[key] = arr[idx]

        for tr in self.transforms:
            row.update(tr(row))

        if self.spec.sample_id_key in row:
            row["sample_id"] = str(row[self.spec.sample_id_key])
        else:
            row["sample_id"] = str(idx)

        return row


def pad_1d_batch(values: List[np.ndarray], pad_value: int = 0) -> np.ndarray:
    max_len = max(int(v.shape[0]) for v in values)
    out = np.full((len(values), max_len), pad_value, dtype=np.int64)
    for i, v in enumerate(values):
        out[i, : v.shape[0]] = v
    return out


def pretrain_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for pretrain_collate.")
    keys = set().union(*[b.keys() for b in batch])
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [b.get(k) for b in batch]
        if vals[0] is None:
            continue
        if isinstance(vals[0], str):
            out[k] = vals
            continue
        if isinstance(vals[0], np.ndarray):
            if vals[0].ndim == 1:
                arr = pad_1d_batch(vals, pad_value=0)
                out[k] = torch.from_numpy(arr)
                if k in {"input_ids", "view_a", "view_b"}:
                    out[f"{k}_mask"] = (out[k] != 0).to(torch.int64)
            else:
                out[k] = torch.from_numpy(np.stack(vals, axis=0))
            continue
        if np.isscalar(vals[0]):
            out[k] = torch.tensor(vals)
            continue
        out[k] = vals
    return out
