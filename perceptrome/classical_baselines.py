"""Classical ML baselines for genomic modeling.

This module provides a lightweight tree-based baseline API that can be trained
alongside neural methods for benchmark comparisons.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for classical baselines.")


class _MajorityBaseline:
    """Dependency-free fallback classifier."""

    def __init__(self):
        self._p1: float = 0.5

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_MajorityBaseline":
        _require_numpy()
        if x.shape[0] == 0:
            self._p1 = 0.5
            return self
        self._p1 = float(np.mean(y.astype(np.float32)))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _require_numpy()
        p1 = np.full((x.shape[0],), self._p1, dtype=np.float32)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(np.int64)


class TreeSequenceBaseline:
    """Tree baseline with automatic backend selection.

    Backends (in order):
    1) xgboost.XGBClassifier
    2) sklearn.ensemble.HistGradientBoostingClassifier
    3) dependency-free majority baseline
    """

    def __init__(
        self,
        backend: Literal["auto", "xgboost", "sklearn", "majority"] = "auto",
        random_state: int = 42,
    ):
        self.backend = backend
        self.random_state = int(random_state)
        self.model: Optional[object] = None
        self.backend_used: Optional[str] = None

    def _init_backend(self) -> None:
        if self.model is not None:
            return

        options = [self.backend] if self.backend != "auto" else ["xgboost", "sklearn", "majority"]

        for candidate in options:
            if candidate == "xgboost":
                try:
                    from xgboost import XGBClassifier  # type: ignore

                    self.model = XGBClassifier(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=self.random_state,
                    )
                    self.backend_used = "xgboost"
                    return
                except Exception:
                    continue

            if candidate == "sklearn":
                try:
                    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore

                    self.model = HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=63,
                        random_state=self.random_state,
                    )
                    self.backend_used = "sklearn"
                    return
                except Exception:
                    continue

            if candidate == "majority":
                self.model = _MajorityBaseline()
                self.backend_used = "majority"
                return

        self.model = _MajorityBaseline()
        self.backend_used = "majority"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TreeSequenceBaseline":
        _require_numpy()
        self._init_backend()
        assert self.model is not None

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        self.model.fit(x, y)  # type: ignore[attr-defined]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _require_numpy()
        self._init_backend()
        assert self.model is not None

        x = np.asarray(x, dtype=np.float32)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)  # type: ignore[attr-defined]
            return np.asarray(probs, dtype=np.float32)

        pred = np.asarray(self.model.predict(x), dtype=np.float32)  # type: ignore[attr-defined]
        return np.stack([1.0 - pred, pred], axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(np.int64)

    def save(self, path: str) -> None:
        payload = {
            "backend": self.backend,
            "backend_used": self.backend_used,
            "random_state": self.random_state,
            "model": self.model,
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "TreeSequenceBaseline":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        obj = cls(backend=payload.get("backend", "auto"), random_state=int(payload.get("random_state", 42)))
        obj.backend_used = payload.get("backend_used")
        obj.model = payload.get("model")
        return obj
