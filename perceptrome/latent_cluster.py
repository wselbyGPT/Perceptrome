from __future__ import annotations

import csv
import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LatentClusterRecord:
    accession: str
    cluster_id: int
    umap_x: float
    umap_y: float
    mu_norm: float
    mu_mean: float
    n_windows: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatentClusterSummary:
    run_id: str
    catalog_path: str
    n_accessions: int
    n_clusters: int
    latent_dim: int
    umap_n_neighbors: int
    umap_min_dist: float
    failed_accessions: List[str]
    cluster_sizes: Dict[str, int]
    started_at: str
    completed_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


_TSV_COLUMNS = [
    "accession", "cluster_id", "umap_x", "umap_y", "mu_norm", "mu_mean", "n_windows",
]


def write_cluster_records_json(path: str, records: List[LatentClusterRecord]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"records": [r.to_dict() for r in records]}, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_cluster_records_tsv(path: str, records: List[LatentClusterRecord]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_COLUMNS, dialect="excel-tab")
        writer.writeheader()
        for r in records:
            d = r.to_dict()
            writer.writerow({k: d.get(k) for k in _TSV_COLUMNS})
    return path


def write_cluster_summary_json(
    path: str,
    summary: LatentClusterSummary,
    records: List[LatentClusterRecord],
    centroids: np.ndarray,
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = summary.to_dict()
    payload["records"] = [r.to_dict() for r in records]
    payload["centroids"] = centroids.tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def run_kmeans(
    mu_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (labels shape (N,), centroids shape (k, latent_dim))."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "scikit-learn is required for clustering. "
            "Install it with: pip install 'perceptrome[analysis]' or pip install scikit-learn"
        )
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(mu_matrix)
    return labels.astype(np.int32), km.cluster_centers_.astype(np.float32)


def run_umap(
    mu_matrix: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Returns (N, 2) float32 projection."""
    try:
        import umap as umap_module
    except ImportError:
        raise ImportError(
            "umap-learn is required for projection. "
            "Install it with: pip install 'perceptrome[analysis]' or pip install umap-learn"
        )
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(mu_matrix).astype(np.float32)
