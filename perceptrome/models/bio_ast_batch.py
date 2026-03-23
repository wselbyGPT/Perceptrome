from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from perceptrome.bio_ast import CONTAINMENT_EDGE_KIND
from perceptrome.encoding.bio_ast_builder import NODE_TYPE_TO_INT


def _strand_to_int(value: Any) -> int:
    if value in {"+", 1, "+1"}:
        return 1
    if value in {"-", -1, "-1"}:
        return -1
    return 0


@dataclass(frozen=True)
class BioASTBatch:
    node_type_ids: "torch.Tensor"
    span_coords: "torch.Tensor"
    strand: "torch.Tensor"
    frame: "torch.Tensor"
    tree_edge_index: "torch.Tensor"
    node_mask: Optional["torch.Tensor"] = None
    semantic_edge_index: Optional["torch.Tensor"] = None

    @classmethod
    def from_numpy_dict(cls, payload: Mapping[str, np.ndarray]) -> "BioASTBatch":
        if torch is None:
            raise RuntimeError("PyTorch is required for BioASTBatch.")
        node_type_ids = torch.as_tensor(payload["node_type_ids"], dtype=torch.long)
        span_coords = torch.as_tensor(payload["span_coords"], dtype=torch.float32)
        strand = torch.as_tensor(payload["strand"], dtype=torch.long)
        frame = torch.as_tensor(payload["frame"], dtype=torch.long)
        tree_edge_index = torch.as_tensor(payload["tree_edge_index"], dtype=torch.long)
        node_mask_arr = payload.get("node_mask")
        semantic_edge_index_arr = payload.get("semantic_edge_index")
        return cls(
            node_type_ids=node_type_ids,
            span_coords=span_coords,
            strand=strand,
            frame=frame,
            tree_edge_index=tree_edge_index,
            node_mask=torch.as_tensor(node_mask_arr, dtype=torch.bool) if node_mask_arr is not None else None,
            semantic_edge_index=torch.as_tensor(semantic_edge_index_arr, dtype=torch.long) if semantic_edge_index_arr is not None else None,
        )

    @classmethod
    def from_canonical_json(cls, payload: Mapping[str, Any], include_semantic_edges: bool = False) -> "BioASTBatch":
        return cls.from_numpy_dict(canonical_bio_ast_json_to_numpy(payload, include_semantic_edges=include_semantic_edges))

    @classmethod
    def from_canonical_json_path(cls, path: str | Path, include_semantic_edges: bool = False) -> "BioASTBatch":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_canonical_json(payload, include_semantic_edges=include_semantic_edges)

    def as_model_inputs(self) -> Dict[str, "torch.Tensor"]:
        out = {
            "node_type_ids": self.node_type_ids,
            "coords": self.span_coords,
            "span_coords": self.span_coords,
            "strand": self.strand,
            "frame": self.frame,
            "edge_index": self.tree_edge_index,
            "tree_edge_index": self.tree_edge_index,
        }
        if self.node_mask is not None:
            out["node_mask"] = self.node_mask
        if self.semantic_edge_index is not None:
            out["semantic_edge_index"] = self.semantic_edge_index
        return out


def canonical_bio_ast_json_to_numpy(payload: Mapping[str, Any], include_semantic_edges: bool = False) -> Dict[str, np.ndarray]:
    nodes = list(payload.get("nodes", []))
    edges = list(payload.get("edges", []))
    node_index = {str(node.get("canonical_id", "")): idx for idx, node in enumerate(nodes)}

    node_type_ids = np.asarray([NODE_TYPE_TO_INT.get(str(node.get("node_type", "")).lower(), -1) for node in nodes], dtype=np.int64)
    span_coords = np.asarray(
        [[int(node.get("start", -1) or -1), int(node.get("end", -1) or -1)] for node in nodes],
        dtype=np.float32,
    )
    strand = np.asarray([_strand_to_int(node.get("strand")) for node in nodes], dtype=np.int64)
    frame = np.asarray([int(node.get("frame", -1) if node.get("frame") is not None else -1) for node in nodes], dtype=np.int64)

    tree_pairs = []
    semantic_pairs = []
    for edge in edges:
        src = node_index.get(str(edge.get("source_id") or edge.get("parent_id") or ""))
        dst = node_index.get(str(edge.get("target_id") or edge.get("child_id") or ""))
        if src is None or dst is None:
            continue
        if str(edge.get("kind", "")).lower() == CONTAINMENT_EDGE_KIND:
            tree_pairs.append((src, dst))
        elif include_semantic_edges:
            semantic_pairs.append((src, dst))

    def _edge_index(pairs: Sequence[tuple[int, int]]) -> np.ndarray:
        if not pairs:
            return np.zeros((2, 0), dtype=np.int64)
        return np.asarray(pairs, dtype=np.int64).T

    return {
        "node_type_ids": node_type_ids,
        "span_coords": span_coords,
        "strand": strand,
        "frame": frame,
        "tree_edge_index": _edge_index(tree_pairs),
        "node_mask": np.ones((len(nodes),), dtype=bool),
        **({"semantic_edge_index": _edge_index(semantic_pairs)} if include_semantic_edges else {}),
    }


def collate_bio_ast_batches(items: Sequence[Optional[BioASTBatch]]) -> Optional[Dict[str, "torch.Tensor"]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for collate_bio_ast_batches.")
    present = [item for item in items if item is not None]
    if not present:
        return None
    max_nodes = max(int(item.node_type_ids.shape[0]) for item in present)
    batch_size = len(items)

    node_type_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    span_coords = torch.full((batch_size, max_nodes, 2), -1.0, dtype=torch.float32)
    strand = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    frame = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    tree_edges = []
    semantic_edges = []
    has_semantic = any(item is not None and item.semantic_edge_index is not None for item in items)

    for batch_idx, item in enumerate(items):
        if item is None:
            continue
        count = int(item.node_type_ids.shape[0])
        node_type_ids[batch_idx, :count] = item.node_type_ids
        span_coords[batch_idx, :count] = item.span_coords
        strand[batch_idx, :count] = item.strand
        frame[batch_idx, :count] = item.frame
        node_mask[batch_idx, :count] = True if item.node_mask is None else item.node_mask.bool()
        tree_edges.append(item.tree_edge_index)
        semantic_edges.append(item.semantic_edge_index if item.semantic_edge_index is not None else torch.zeros((2, 0), dtype=torch.long))

    out = {
        "node_type_ids": node_type_ids,
        "coords": span_coords,
        "span_coords": span_coords,
        "strand": strand,
        "frame": frame,
        "node_mask": node_mask,
        "edge_index": tree_edges,
        "tree_edge_index": tree_edges,
    }
    if has_semantic:
        out["semantic_edge_index"] = semantic_edges
    return out
