from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from perceptrome.bio_ast import ASTNode, BioAST


STORAGE_MAP_SCHEMA = "bio_ast_storage_map_v1"
_TRACKABLE_NODE_TYPES = (
    "gene",
    "orf",
    "cds",
    "region",
    "domain",
    "sme",
    "microfeature",
    "residue",
    "kmer",
)
_NODE_TYPE_SORT_ORDER = {node_type: index for index, node_type in enumerate(_TRACKABLE_NODE_TYPES)}
_STRAND_SORT_ORDER = {"+": 0, "-": 1, None: 2, "": 2}


@dataclass(frozen=True)
class _SegmentSeed:
    canonical_id: str
    parent_id: Optional[str]
    node_type: str
    start: int
    end: int
    track_key: Tuple[int, str, str]


def _normalize_strand(value: Optional[str]) -> str:
    if value == "+":
        return "+"
    if value == "-":
        return "-"
    return "unstranded"


def _track_key(node: ASTNode) -> Tuple[int, str, str]:
    strand = node.strand if node.strand in {"+", "-"} else None
    return (
        _STRAND_SORT_ORDER.get(strand, 2),
        _normalize_strand(strand),
        f"{_NODE_TYPE_SORT_ORDER.get(node.node_type, len(_NODE_TYPE_SORT_ORDER)):04d}:{node.node_type}",
    )


def build_storage_map_payload(ast: BioAST, sequence_length: int, *, accession: Optional[str] = None) -> Dict[str, Any]:
    roots = [node for node in ast.nodes if not node.parent_id]
    root = sorted(roots, key=lambda node: (node.start is None, node.start or 0, node.canonical_id))[0] if roots else None

    seeds: List[_SegmentSeed] = []
    for node in ast.nodes:
        if node.node_type not in _TRACKABLE_NODE_TYPES:
            continue
        if node.start is None or node.end is None:
            continue
        seeds.append(
            _SegmentSeed(
                canonical_id=node.canonical_id,
                parent_id=node.parent_id,
                node_type=node.node_type,
                start=int(node.start),
                end=int(node.end),
                track_key=_track_key(node),
            )
        )

    ordered_track_keys = sorted({seed.track_key for seed in seeds})
    track_index_by_key = {track_key: index for index, track_key in enumerate(ordered_track_keys)}
    grouped_seeds: Dict[Tuple[int, str, str], List[_SegmentSeed]] = {track_key: [] for track_key in ordered_track_keys}
    for seed in seeds:
        grouped_seeds[seed.track_key].append(seed)

    coordinate_segments: List[Dict[str, Any]] = []
    tracks: List[Dict[str, Any]] = []
    for track_key in ordered_track_keys:
        _, strand_label, node_type_key = track_key
        node_type = node_type_key.split(":", 1)[1]
        track_segments = sorted(grouped_seeds[track_key], key=lambda seed: (seed.start, seed.end, seed.canonical_id))
        lane_ends: List[int] = []
        segment_offset = len(coordinate_segments)
        track_id = f"track:{track_index_by_key[track_key]}"

        for seed in track_segments:
            for lane_index, lane_end in enumerate(lane_ends):
                if lane_end < seed.start:
                    lane_ends[lane_index] = seed.end
                    break
            else:
                lane_index = len(lane_ends)
                lane_ends.append(seed.end)

            coordinate_segments.append(
                {
                    "segment_id": seed.canonical_id,
                    "node_id": seed.canonical_id,
                    "parent_id": seed.parent_id,
                    "node_type": seed.node_type,
                    "strand": strand_label,
                    "track_id": track_id,
                    "track_index": track_index_by_key[track_key],
                    "lane_index": lane_index,
                    "start": seed.start,
                    "end": seed.end,
                    "length": seed.end - seed.start + 1,
                }
            )

        tracks.append(
            {
                "track_id": track_id,
                "track_index": track_index_by_key[track_key],
                "strand": strand_label,
                "node_type": node_type,
                "lane_count": len(lane_ends),
                "segment_count": len(track_segments),
                "segment_range": [segment_offset, len(coordinate_segments)],
            }
        )

    return {
        "schema": STORAGE_MAP_SCHEMA,
        "accession": accession,
        "sequence_length": int(sequence_length),
        "topology": {
            "root_id": root.canonical_id if root else None,
            "root_type": root.node_type if root else None,
            "is_circular": bool(root and root.node_type == "plasmid"),
        },
        "tracks": tracks,
        "coordinate_segments": coordinate_segments,
    }
