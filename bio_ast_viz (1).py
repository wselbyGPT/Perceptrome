from __future__ import annotations

import argparse
import curses
import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BASES = "ACGT"
STOP_CODONS = {"TAA", "TAG", "TGA"}
PULSE_CHARS = ".oO*"


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(table)[::-1]


def gc_ratio(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for ch in seq if ch in "GC")
    return gc / len(seq)


def random_dna(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(BASES) for _ in range(max(0, n)))


def load_fasta_sequence(path: str) -> Tuple[str, List[str]]:
    fasta_path = Path(path).expanduser()
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if not fasta_path.is_file():
        raise ValueError(f"Not a file: {fasta_path}")

    headers: List[str] = []
    chunks: List[str] = []
    current: List[str] = []

    with fasta_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    chunks.append("".join(current))
                    current = []
                headers.append(line[1:].strip() or f"record_{len(headers) + 1}")
                continue
            cleaned = re.sub(r"[^ACGTUNacgtun]", "", line).upper().replace("U", "T")
            if cleaned:
                current.append(cleaned)
        if current:
            chunks.append("".join(current))

    if not chunks:
        raise ValueError("No FASTA sequence data found.")

    sequence = "".join(chunks)
    return sequence, headers


def make_repeat_block(rng: random.Random) -> str:
    motif_len = rng.choice([2, 3, 4])
    motif = random_dna(rng, motif_len)
    reps = rng.randint(2, 4)
    return motif * reps


def make_gc_cluster(rng: random.Random) -> str:
    length = rng.randint(6, 12)
    return "".join(rng.choice("GCGCGCTC") for _ in range(length))


def make_palindrome(rng: random.Random) -> str:
    half_len = rng.choice([3, 4])
    half = random_dna(rng, half_len)
    return half + reverse_complement(half)


def make_coding_gene(rng: random.Random) -> str:
    codon_pool = [
        "GCC", "GGC", "CGT", "AAC", "CAG", "GAA", "TTC", "CTG", "GTA",
        "CAA", "AAA", "ACC", "GCG", "TGG", "ATC", "CCG", "GCT", "GAT",
    ]
    n_codons = rng.randint(5, 11)
    body: List[str] = []
    for _ in range(n_codons):
        codon = rng.choice(codon_pool)
        body.append(codon)
        if rng.random() < 0.18:
            body.append(rng.choice(["GCG", "CGC", "CCG"]))
    stop = rng.choice(sorted(STOP_CODONS))
    spacer = random_dna(rng, rng.randint(3, 7))
    tail = random_dna(rng, rng.randint(4, 12))
    return "TATAAT" + spacer + "ATG" + "".join(body) + stop + tail


def generate_structured_sequence(rng: random.Random, target_len: int = 1700) -> str:
    parts: List[str] = []
    while sum(len(p) for p in parts) < target_len:
        roll = rng.random()
        if roll < 0.22:
            parts.append(random_dna(rng, rng.randint(18, 45)))
        elif roll < 0.54:
            parts.append(make_coding_gene(rng))
        elif roll < 0.70:
            parts.append(make_repeat_block(rng))
        elif roll < 0.85:
            parts.append(make_gc_cluster(rng))
        else:
            parts.append(make_palindrome(rng))
            parts.append(random_dna(rng, rng.randint(6, 18)))
    seq = "".join(parts)
    if len(seq) < target_len:
        seq += random_dna(rng, target_len - len(seq))
    return seq[:target_len]


@dataclass
class ASTNode:
    node_id: int
    kind: str
    label: str
    start: int
    end: int
    score: float = 1.0
    children: List["ASTNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.node_id,
            "type": self.kind,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class TokenEvent:
    kind: str
    label: str
    start: int
    end: int
    detail: str
    created_at: float


@dataclass
class RenderLine:
    text: str
    node_id: int
    pulse_cols: List[int]


class BioASTVisualizer:
    def __init__(self, stdscr: "curses._CursesWindow", initial_fasta: Optional[str] = None) -> None:
        self.stdscr = stdscr
        self.rng = random.Random()
        self._terminal_has_color = curses.has_colors()
        self.color_enabled = self._terminal_has_color
        self.compact_tokens = False
        self.highlight_motifs = True
        self.ast_mode = "tree"
        self.ast_follow = True
        self.step_delay = 0.05
        self.paused = False
        self.status_message = ""
        self.last_export_path = ""
        self.flash_until = 0.0
        self.hot_spans: List[Tuple[int, int, float, str]] = []
        self.last_frame_time = time.monotonic()
        self.current_source_mode = "generated"
        self.current_source_label = "generated"
        self.current_source_path: Optional[str] = None
        self.fasta_headers: List[str] = []
        self.collapsed_nodes: set[int] = set()
        self.selected_node_id: Optional[int] = None
        self.motif_query = ""
        self.motif_hits: List[int] = []
        self.motif_hit_index = 0
        self._cached_tree_lines: List[RenderLine] = []

        self._init_curses()
        self._init_colors()
        self.reset(generate_new=True)
        if initial_fasta:
            self.load_fasta(initial_fasta)

    def _init_curses(self) -> None:
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        self.stdscr.nodelay(True)
        self.stdscr.keypad(True)

    def _init_colors(self) -> None:
        if not self._terminal_has_color:
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)     # border / title
        curses.init_pair(2, curses.COLOR_GREEN, -1)    # A
        curses.init_pair(3, curses.COLOR_BLUE, -1)     # C
        curses.init_pair(4, curses.COLOR_YELLOW, -1)   # G
        curses.init_pair(5, curses.COLOR_RED, -1)      # T
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # active / cursor
        curses.init_pair(7, curses.COLOR_WHITE, -1)    # neutral text
        curses.init_pair(8, curses.COLOR_GREEN, -1)    # promoter / gene
        curses.init_pair(9, curses.COLOR_YELLOW, -1)   # repeat / gc
        curses.init_pair(10, curses.COLOR_MAGENTA, -1) # palindrome / export

    def cp(self, n: int) -> int:
        if not self.color_enabled or not self._terminal_has_color:
            return 0
        return curses.color_pair(n)

    def reset(self, generate_new: bool = False) -> None:
        if generate_new or self.current_source_mode != "fasta":
            self.sequence = generate_structured_sequence(self.rng)
            self.current_source_mode = "generated"
            self.current_source_label = "structured generator"
            self.current_source_path = None
            self.fasta_headers = []
        else:
            self.load_fasta(self.current_source_path or "")
            return
        self._reset_runtime_state()
        self.status_message = "Structured DNA stream ready. Press l to load FASTA or / to search motifs."

    def _reset_runtime_state(self) -> None:
        self.cursor = 0
        self.window_size = 8
        self.token_log: List[TokenEvent] = []
        self.node_counter = 0
        self.node_by_id: Dict[int, ASTNode] = {}
        self.parent_by_id: Dict[int, Optional[int]] = {}
        self.root = self._new_node("Genome", "Genome", 0, max(0, len(self.sequence) - 1))
        self.current_region: Optional[ASTNode] = None
        self.current_gene: Optional[ASTNode] = None
        self.current_gene_start: Optional[int] = None
        self.active_node_id = self.root.node_id
        self.selected_node_id = self.root.node_id
        self.active_token: Optional[TokenEvent] = None
        self.kind_counts: Dict[str, int] = {
            "Promoter": 0,
            "Gene": 0,
            "StartCodon": 0,
            "Codon": 0,
            "StopCodon": 0,
            "Repeat": 0,
            "GCCluster": 0,
            "Palindrome": 0,
        }
        self.skip_until: Dict[str, int] = {
            "Promoter": 0,
            "Repeat": 0,
            "GCCluster": 0,
            "Palindrome": 0,
        }
        self.flash_until = 0.0
        self.hot_spans.clear()
        self.collapsed_nodes = set()
        self.last_export_path = ""
        self._cached_tree_lines = []
        self._refresh_motif_hits(recenter=False)

    def _new_node(self, kind: str, label: str, start: int, end: int, score: float = 1.0) -> ASTNode:
        self.node_counter += 1
        node = ASTNode(
            node_id=self.node_counter,
            kind=kind,
            label=label,
            start=start,
            end=end,
            score=score,
        )
        self.node_by_id[node.node_id] = node
        self.parent_by_id[node.node_id] = None
        return node

    def _touch_region(self, start: int, end: int) -> ASTNode:
        if self.current_region is None or start - self.current_region.end > 28:
            label = f"Region {len(self.root.children) + 1}"
            self.current_region = self._new_node("Region", label, start, max(end, start))
            self._append_child(self.root, self.current_region)
        self.current_region.end = max(self.current_region.end, end)
        return self.current_region

    def _append_child(self, parent: ASTNode, child: ASTNode) -> None:
        parent.children.append(child)
        parent.end = max(parent.end, child.end)
        self.parent_by_id[child.node_id] = parent.node_id
        if self.current_region is not None:
            self.current_region.end = max(self.current_region.end, child.end)
        self.active_node_id = child.node_id
        if self.ast_follow and self.selected_node_id in {None, parent.node_id, self.active_node_id}:
            self.selected_node_id = child.node_id

    def _note_hot_span(self, start: int, end: int, kind: str) -> None:
        now = time.monotonic()
        self.hot_spans.append((start, end, now + 1.0, kind))
        self.flash_until = now + 0.25

    def _push_token(self, kind: str, label: str, start: int, end: int, detail: str) -> None:
        event = TokenEvent(kind=kind, label=label, start=start, end=end, detail=detail, created_at=time.monotonic())
        self.token_log.append(event)
        self.token_log = self.token_log[-500:]
        self.active_token = event
        if kind in self.kind_counts:
            self.kind_counts[kind] += 1
        self._note_hot_span(start, end, kind)

    def _create_promoter(self, i: int) -> None:
        region = self._touch_region(i, i + 5)
        node = self._new_node("Promoter", "TATAAT", i, i + 5, score=1.0)
        self._append_child(region, node)
        self._push_token("Promoter", "TATAAT", i, i + 5, "canonical promoter")
        self.skip_until["Promoter"] = i + 6
        self.cursor += 6

    def _open_gene(self, i: int) -> None:
        region = self._touch_region(i, i + 2)
        gene = self._new_node("Gene", f"Gene@{i}", i, i + 2, score=1.0)
        self._append_child(region, gene)
        start_node = self._new_node("StartCodon", "ATG", i, i + 2, score=1.0)
        self._append_child(gene, start_node)
        self.current_gene = gene
        self.current_gene_start = i
        self._push_token("Gene", gene.label, i, i + 2, "gene opened")
        self._push_token("StartCodon", "ATG", i, i + 2, "start codon")
        self.cursor += 3

    def _append_gene_codon(self, i: int, codon: str) -> None:
        assert self.current_gene is not None
        if codon in STOP_CODONS:
            node = self._new_node("StopCodon", codon, i, i + 2, score=1.0)
            self._append_child(self.current_gene, node)
            self.current_gene.end = i + 2
            self._push_token("StopCodon", codon, i, i + 2, "translation stop")
            self.current_gene = None
            self.current_gene_start = None
        else:
            node = self._new_node("Codon", codon, i, i + 2, score=0.7)
            self._append_child(self.current_gene, node)
            detail = "GC-rich codon" if gc_ratio(codon) >= 0.67 else "coding triplet"
            self._push_token("Codon", codon, i, i + 2, detail)
            if self.highlight_motifs and gc_ratio(codon) == 1.0:
                gc_node = self._new_node("GCCluster", codon, i, i + 2, score=0.85)
                self._append_child(self.current_gene, gc_node)
                self._push_token("GCCluster", codon, i, i + 2, "all-G/C codon")
        self.cursor += 3

    def _match_repeat(self, i: int) -> Optional[str]:
        if i < self.skip_until["Repeat"]:
            return None
        for motif_len in (4, 3, 2):
            motif = self.sequence[i:i + motif_len]
            if len(motif) < motif_len or "N" in motif:
                continue
            reps = 1
            while self.sequence[i + reps * motif_len:i + (reps + 1) * motif_len] == motif:
                reps += 1
            if reps >= 2 and motif not in {"AAAA", "TTTT", "CCCC", "GGGG"}:
                return motif * reps
        return None

    def _match_palindrome(self, i: int) -> Optional[str]:
        if i < self.skip_until["Palindrome"]:
            return None
        for length in (8, 6):
            chunk = self.sequence[i:i + length]
            if len(chunk) == length and "N" not in chunk and chunk == reverse_complement(chunk):
                return chunk
        return None

    def _match_gc_cluster(self, i: int) -> Optional[str]:
        if i < self.skip_until["GCCluster"]:
            return None
        for length in (10, 8, 6):
            chunk = self.sequence[i:i + length]
            if len(chunk) == length and "N" not in chunk and gc_ratio(chunk) >= 0.83:
                return chunk
        return None

    def _create_misc_feature(self, kind: str, label: str, i: int, length: int, detail: str) -> None:
        region = self._touch_region(i, i + length - 1)
        node = self._new_node(kind, label, i, i + length - 1, score=0.8)
        self._append_child(region, node)
        self._push_token(kind, label, i, i + length - 1, detail)
        self.skip_until[kind] = i + length
        self.cursor += length

    def scan_step(self) -> None:
        if self.cursor >= len(self.sequence) - 2:
            self.paused = True
            self.status_message = "End of sequence reached. Press r to restart or l to load FASTA."
            return

        i = self.cursor
        if self.current_gene is None and self.sequence[i:i + 6] == "TATAAT" and i >= self.skip_until["Promoter"]:
            self._create_promoter(i)
            return

        if self.current_gene is None and self.sequence[i:i + 3] == "ATG":
            self._open_gene(i)
            return

        if self.current_gene is not None and self.sequence[i:i + 3]:
            offset = i - (self.current_gene_start or i)
            if offset >= 3 and offset % 3 == 0:
                codon = self.sequence[i:i + 3]
                if len(codon) == 3 and "N" not in codon:
                    self._append_gene_codon(i, codon)
                    return

        if self.highlight_motifs:
            repeat = self._match_repeat(i)
            if repeat:
                self._create_misc_feature("Repeat", repeat, i, len(repeat), "tandem repeat")
                return

            pal = self._match_palindrome(i)
            if pal:
                self._create_misc_feature("Palindrome", pal, i, len(pal), "reverse-complement palindrome")
                return

            gc = self._match_gc_cluster(i)
            if gc:
                self._create_misc_feature("GCCluster", gc, i, len(gc), f"GC={gc_ratio(gc):.0%}")
                return

        self.cursor += 1

    def export_json(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"bio_ast_export_{stamp}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.root.to_dict(), fh, indent=2)
        self.last_export_path = path
        self.status_message = f"Exported AST to {path}"

    def load_fasta(self, path: str) -> None:
        sequence, headers = load_fasta_sequence(path)
        self.sequence = sequence
        self.current_source_mode = "fasta"
        self.current_source_path = str(Path(path).expanduser())
        self.current_source_label = Path(path).name
        self.fasta_headers = headers
        self._reset_runtime_state()
        self.paused = False
        self.status_message = f"Loaded FASTA: {self.current_source_label} ({len(sequence)} bases, {len(headers)} record(s))"

    def _sanitize_motif(self, text: str) -> str:
        return re.sub(r"[^ACGTN]", "", text.upper())

    def _refresh_motif_hits(self, recenter: bool = False) -> None:
        if not self.motif_query:
            self.motif_hits = []
            self.motif_hit_index = 0
            return
        motif = self.motif_query
        self.motif_hits = [m.start() for m in re.finditer(f"(?={re.escape(motif)})", self.sequence)]
        if not self.motif_hits:
            self.motif_hit_index = 0
            self.status_message = f"No motif hits for {motif}."
            return
        self.motif_hit_index = min(self.motif_hit_index, len(self.motif_hits) - 1)
        if recenter:
            self.cursor = self.motif_hits[self.motif_hit_index]
        self.status_message = f"Motif {motif}: {len(self.motif_hits)} hit(s). Use [ and ] to navigate."

    def _goto_motif_hit(self, delta: int) -> None:
        if not self.motif_hits:
            self.status_message = "No motif hits loaded. Press / to search."
            return
        self.motif_hit_index = (self.motif_hit_index + delta) % len(self.motif_hits)
        self.cursor = self.motif_hits[self.motif_hit_index]
        hit = self.motif_hits[self.motif_hit_index]
        self.status_message = f"Motif hit {self.motif_hit_index + 1}/{len(self.motif_hits)} at {hit}."

    def _prompt_text(self, prompt: str) -> Optional[str]:
        h, w = self.stdscr.getmaxyx()
        if h < 2 or w < len(prompt) + 3:
            return None
        self.stdscr.nodelay(False)
        curses.echo()
        curses.curs_set(1)
        try:
            self.stdscr.move(h - 1, 0)
            self.stdscr.clrtoeol()
            self._safe_add(self.stdscr, h - 1, 0, prompt, self.cp(6) | curses.A_BOLD)
            self.stdscr.refresh()
            raw = self.stdscr.getstr(h - 1, min(len(prompt), w - 1), max(1, w - len(prompt) - 1))
            return raw.decode("utf-8", "ignore").strip()
        except KeyboardInterrupt:
            return None
        finally:
            curses.noecho()
            curses.curs_set(0)
            self.stdscr.nodelay(True)

    def _path_to_root(self, node_id: Optional[int]) -> List[int]:
        path: List[int] = []
        while node_id is not None:
            path.append(node_id)
            node_id = self.parent_by_id.get(node_id)
        return list(reversed(path))

    def _active_path_ids(self) -> set[int]:
        return set(self._path_to_root(self.active_node_id))

    def _node_text(self, node: ASTNode) -> str:
        has_children = bool(node.children)
        collapsed = node.node_id in self.collapsed_nodes
        marker = "[+]" if has_children and collapsed else "[-]" if has_children else "   "
        if self.ast_mode == "compact":
            return f"{marker} {node.kind}: {node.label} [{node.start}:{node.end}]"
        return f"{marker} [{node.kind}] {node.label} [{node.start}:{node.end}]"

    def _tree_lines(self, node: ASTNode) -> Tuple[List[RenderLine], int]:
        lines: List[RenderLine] = []
        active_path = self._active_path_ids()

        def walk(cur: ASTNode, ancestor_continues: List[bool], ancestor_active: List[bool], is_last: bool, is_root: bool = False) -> None:
            pulse_cols: List[int] = []
            prefix_parts: List[str] = []
            for idx, cont in enumerate(ancestor_continues):
                col = len("".join(prefix_parts))
                prefix_parts.append("|   " if cont else "    ")
                if cont and ancestor_active[idx]:
                    pulse_cols.append(col)
            prefix = "".join(prefix_parts)
            if is_root:
                text = self._node_text(cur)
            else:
                connector_col = len(prefix)
                connector = "`-- " if is_last else "|-- "
                prefix += connector
                if self.parent_by_id.get(cur.node_id) in active_path or cur.node_id in active_path:
                    pulse_cols.append(connector_col)
                text = prefix + self._node_text(cur)
            lines.append(RenderLine(text=text, node_id=cur.node_id, pulse_cols=pulse_cols))

            if cur.node_id in self.collapsed_nodes:
                return

            total = len(cur.children)
            for idx, child in enumerate(cur.children):
                child_last = idx == total - 1
                walk(
                    child,
                    ancestor_continues + ([not is_last] if not is_root else []),
                    ancestor_active + ([cur.node_id in active_path] if not is_root else []),
                    child_last,
                    False,
                )

        walk(node, [], [], True, True)
        selected_idx = next((i for i, line in enumerate(lines) if line.node_id == self.selected_node_id), 0)
        self._cached_tree_lines = lines
        return lines, selected_idx

    def _selected_node(self) -> Optional[ASTNode]:
        if self.selected_node_id is None:
            return None
        return self.node_by_id.get(self.selected_node_id)

    def _move_selection(self, delta: int) -> None:
        lines, idx = self._tree_lines(self.root)
        if not lines:
            return
        idx = max(0, min(len(lines) - 1, idx + delta))
        self.selected_node_id = lines[idx].node_id
        node = self.node_by_id[self.selected_node_id]
        self.status_message = f"Selected {node.kind} {node.label} [{node.start}:{node.end}]"

    def _collapse_selected(self) -> None:
        node = self._selected_node()
        if node is None or not node.children:
            return
        self.collapsed_nodes.add(node.node_id)
        self.status_message = f"Collapsed {node.kind} {node.label}."

    def _expand_selected(self) -> None:
        node = self._selected_node()
        if node is None:
            return
        if node.node_id in self.collapsed_nodes:
            self.collapsed_nodes.discard(node.node_id)
            self.status_message = f"Expanded {node.kind} {node.label}."
            return
        if node.children:
            self.selected_node_id = node.children[0].node_id
            child = node.children[0]
            self.status_message = f"Selected child {child.kind} {child.label}."

    def _toggle_selected_collapse(self) -> None:
        node = self._selected_node()
        if node is None or not node.children:
            return
        if node.node_id in self.collapsed_nodes:
            self.collapsed_nodes.discard(node.node_id)
            self.status_message = f"Expanded {node.kind} {node.label}."
        else:
            self.collapsed_nodes.add(node.node_id)
            self.status_message = f"Collapsed {node.kind} {node.label}."

    def _select_parent_or_collapse(self) -> None:
        node = self._selected_node()
        if node is None:
            return
        if node.children and node.node_id not in self.collapsed_nodes:
            self.collapsed_nodes.add(node.node_id)
            self.status_message = f"Collapsed {node.kind} {node.label}."
            return
        parent_id = self.parent_by_id.get(node.node_id)
        if parent_id is not None:
            self.selected_node_id = parent_id
            parent = self.node_by_id[parent_id]
            self.status_message = f"Selected parent {parent.kind} {parent.label}."

    def _current_search_snippets(self, limit: int = 8) -> List[str]:
        if not self.motif_hits:
            return []
        motif_len = len(self.motif_query)
        around = self.motif_hit_index
        start_idx = max(0, around - limit // 2)
        visible_hits = self.motif_hits[start_idx:start_idx + limit]
        rows: List[str] = []
        for pos in visible_hits:
            left = max(0, pos - 5)
            right = min(len(self.sequence), pos + motif_len + 5)
            snippet = self.sequence[left:right]
            marker = ">" if pos == self.motif_hits[self.motif_hit_index] else " "
            rows.append(f"{marker} {pos:6d}  {snippet}")
        return rows

    def handle_input(self) -> bool:
        while True:
            ch = self.stdscr.getch()
            if ch == -1:
                return True
            if ch in (ord("q"), ord("Q")):
                return False
            if ch == ord(" "):
                self.paused = not self.paused
                self.status_message = "Paused." if self.paused else "Running."
            elif ch in (ord("j"), ord("J")):
                self.step_delay = min(0.30, self.step_delay + 0.01)
                self.status_message = f"Speed adjusted: {self.step_delay:.2f}s per step"
            elif ch in (ord("k"), ord("K")):
                self.step_delay = max(0.01, self.step_delay - 0.01)
                self.status_message = f"Speed adjusted: {self.step_delay:.2f}s per step"
            elif ch in (ord("t"), ord("T")):
                self.compact_tokens = not self.compact_tokens
                self.status_message = "Token pane: compact" if self.compact_tokens else "Token pane: verbose"
            elif ch in (ord("m"), ord("M")):
                self.highlight_motifs = not self.highlight_motifs
                self.status_message = "Motif highlighting ON" if self.highlight_motifs else "Motif highlighting OFF"
            elif ch in (ord("g"), ord("G")):
                self.ast_mode = "compact" if self.ast_mode == "tree" else "tree"
                self.status_message = f"AST mode: {self.ast_mode}"
            elif ch in (ord("a"), ord("A")):
                self.ast_follow = not self.ast_follow
                self.status_message = "AST auto-follow ON" if self.ast_follow else "AST auto-follow OFF"
            elif ch in (ord("e"), ord("E")):
                self.export_json()
            elif ch in (ord("r"), ord("R")):
                self.reset(generate_new=self.current_source_mode != "fasta")
            elif ch in (ord("c"), ord("C")):
                self.color_enabled = not self.color_enabled and self._terminal_has_color or (not self.color_enabled and self._terminal_has_color)
                self.status_message = "Color ON" if self.color_enabled else "Color OFF"
            elif ch in (ord("n"), ord("N")):
                if self.paused:
                    self.scan_step()
                    self.status_message = "Stepped one frame."
            elif ch in (ord("l"), ord("L")):
                path = self._prompt_text("Load FASTA path: ")
                if path:
                    try:
                        self.load_fasta(path)
                    except Exception as exc:
                        self.status_message = f"FASTA load failed: {exc}"
            elif ch == ord("/"):
                query = self._prompt_text("Search motif (ACGTN): ")
                if query is None:
                    continue
                motif = self._sanitize_motif(query)
                if not motif:
                    self.motif_query = ""
                    self.motif_hits = []
                    self.status_message = "Motif search cleared."
                else:
                    self.motif_query = motif
                    self.motif_hit_index = 0
                    self._refresh_motif_hits(recenter=True)
            elif ch == ord("x"):
                self.motif_query = ""
                self.motif_hits = []
                self.status_message = "Motif search cleared."
            elif ch == ord("]"):
                self._goto_motif_hit(+1)
            elif ch == ord("["):
                self._goto_motif_hit(-1)
            elif ch in (curses.KEY_UP, ord("w"), ord("W")):
                self._move_selection(-1)
            elif ch in (curses.KEY_DOWN, ord("s"), ord("S")):
                self._move_selection(+1)
            elif ch in (curses.KEY_RIGHT, ord("d"), ord("D")):
                self._expand_selected()
            elif ch in (curses.KEY_LEFT, ord("h"), ord("H")):
                self._select_parent_or_collapse()
            elif ch in (10, 13):
                self._toggle_selected_collapse()

    def tick(self) -> None:
        now = time.monotonic()
        self.hot_spans = [span for span in self.hot_spans if span[2] >= now]
        if not self.paused and now - self.last_frame_time >= self.step_delay:
            self.scan_step()
            self.last_frame_time = now

    def _base_attr(self, ch: str) -> int:
        if ch == "A":
            return self.cp(2)
        if ch == "C":
            return self.cp(3)
        if ch == "G":
            return self.cp(4)
        if ch == "T":
            return self.cp(5)
        return self.cp(7)

    def _token_attr(self, kind: str) -> int:
        if kind in {"Promoter", "Gene", "StartCodon", "StopCodon"}:
            return self.cp(8) | curses.A_BOLD
        if kind in {"Repeat", "GCCluster"}:
            return self.cp(9) | curses.A_BOLD
        if kind == "Palindrome":
            return self.cp(10) | curses.A_BOLD
        return self.cp(7)

    def _safe_add(self, win: "curses._CursesWindow", y: int, x: int, text: str, attr: int = 0) -> None:
        if not text:
            return
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        max_len = max(0, w - x - 1)
        if max_len <= 0:
            return
        try:
            win.addnstr(y, x, text, max_len, attr)
        except curses.error:
            pass

    def _draw_box(self, win: "curses._CursesWindow", title: str, flash: bool = False) -> None:
        win.erase()
        border_attr = self.cp(6) | curses.A_BOLD if flash else self.cp(1) | curses.A_BOLD
        try:
            win.attron(border_attr)
            win.box()
            win.attroff(border_attr)
        except curses.error:
            pass
        self._safe_add(win, 0, 2, f" {title} ", border_attr)

    def _motif_hit_span_attr(self, pos: int) -> int:
        if not self.motif_query or not self.motif_hits:
            return 0
        motif_len = len(self.motif_query)
        attr = 0
        for idx, hit in enumerate(self.motif_hits):
            if hit <= pos < hit + motif_len:
                attr |= curses.A_BOLD
                if idx == self.motif_hit_index:
                    attr |= curses.A_REVERSE
                break
        return attr

    def _render_raw_sequence(self, win: "curses._CursesWindow") -> None:
        source = self.current_source_label[:24]
        self._draw_box(win, f" Raw Sequence ({source}) ")
        h, w = win.getmaxyx()
        ih, iw = h - 2, w - 2
        if ih <= 0 or iw <= 0:
            return

        total_chars = ih * iw
        start = max(0, min(self.cursor - total_chars // 3, max(0, len(self.sequence) - total_chars)))
        highlight_range = range(self.cursor, min(len(self.sequence), self.cursor + self.window_size))

        for local_idx in range(total_chars):
            pos = start + local_idx
            if pos >= len(self.sequence):
                break
            y = 1 + local_idx // iw
            x = 1 + local_idx % iw
            ch = self.sequence[pos]
            attr = self._base_attr(ch)

            for span_start, span_end, _, _kind in self.hot_spans:
                if span_start <= pos <= span_end:
                    attr |= curses.A_BOLD
                    break

            attr |= self._motif_hit_span_attr(pos)

            if pos in highlight_range:
                attr |= curses.A_REVERSE | curses.A_BOLD

            self._safe_add(win, y, x, ch, attr)

        footer = f"idx={self.cursor}/{len(self.sequence)-1}  window={self.sequence[self.cursor:self.cursor+self.window_size]}"
        self._safe_add(win, h - 1, 2, footer[: max(0, w - 4)], self.cp(7))

    def _format_token(self, token: TokenEvent) -> str:
        if self.compact_tokens:
            return f"{token.start:05d} {token.kind[:3].upper():<3} {token.label}"
        return f"{token.start:05d}-{token.end:05d} {token.kind:<11} {token.label:<14} {token.detail}"

    def _render_tokens(self, win: "curses._CursesWindow") -> None:
        suffix = f"  /{self.motif_query}" if self.motif_query else ""
        self._draw_box(win, f" Tokens{suffix} ")
        h, _ = win.getmaxyx()
        rows = h - 2
        visible = self.token_log[-rows:] if rows > 0 else []
        start_row = max(1, h - 1 - len(visible))
        for idx, token in enumerate(visible):
            y = start_row + idx
            text = self._format_token(token)
            attr = self._token_attr(token.kind)
            if self.active_token and token is self.active_token:
                attr |= curses.A_REVERSE
            self._safe_add(win, y, 1, text, attr)

    def _render_ast(self, win: "curses._CursesWindow") -> None:
        title = " Bio-AST (enter collapse, arrows navigate) "
        self._draw_box(win, title)
        h, w = win.getmaxyx()
        rows = h - 2
        if rows <= 0:
            return

        lines, selected_idx = self._tree_lines(self.root)
        if not lines:
            return
        if self.ast_follow and self.selected_node_id is None:
            self.selected_node_id = self.active_node_id
        if self.selected_node_id is None:
            self.selected_node_id = lines[0].node_id
            selected_idx = 0

        start = 0
        if self.ast_follow and selected_idx >= rows:
            start = max(0, selected_idx - rows // 2)
        visible = lines[start:start + rows]
        phase = int(time.monotonic() * 10)

        for idx, line in enumerate(visible):
            attr = self.cp(7)
            if line.node_id == self.selected_node_id:
                attr = self.cp(6) | curses.A_BOLD | curses.A_REVERSE
            elif line.node_id == self.active_node_id:
                attr = self.cp(6) | curses.A_BOLD

            chars = list(line.text)
            for pulse_idx, col in enumerate(line.pulse_cols):
                if 0 <= col < len(chars) and chars[col] != " ":
                    chars[col] = PULSE_CHARS[(phase + pulse_idx + idx) % len(PULSE_CHARS)]
            display = "".join(chars)
            self._safe_add(win, 1 + idx, 1, display[: max(0, w - 3)], attr)

    def _render_annotations(self, win: "curses._CursesWindow") -> None:
        flash = time.monotonic() < self.flash_until
        self._draw_box(win, " Annotations / State ", flash=flash)
        h, w = win.getmaxyx()
        lines: List[Tuple[str, int]] = []

        local = self.sequence[self.cursor:self.cursor + 24]
        local_gc = gc_ratio(local)
        selected = self._selected_node()
        selected_label = "none" if selected is None else f"{selected.kind} {selected.label} [{selected.start}:{selected.end}]"

        lines.append((f"Source        : {self.current_source_label}", self.cp(7)))
        lines.append((f"Cursor        : {self.cursor}", self.cp(7)))
        lines.append((f"Step delay    : {self.step_delay:.2f}s", self.cp(7)))
        lines.append((f"Parser state  : {'IN_GENE' if self.current_gene else 'SCAN'}", self.cp(8) | curses.A_BOLD))
        lines.append((f"Local window  : {local}", self.cp(7)))
        lines.append((f"Local GC      : {local_gc:.0%}", self.cp(9) | curses.A_BOLD))
        lines.append((f"Node count    : {self.node_counter}", self.cp(7)))
        token_label = "none" if not self.active_token else f"{self.active_token.kind} {self.active_token.label}"
        lines.append((f"Active token  : {token_label}", self.cp(10) | curses.A_BOLD))
        lines.append((f"Selected node : {selected_label}", self.cp(7)))
        counts = "  ".join(f"{k[:4]}={v}" for k, v in self.kind_counts.items())
        lines.append((counts, self.cp(7)))

        if self.motif_query:
            lines.append((f"Motif search  : {self.motif_query}", self.cp(1) | curses.A_BOLD))
            if self.motif_hits:
                cur_hit = self.motif_hits[self.motif_hit_index]
                lines.append((f"Hit position  : {cur_hit} ({self.motif_hit_index + 1}/{len(self.motif_hits)})", self.cp(9) | curses.A_BOLD))
                for row in self._current_search_snippets(limit=max(2, min(8, h // 5))):
                    lines.append((row, self.cp(7)))
            else:
                lines.append(("Hit position  : none", self.cp(7)))
        else:
            lines.append(("Motif search  : off", self.cp(7)))

        if self.fasta_headers:
            lines.append((f"FASTA records : {len(self.fasta_headers)}", self.cp(7)))
            head = self.fasta_headers[0][: max(8, w - 18)]
            lines.append((f"First header  : {head}", self.cp(7)))
        if self.last_export_path:
            lines.append((f"Last export   : {self.last_export_path}", self.cp(10)))
        lines.append(("", 0))
        lines.append(("Controls:", self.cp(1) | curses.A_BOLD))
        lines.append(("space pause  n step  r reset  l load FASTA  e export", self.cp(7)))
        lines.append(("arrows/WASD AST nav  enter toggle  / search  [ ] hits  x clear", self.cp(7)))
        lines.append(("j/k speed  t tokens  m motifs  g ast  a follow  c colors  q quit", self.cp(7)))
        lines.append(("", 0))
        lines.append((self.status_message, self.cp(6) | curses.A_BOLD))

        max_lines = h - 2
        for idx, (text, attr) in enumerate(lines[:max_lines]):
            self._safe_add(win, 1 + idx, 1, text, attr)

        bar_width = max(10, w - 4)
        progress = min(1.0, self.cursor / max(1, len(self.sequence) - 1))
        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * max(0, bar_width - filled)
        self._safe_add(win, h - 1, 2, bar[: bar_width], self.cp(8) | curses.A_BOLD)

    def render(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        if h < 24 or w < 100:
            msg1 = "Resize terminal to at least 100x24 for the 4-pane Bio-AST view."
            msg2 = f"Current size: {w}x{h}"
            msg3 = "Controls still work after resize; press l to load FASTA once it fits."
            self._safe_add(self.stdscr, max(0, h // 2 - 1), max(0, (w - len(msg1)) // 2), msg1, self.cp(6) | curses.A_BOLD)
            self._safe_add(self.stdscr, max(0, h // 2), max(0, (w - len(msg2)) // 2), msg2, self.cp(7))
            self._safe_add(self.stdscr, max(0, h // 2 + 1), max(0, (w - len(msg3)) // 2), msg3, self.cp(7))
            self.stdscr.refresh()
            return

        mid_y = h // 2
        mid_x = w // 2

        raw_win = self.stdscr.derwin(mid_y, mid_x, 0, 0)
        tok_win = self.stdscr.derwin(mid_y, w - mid_x, 0, mid_x)
        ast_win = self.stdscr.derwin(h - mid_y, mid_x, mid_y, 0)
        ann_win = self.stdscr.derwin(h - mid_y, w - mid_x, mid_y, mid_x)

        self._render_raw_sequence(raw_win)
        self._render_tokens(tok_win)
        self._render_ast(ast_win)
        self._render_annotations(ann_win)
        self.stdscr.refresh()

    def run(self) -> None:
        while True:
            if not self.handle_input():
                break
            self.tick()
            self.render()
            time.sleep(0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Bio-AST curses visualizer for WSL/Ubuntu terminal.")
    parser.add_argument("fasta", nargs="?", default=None, help="Optional FASTA file to load at startup.")
    return parser.parse_args()


def main(stdscr: "curses._CursesWindow", initial_fasta: Optional[str]) -> None:
    app = BioASTVisualizer(stdscr, initial_fasta=initial_fasta)
    app.run()


if __name__ == "__main__":
    args = parse_args()
    curses.wrapper(lambda stdscr: main(stdscr, args.fasta))
