# Architecture

## Goal
A streaming pipeline that repeatedly:
1) fetches sequences (FASTA/GenBank),
2) encodes them into tokens/windows,
3) trains an online model,
4) visualizes training dynamics ("scope"),
5) validates modes (e.g., proteome mode).

## Key modules (current)
- `scripts/ncbi_fetch.py` — data acquisition + caching
- `scripts/encoding.py` — tokenization/encoding (e.g., codon mode)
- `scripts/training.py` — streaming trainer / checkpointing
- `scripts/scope.py` — live diagnostics visualizer
- `config/stream_config.yaml` — runtime parameters

## State
- `state/progress.json` tracks resume state and counters
- `model/checkpoints/` stores checkpoints

## Notes
This file is intentionally minimal for now; expand as components stabilize.
