# Perceptrome

Perceptrome is a streaming VAE playground for genome/proteome experiments:
fetch → encode → train → scope (visualize) → generate.

## Quick start

```bash
# from repo root
python3 stream_train.py --help

# initialize folders + state
python3 stream_train.py init

# fetch and encode a plasmid
python3 stream_train.py fetch-one L09137.2
python3 stream_train.py encode-one L09137.2 --window-size 512 --stride 256

# train on one accession
python3 stream_train.py train-one L09137.2 --steps 50 --batch-size 32
```

## CLI overview

Run `python3 stream_train.py --help` for the full list of subcommands. Common workflows:

- `init`: create cache/log/state directories
- `fetch-one`: download an accession (FASTA or GenBank)
- `encode-one`: encode sequence windows (base/codon/aa)
- `train-one`: train on one accession
- `scope-one`: view reconstruction errors in a curses UI
- `validate`: export reconstruction metrics to CSV/JSON
- `stream`: iterate through a catalog file
- `generate-plasmid` / `generate-protein`: sample sequences from the model

For the configuration schema and CLI flag meanings, see [`docs/CONFIG.md`](docs/CONFIG.md).
