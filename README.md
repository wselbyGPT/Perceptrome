# Perceptrome

Perceptrome is a Python toolkit for **streaming representation learning on genomic and proteomic sequences**.
It provides a CLI-first workflow for:

- building accession catalogs,
- fetching and encoding sequence records,
- training VAE-style models in streaming mode,
- inspecting reconstruction/error dynamics with scope tools,
- and generating candidate plasmid/protein sequences from trained models.

## Highlights

- **CLI for end-to-end workflows** (catalog, fetch, encode, train, stream, generate).
- **Multiple tokenization modes**: base, codon, and amino-acid/proteome (`aa`).
- **Config-driven runs** via YAML (`config/stream_config.yaml`) with CLI overrides.
- **NCBI-integrated data acquisition** with local caching for FASTA/GenBank/encoded artifacts.
- **Training + observability utilities** including TensorBoard launcher and scope visualizers.
- **Generation + validation commands** for plasmid and protein candidates.

## Repository layout

- `perceptrome/` – core package (CLI, encoding, model/training, generation, scope UI).
- `config/` – starter configs and curated accession/corpus files.
- `accessions/` – accession lists by biological category.
- `tests/` – unit/smoke tests for CLI and core utilities.
- `gui_qt/` and `gui.py` – Qt GUI entry points for running commands interactively.
- `raylib_visualizer/` – optional C/Raylib scope visualizer.

## Installation

### 1) Clone and enter the repo

```bash
git clone <your-fork-or-this-repo-url>
cd Perceptrome
```

### 2) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

> `setup.py` also defines a console entrypoint named `perceptrome`.

## Quick start

### 1) Initialize project directories/state

```bash
perceptrome init --config config/stream_config.yaml
```

### 2) Build or inspect a catalog

Inspect an existing catalog-like file:

```bash
perceptrome catalog-show config/plasmids_100.txt
```

Generate a new catalog from a schema:

```bash
perceptrome catalog-generate --schema path/to/catalog_schema.json --output state/catalog.txt
```

### 3) Train in streaming mode

```bash
perceptrome stream --catalog config/plasmids_100.txt
```

### 4) Generate candidates from a trained model

Generate plasmid sequence:

```bash
perceptrome generate-plasmid --length-bp 10000 --output generated/novel_plasmid.fasta
```

Generate protein sequence:

```bash
perceptrome generate-protein --length-aa 600 --output generated/novel_protein.faa
```

### 5) Validate a generated plasmid against references

```bash
perceptrome validate-plasmid \
  --generated-fasta generated/novel_plasmid.fasta \
  --catalog config/plasmids_100.txt \
  --top-n 5 \
  --output-json generated/validation.json
```

## CLI overview

Global help:

```bash
perceptrome --help
```

Primary commands:

- `init`
- `catalog-show`, `catalog-generate`
- `split-create`, `split-show`
- `fetch-one`
- `encode-one`
- `train-one`
- `stream`
- `scope-one`, `scope-stream`
- `tensorboard`
- `generate-plasmid`, `validate-plasmid`, `generate-protein`

Most commands accept `--config` to point at a YAML config file.

## Configuration

Default config: `config/stream_config.yaml`.

The config includes:

- `ncbi`: email/api key and retry/backoff settings,
- `training`: tokenizer/model/windowing/loss/logging parameters,
- `io`: cache/model/log/state paths.

You can override many `training` values at the command line (for example model type, tokenizer, window/stride, batch size, and training steps).

## Running tests

```bash
python -m unittest discover -s tests -v
```

## Optional tools

- **Qt GUI**
  - Run with: `python -m gui_qt`
- **Raylib scope visualizer**
  - See `raylib_visualizer/README.md` for build/run instructions.

## Notes

- Some commands contact NCBI; set a real email in config for polite API usage.
- First runs may be slower due to cache warm-up and encoding.
- The default config is intentionally lightweight for smoke-testing and iteration.

## License

No explicit license file is currently included in this repository. Add a `LICENSE` file if you plan to distribute the project.
