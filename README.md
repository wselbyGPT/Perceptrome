# Perceptrome

Perceptrome is a Python toolkit for **streaming representation learning on genomic and proteomic sequences**.
It provides a CLI-first workflow for:

- building accession catalogs,
- fetching and encoding sequence records,
- training VAE-style models in streaming mode,
- inspecting reconstruction/error dynamics with scope tools,
- and generating candidate plasmid/protein sequences from trained models.

The repository also includes **Perceptrome Web**, a browser-based React/FastAPI application for authenticated administration, dataset/runs views, and websocket-backed run monitoring. See [`perceptrome_web/README.md`](perceptrome_web/README.md) for setup details covering the client, API server, PostgreSQL, Alembic migrations, bootstrap admin flow, SPA/API wiring, and WebSocket expectations.

## Highlights

- **CLI for end-to-end workflows** (catalog, fetch, encode, train, stream, generate).
- **Multiple tokenization modes**: base, codon, and amino-acid/proteome (`aa`).
- **Config-driven runs** via YAML (`config/stream_config.yaml`) with CLI overrides.
- **NCBI-integrated data acquisition** with local caching for FASTA/GenBank/encoded artifacts.
- **Training + observability utilities** including TensorBoard launcher and scope visualizers.
- **Generation + validation commands** for plasmid and protein candidates.
- **Web application companion** for authenticated API access, admin flows, and live run telemetry.

## Repository layout

- `perceptrome/` – core package (CLI, encoding, model/training, generation, scope UI).
- `perceptrome_web/` – web client/server application; production is PostgreSQL-first and uses mandatory Alembic migrations. SQLite remains a convenience mode for tests or local-only workflows.
- `config/` – starter configs and curated accession/corpus files.
- `accessions/` – accession lists by biological category.
- `tests/` – unit/smoke tests for CLI and core utilities.
- `raylib_visualizer/` – optional C/Raylib scope visualizer.

## AWS EC2 deployment

For a ready-to-run EC2 + Nginx + Route53 deployment bundle, see [`infra/aws/README.md`](infra/aws/README.md).

## Installation

Perceptrome now uses a split bootstrap layout so each workflow installs only what it needs.

### Python install paths

| Workflow | Command | Includes |
| --- | --- | --- |
| Core / CLI | `python -m pip install -r requirements/core.txt` | `perceptrome` package plus lightweight CLI/runtime deps |
| Web server | `python -m pip install -r requirements/web.txt` | FastAPI, SQLAlchemy, Alembic, auth/session deps |
| Dev / test | `python -m pip install -r requirements/dev.txt` | Combined Python deps used across local development |
| Optional GPU | `python -m pip install -r requirements/gpu-cu12.txt` | Torch + CUDA 12 runtime packages for accelerated workloads |

`requirements.txt` is kept as a compatibility shim and currently points at the CUDA-oriented GPU stack. New setups should prefer the split files above.

### Recommended local bootstrap commands

Create and activate a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Then choose one of the following:

#### Core / CLI only

```bash
python -m pip install -r requirements/core.txt
perceptrome --config config/stream_config.yaml init
```

#### Web local development

```bash
python -m pip install -r requirements/web.txt
npm install --prefix perceptrome_web/client
python -m alembic -c perceptrome_web/server/alembic.ini upgrade head
```

#### Dev / test workspace

```bash
python -m pip install -r requirements/dev.txt
npm install --prefix perceptrome_web/client
```

#### Optional GPU training/generation add-on

```bash
python -m pip install -r requirements/gpu-cu12.txt
```

### Makefile shortcuts

```bash
make setup-core
make setup-web
make setup-dev
make setup-gpu
```

Use `make help` to list the available bootstrap and local-run targets.

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

Generate multiple plasmid candidates, rank by simple heuristics, and include optional reconstruction scoring:

```bash
perceptrome generate-plasmid \
  --length-bp 10000 \
  --num-candidates 8 \
  --top-k 3 \
  --roundtrip-score \
  --recon-weight 0.1 \
  --output generated/novel_plasmid.fasta
```

Generate protein sequence:

```bash
perceptrome generate-protein --length-aa 600 --output generated/novel_protein.faa
```

Generate and rank multiple protein candidates in one request:

```bash
perceptrome generate-protein \
  --length-aa 600 \
  --num-candidates 8 \
  --top-k 3 \
  --roundtrip-score \
  --recon-weight 0.1 \
  --output generated/novel_protein.faa
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

### Hierarchical dual-path model

A new `training.model_type: hierarchical` path is available. It keeps two late-fused branches alive:

- sequence branch: local 1D CNN encoder
- Bio-AST branch: node encoder + tree RvNN encoder

Then the model applies late fusion, predicts latent `mu/logvar`, decodes sequence reconstruction, and exposes critic heads (`validity`, `novelty`, `gc_fraction`).

Useful config keys under `training`:

- `hierarchical_latent_dim`
- `ast_tree_layers`, `ast_node_type_vocab_size`
- `hierarchical_ablation_mode` (`hierarchical`, `cnn_only`, `ast_only`)
- staged schedule knobs: `stage_a_steps`, `stage_b_steps`, `stage_c_steps`, `stage_d_steps`
- `critic_loss_weight`

The existing model types remain unchanged (`mlp`, `transformer`, `ssm`, `tree`, `hybrid`).

## Running tests

```bash
python -m unittest discover -s tests -v
```

## Optional tools

- **Raylib scope visualizer**
  - See `raylib_visualizer/README.md` for build/run instructions.

## Notes

- Some commands contact NCBI; set a real email in config for polite API usage.
- First runs may be slower due to cache warm-up and encoding.
- The default config is intentionally lightweight for smoke-testing and iteration.

## License

No explicit license file is currently included in this repository. Add a `LICENSE` file if you plan to distribute the project.
