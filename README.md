# Perceptrome

Perceptrome is a Python toolkit for **streaming representation learning on genomic and proteomic sequences**.
It provides a CLI-first workflow for:

- building accession catalogs,
- fetching and encoding sequence records,
- training VAE-style models in streaming mode,
- inspecting reconstruction/error dynamics with scope tools,
- generating candidate plasmid/protein sequences from trained models,
- folding protein FASTA inputs via ColabFold/AlphaFold3 into run-tracked structure artifacts,
- annotating genomes (GenBank/FASTA → Bio-AST → BRG regulatory features), and
- exploring the learned latent space via clustering, interpolation, and seed selection.

The repository also includes **Perceptrome Web**, a browser-based React/FastAPI application for authenticated administration, dataset/runs views, and websocket-backed run monitoring. See [`perceptrome_web/README.md`](perceptrome_web/README.md) for setup details covering the client, API server, PostgreSQL, Alembic migrations, bootstrap admin flow, SPA/API wiring, and WebSocket expectations.

## Highlights

- **CLI for end-to-end workflows** (catalog, fetch, encode, train, stream, generate).
- **Multiple tokenization modes**: base, codon, and amino-acid/proteome (`aa`).
- **Config-driven runs** via YAML (`config/stream_config.yaml`) with CLI overrides.
- **NCBI-integrated data acquisition** with local caching for FASTA/GenBank/encoded artifacts.
- **Training + observability utilities** including TensorBoard launcher and scope visualizers.
- **Generation + validation commands** for plasmid and protein candidates.
- **Structure lane (ColabFold/AlphaFold3)** with normalized fold summaries and manifest-indexed artifacts.
- **Genome annotation lane** that parses GenBank/FASTA input through Bio-AST and the BRG inference layer, emitting per-sequence feature counts (genes, CDS, promoters, operators, RBS, terminators) as run-tracked JSON/TSV artifacts.
- **Latent space analysis tools** — `latent-cluster` (k-means + UMAP projections), `latent-interpolate` (walk the straight-line path between two encoded accessions), and `latent-seeds` (pick archetype/outlier accessions per cluster to seed the design loop).
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
| Latent analysis | `pip install 'perceptrome[analysis]'` | scikit-learn (k-means) and umap-learn (UMAP projections) for `latent-cluster`, `latent-seeds` |

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

Use `make help` to list the available bootstrap and local development targets.

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

### 6) Fold proteins (monomer)

Perceptrome supports two pluggable structure backends via `--engine`:

- `colabfold` *(default)* — wraps ColabFold's AlphaFold2 inference pipeline.
- `alphafold3` — direct DeepMind AlphaFold 3 `run_alphafold.py` invocation.

Both backends write into the same run layout and share the same summary/manifest schema, so downstream tools (`fold-inspect`, `fold-export`) are engine-agnostic.

#### ColabFold backend

Perceptrome assumes ColabFold is installed separately and available either via:

- `--colabfold-bin /path/to/colabfold_batch`
- `PERCEPTROME_COLABFOLD_BIN=/path/to/colabfold_batch`
- `colabfold_batch` on `PATH`

Single protein:

```bash
perceptrome fold-one proteins/my_target.fasta --num-recycle 3 --num-models 5
```

Batch directory:

```bash
perceptrome fold-batch proteins/ --min-protein-aa 50 --max-protein-aa 1200 --keep-going
```

#### AlphaFold 3 backend

AlphaFold 3 must be installed separately per DeepMind's instructions
(<https://github.com/google-deepmind/alphafold3>). You will need three things
on disk:

1. The `run_alphafold.py` entrypoint.
2. A model parameters directory (weights obtained via DeepMind's access
   request process).
3. A sequence databases directory populated by the AF3 `fetch_databases.sh`
   helper (BFD, UniRef, MGnify, RNAcentral, etc.).

These can be provided via CLI flags, environment variables, or
`config/stream_config.yaml` (`structure.alphafold3.*`).

| Setting | CLI flag | Environment variable |
| --- | --- | --- |
| Entrypoint | `--alphafold3-bin` | `PERCEPTROME_ALPHAFOLD3_BIN` |
| Model parameters dir | `--alphafold3-model-dir` | `PERCEPTROME_ALPHAFOLD3_MODEL_DIR` |
| Sequence databases dir | `--alphafold3-db-dir` | `PERCEPTROME_ALPHAFOLD3_DB_DIR` |

Single protein:

```bash
perceptrome fold-one proteins/my_target.fasta \
  --engine alphafold3 \
  --alphafold3-bin /opt/alphafold3/run_alphafold.py \
  --alphafold3-model-dir /opt/alphafold3/models \
  --alphafold3-db-dir /opt/alphafold3/databases \
  --num-seeds 1 --num-diffusion-samples 5
```

Batch directory (honors the same filtering/resume flags as the ColabFold path):

```bash
perceptrome fold-batch proteins/ \
  --engine alphafold3 \
  --min-protein-aa 50 --max-protein-aa 1200 \
  --keep-going
```

Under the hood the AlphaFold 3 backend:

- Converts each protein FASTA into an AF3 JSON input spec
  (`{"dialect": "alphafold3", "sequences": [{"protein": {...}}]}`) written to
  `runs/<run_id>/artifacts/fold/<protein_id>/<protein_id>.input.json`.
- Runs `run_alphafold.py --json_path=... --output_dir=... --model_dir=... --db_dir=...`.
- Parses `<protein_id>_summary_confidences.json` (pTM, ranking score) and
  `<protein_id>_confidences.json` (token plDDT) into the normalized
  `FoldSummaryRecord` schema alongside the rank-1 CIF structure.

Inspect and export:

```bash
perceptrome fold-inspect <run_id>
perceptrome fold-export <run_id>
```

Outputs are written into the standard run layout:

- `runs/<run_id>/inputs/` (copied FASTA inputs)
- `runs/<run_id>/artifacts/fold/...` (raw ColabFold outputs)
- `runs/<run_id>/outputs/summary.json|summary.tsv`
- `runs/<run_id>/outputs/batch_summary.json|batch_summary.tsv` (batch runs)
- `runs/<run_id>/provenance/*.log` (stdout/stderr logs)

This milestone intentionally excludes: multimer orchestration, RFD3 integration, GUI molecular viewers, and direct training-loop coupling.

For a WSL-first, command-by-command protein lane walkthrough (including ColabFold discovery checks, run layout contracts, and resume/retry troubleshooting), see [`docs/protein_lane_wsl.md`](docs/protein_lane_wsl.md).

### 7) Annotate genomes

Annotate a single GenBank or FASTA file:

```bash
perceptrome genome-annotate-one genomes/my_plasmid.gb
```

Batch-annotate a directory:

```bash
perceptrome genome-annotate-batch genomes/ --keep-going --resume
```

Inspect annotation counts from a run:

```bash
perceptrome genome-annotate-inspect <run_id>
```

Export results to JSON + TSV:

```bash
perceptrome genome-annotate-export <run_id> --output-dir results/annotations/
```

### 8) Latent space analysis

**Cluster** an encoded catalog (requires `pip install 'perceptrome[analysis]'`):

```bash
perceptrome latent-cluster \
  --catalog config/plasmids_100.txt \
  --n-clusters 10 \
  --tokenizer base
```

**Select seeds** (archetype + outlier per cluster) from the cluster run:

```bash
perceptrome latent-seeds <run_id> --outliers
```

**Interpolate** between two accessions along their latent-space path:

```bash
perceptrome latent-interpolate ACC_A ACC_B \
  --steps 12 \
  --output results/interp_A_to_B.fasta
```

## Genome annotation lane

The genome annotation lane processes GenBank or FASTA inputs through Perceptrome's Bio-AST parser and the BRG regulatory inference layer, producing per-sequence feature counts indexed as run artifacts.

### `genome-annotate-one`

Annotate a single file and write results to `runs/<run_id>/`.

```bash
perceptrome genome-annotate-one genomes/my_plasmid.gb [--run-id RUN_ID]
```

Outputs written:
- `runs/<run_id>/outputs/genome_summary.json` — structured record with accession, sequence length, CDS source, and all feature counts.
- `runs/<run_id>/outputs/genome_summary.tsv` — flat tabular copy.
- `runs/<run_id>/artifacts/genome/<accession>/annotation.json` — full Bio-AST annotation artifact.

### `genome-annotate-batch`

Annotate all GenBank/FASTA files in a directory.

```bash
perceptrome genome-annotate-batch genomes/ [--keep-going] [--resume]
```

| Flag | Effect |
| --- | --- |
| `--keep-going` | Continue on per-file errors rather than aborting. |
| `--resume` | Skip accessions that already have an annotation artifact from a prior run. |

### `genome-annotate-inspect`

Print a human-readable summary of counts from a finished run.

```bash
perceptrome genome-annotate-inspect <run_id_or_path>
```

Accepts either a run ID (resolved relative to `runs/`) or a direct path to `genome_summary.json`.

### `genome-annotate-export`

Write the run's annotation records to a standalone JSON + TSV for downstream analysis.

```bash
perceptrome genome-annotate-export <run_id_or_path> [--output-dir DIR]
```

Default output directory is the current working directory.

## Latent space analysis

These three commands form a closed design loop: **cluster → seeds → interpolate**.

### `latent-cluster`

Encode a catalog in bulk, collect the per-accession `mu` vectors (mean-pooled across windows), run k-means, and optionally project to 2D via UMAP.

```bash
perceptrome latent-cluster \
  --catalog config/plasmids_100.txt \
  --n-clusters 8 \
  --umap-n-neighbors 15 \
  --umap-min-dist 0.1 \
  --tokenizer base \
  [--run-id RUN_ID]
```

Requires `pip install 'perceptrome[analysis]'` (scikit-learn + umap-learn). Both dependencies are soft-imported — if they are absent the command exits with a helpful install message.

Outputs:
- `runs/<run_id>/outputs/latent_cluster_summary.json` — per-accession cluster ID, UMAP x/y, mu norm/mean, and cluster centroids.
- `runs/<run_id>/outputs/latent_cluster_summary.tsv` — flat tabular copy.
- `runs/<run_id>/artifacts/latent_vectors.npy` — raw mu matrix (shape `[N, latent_dim]`), used by `latent-seeds`.

### `latent-seeds`

Given a `latent-cluster` run, rank accessions within each cluster by L2 distance to the centroid and emit a selection catalog.

```bash
perceptrome latent-seeds <run_id_or_path> [--outliers] [--vectors PATH] [--output-dir DIR]
```

| Flag | Effect |
| --- | --- |
| `--outliers` | Also include the farthest accession per cluster (the outlier). |
| `--vectors PATH` | Override the auto-derived `latent_vectors.npy` path. |
| `--output-dir DIR` | Write outputs here instead of the run's `outputs/` directory. |

Outputs:
- `latent_seeds.json` — catalog with one or two selected accessions per cluster, their roles (`archetype` / `outlier`), cluster ID, and distance to centroid.
- `latent_seeds.catalog.txt` — plain accession list ready to pass to `latent-interpolate` or `stream`.

### `latent-interpolate`

Encode two accessions, interpolate `--steps` evenly-spaced points along the straight line between their `mu` vectors, decode each point back to sequence tokens, and emit a multi-FASTA.

```bash
perceptrome latent-interpolate ACC_A ACC_B \
  --steps 10 \
  --n-windows 1 \
  --temperature 0.0 \
  --output results/interp.fasta
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--steps` | 8 | Number of interpolation steps including endpoints. |
| `--n-windows` | 1 | Number of decode windows per step (for longer sequences). |
| `--temperature` | 0.0 | 0 = greedy argmax decode; > 0 = temperature sampling. |
| `--output` | `latent_interp.fasta` | Output FASTA path. |

Each FASTA record header encodes the step index and interpolation parameter:

```
>interp_step3_of10|t=0.2222|a=ACC_A|b=ACC_B
```

A JSON summary (`latent_interp_summary.json` alongside the FASTA) records run metadata, window/tokenizer settings, and the per-step `(t, sequence)` table.

### Full cluster → seeds → interpolate workflow

```bash
# 1. Cluster the training catalog
perceptrome latent-cluster \
  --catalog config/plasmids_100.txt \
  --n-clusters 8 \
  --tokenizer base

# 2. Pick archetype + outlier accessions per cluster
perceptrome latent-seeds <run_id> --outliers

# 3. Inspect the seed catalog
cat runs/<run_id>/outputs/latent_seeds.json

# 4. Walk the latent path between two seeds from different clusters
perceptrome latent-interpolate SEED_A SEED_B \
  --steps 12 \
  --output results/interp_SEED_A_to_SEED_B.fasta
```

## UniProt ingestion

Use these commands to estimate query size, download shard FASTA files, and generate a Perceptrome-ready accession catalog.

`uniprot-fetch` resolves shard size with **CLI override > config**. If `--records-per-shard` is omitted, it uses `uniprot.records_per_shard` from `config/stream_config.yaml` (default `50000`).

Count-only checks:

```bash
# Uses `uniprot.default_query` from config/stream_config.yaml
perceptrome uniprot-count
# Explicit query form
perceptrome uniprot-count --query 'fragment:false' --json
```

Fetch reviewed proteins (Swiss-Prot only):

```bash
perceptrome uniprot-fetch \
  --query 'reviewed:true AND taxonomy_id:9606 AND fragment:false' \
  --output-dir cache/fasta/uniprot/human_reviewed \
  --prefix hsap_reviewed \
  --records-per-shard 25000
```

Fetch all proteins for a taxon, including isoforms, and gzip shards:

```bash
perceptrome uniprot-fetch \
  --query 'taxonomy_id:83333 AND fragment:false' \
  --include-isoforms \
  --gzip-output \
  --output-dir cache/fasta/uniprot/ecoli_all \
  --prefix ecoli_all \
  --records-per-shard 10000
```

Fetch all non-isoform records for a broad proteome pull (using config default shard size):

```bash
perceptrome uniprot-fetch \
  --query 'proteome:UP000005640 AND fragment:false' \
  --output-dir cache/fasta/uniprot/human_proteome \
  --prefix human_proteome
```

### UniProt output artifacts

`uniprot-fetch` writes three artifact types under your selected output prefix:

- **Shard FASTA files**: `<prefix>.part-00001.fasta` (or `.fasta.gz` when `--gzip-output` is enabled).
- **Manifest JSON**: `<prefix>.manifest.json` with query metadata, shard paths/checksums, record counts, and accession preview.
- **Generated catalog**: `<prefix>.catalog.txt` (the path printed as `catalog:` in CLI output), containing extracted UniProt accessions for downstream jobs.

### Downstream integration (AA pipeline)

After ingestion, use the generated catalog path directly with existing training/encoding flows. A common sequence is:

```bash
# 1) Create deterministic splits from the generated UniProt catalog
perceptrome split-create \
  --catalog cache/fasta/uniprot/human_reviewed/hsap_reviewed.catalog.txt \
  --name uniprot_hsap

# 2) Train in AA mode against the same catalog
perceptrome stream \
  --catalog cache/fasta/uniprot/human_reviewed/hsap_reviewed.catalog.txt \
  --tokenizer aa \
  --source fasta

# 3) Encode a representative accession from that catalog
perceptrome encode-one P12345 --tokenizer aa --source fasta
```

## CLI overview

Global help:

```bash
perceptrome --help
```

Primary commands:

- `init`
- `catalog-show`, `catalog-generate`
- `uniprot-count`, `uniprot-fetch`
- `split-create`, `split-show`
- `fetch-one`
- `encode-one`
- `train-one`
- `stream`
- `scope-one`, `scope-stream`
- `tensorboard`
- `generate-plasmid`, `validate-plasmid`, `generate-protein`
- `fold-one`, `fold-batch`, `fold-inspect`, `fold-export`
- `genome-annotate-one`, `genome-annotate-batch`, `genome-annotate-inspect`, `genome-annotate-export`
- `latent-cluster`, `latent-interpolate`, `latent-seeds`

Most commands accept `--config` to point at a YAML config file.

## Configuration

Default config: `config/stream_config.yaml`.

The config includes:

- `ncbi`: email/api key and retry/backoff settings,
- `uniprot`: UniProt API/query/download settings,
- `training`: tokenizer/model/windowing/loss/logging parameters,
- `io`: cache/model/log/state paths.

Default `uniprot` keys in `config/stream_config.yaml`:

- `base_url: "https://rest.uniprot.org"`
- `default_query: "reviewed:true AND fragment:false"`
- `include_isoforms: false`
- `request_timeout: 60`
- `retries: 3`
- `backoff_seconds: 2.0`
- `records_per_shard: 50000`
- `gzip_output: false`

You can override many `training` values at the command line (for example model type, tokenizer, window/stride, batch size, and training steps).

### Hierarchical dual-path model

A new `training.model_type: hierarchical` path is available. It keeps two late-fused branches alive:

- sequence branch: local 1D CNN encoder
- Bio-AST branch: node encoder + tree RvNN encoder

### Bio-AST + Bio-Regulatory Graph (BRG)

The packaged schema in `perceptrome/bio_ast.py` now supports a backward-compatible BRG layer.

- **AST layer**: structural containment and coordinate spans.
- **BRG layer**: semantic/regulatory relationships (promoters, operators, RBS, terminators, transcript units, operons, and plasmid modules).

New BRG inference utilities live in `perceptrome/bio_reg_graph.py` and provide deterministic rule-based helpers:

- `infer_regulatory_features(...)`
- `infer_transcript_units(...)`
- `infer_operons(...)`
- `infer_modules(...)`
- `build_bio_regulatory_graph(...)`

For the standalone curses visualizer (`bio_ast_viz (1).py`):

- `e` exports tree AST JSON.
- `b` exports BRG-aware JSON with relationship edges.
- `g` toggles tree/graph inspection mode for selected-node edge introspection.

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
