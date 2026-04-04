# Protein lane CLI runbook (WSL Ubuntu)

This runbook is a practical, repeatable path for running a **protein-focused lane** in WSL Ubuntu:

`uniprot-count -> uniprot-fetch -> split-create -> stream (aa) -> generate-protein -> fold-* -> inspect/export`

It also documents run layout expectations, manifest locations, and recovery guidance.

## 1) WSL Ubuntu prerequisites

## Supported baseline

- WSL2 + Ubuntu 22.04/24.04.
- Python 3.10+ (3.11 recommended).
- Build/runtime tools for Python dependencies.

## Package install baseline

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  build-essential pkg-config git curl wget jq ca-certificates
```

Optional but useful:

```bash
sudo apt install -y unzip htop tmux
```

## Verify tools

```bash
python3 --version
pip3 --version
git --version
```

---

## 2) Virtualenv setup and requirements file choice

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies based on your intent:

- **CLI-only lane (recommended for this runbook):**
  ```bash
  python -m pip install -r requirements/core.txt
  ```
- **Developer/test workspace:**
  ```bash
  python -m pip install -r requirements/dev.txt
  ```
- **GPU add-on packages (optional):**
  ```bash
  python -m pip install -r requirements/gpu-cu12.txt
  ```

> Notes:
> - Prefer split requirement files over `requirements.txt` for new setups.
> - Run `perceptrome --help` after install to confirm entry point resolution.

---

## 3) ColabFold binary discovery validation (required before `fold-*`)

Perceptrome resolves the ColabFold executable in this order:

1. `--colabfold-bin /path/to/colabfold_batch`
2. `PERCEPTROME_COLABFOLD_BIN=/path/to/colabfold_batch`
3. `colabfold_batch` (or `colabfold_batch.sh`) on `PATH`

If none resolve, fold commands fail with a “Unable to locate ColabFold executable” error.

## Quick validation checks

### A) Explicit flag path

```bash
/path/to/colabfold_batch --help >/dev/null
perceptrome fold-one proteins/sample.fasta --colabfold-bin /path/to/colabfold_batch --num-recycle 1 --num-models 1
```

### B) Environment variable path

```bash
export PERCEPTROME_COLABFOLD_BIN=/path/to/colabfold_batch
"$PERCEPTROME_COLABFOLD_BIN" --help >/dev/null
```

### C) PATH discovery

```bash
which colabfold_batch || which colabfold_batch.sh
colabfold_batch --help >/dev/null 2>&1 || colabfold_batch.sh --help >/dev/null 2>&1
```

If you maintain multiple installs, prefer `--colabfold-bin` per command for deterministic behavior.

---

## 4) Complete protein lane walkthrough

All examples assume:

- Repository root as working directory.
- Active virtualenv.
- Default config file: `config/stream_config.yaml`.

## 4.1 Initialize run/state directories

```bash
perceptrome init --config config/stream_config.yaml
```

## 4.2 Estimate dataset size (`uniprot-count`)

```bash
perceptrome uniprot-count \
  --query 'reviewed:true AND taxonomy_id:9606 AND fragment:false' \
  --json
```

Use this count to choose `--records-per-shard` for fetch.

## 4.3 Download dataset and build catalog (`uniprot-fetch`)

```bash
perceptrome uniprot-fetch \
  --query 'reviewed:true AND taxonomy_id:9606 AND fragment:false' \
  --output-dir cache/fasta/uniprot/human_reviewed \
  --prefix hsap_reviewed \
  --records-per-shard 25000
```

Expected artifacts:

- `cache/fasta/uniprot/human_reviewed/hsap_reviewed.part-*.fasta[.gz]`
- `cache/fasta/uniprot/human_reviewed/hsap_reviewed.manifest.json`
- `cache/fasta/uniprot/human_reviewed/hsap_reviewed.catalog.txt`

## 4.4 Create deterministic split (`split-create`)

```bash
perceptrome split-create \
  --catalog cache/fasta/uniprot/human_reviewed/hsap_reviewed.catalog.txt \
  --name hsap_reviewed_aa \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 1337
```

Split file default location:

- `state/splits/hsap_reviewed_aa.json`

## 4.5 Train stream lane in amino-acid mode (`stream aa`)

```bash
perceptrome stream \
  --catalog cache/fasta/uniprot/human_reviewed/hsap_reviewed.catalog.txt \
  --tokenizer aa \
  --source fasta
```

Useful optional controls:

- `--experiment-id <id>` to force run ID.
- `--max-epochs`, `--batch-size`, `--window-size`, `--stride` for lane tuning.

## 4.6 Generate candidate proteins (`generate-protein`)

Single candidate:

```bash
perceptrome generate-protein \
  --length-aa 600 \
  --output generated/novel_protein.faa
```

Ranked candidates:

```bash
perceptrome generate-protein \
  --length-aa 600 \
  --num-candidates 8 \
  --top-k 3 \
  --roundtrip-score \
  --recon-weight 0.1 \
  --top-k-output generated/novel_protein_topk.faa \
  --output generated/novel_protein.faa
```

## 4.7 Fold generated proteins (`fold-one` / `fold-batch`)

Single FASTA:

```bash
perceptrome fold-one generated/novel_protein.faa \
  --num-recycle 3 \
  --num-models 5
```

Batch directory with filtering + resume support:

```bash
perceptrome fold-batch generated/ \
  --min-protein-aa 50 \
  --max-protein-aa 1200 \
  --resume \
  --keep-going
```

## 4.8 Inspect and export summaries

```bash
perceptrome fold-inspect <run_id>
perceptrome fold-export <run_id>
```

You can also pass a direct path to `summary.json`/`batch_summary.json` in both commands.

---

## 5) Run layout contract and manifest locations

Perceptrome creates a canonical run tree at `runs/<run_id>/`:

- `inputs/` - copied input files (for fold, input FASTA copies).
- `artifacts/` - command artifacts (including `artifacts/fold/<protein_id>/...`).
- `metrics/` - metric outputs when applicable.
- `outputs/` - machine-readable summaries/exports.
- `provenance/` - logs (`stdout`/`stderr` for fold commands).
- `manifest.json` - canonical run index and merged artifact/metrics/provenance updates.

For protein folding runs, check:

- `runs/<run_id>/outputs/summary.json`
- `runs/<run_id>/outputs/summary.tsv`
- `runs/<run_id>/outputs/batch_summary.json` (batch)
- `runs/<run_id>/outputs/batch_summary.tsv` (batch)
- `runs/<run_id>/manifest.json`

For UniProt fetches, run-level output is separate from `runs/` and written by prefix:

- `<prefix>.manifest.json` (dataset/shard manifest)
- `<prefix>.catalog.txt` (accession catalog)

---

## 6) Resume/retry and troubleshooting guidance

## Resume/retry patterns

- **`uniprot-fetch --resume`**
  - Reuses existing complete shard set if manifest+checksums+shard naming still match.
  - If query/prefix/shard parameters changed, run a fresh prefix or remove partial outputs.

- **`fold-batch --resume`**
  - Reuses existing successful fold outputs per protein under `runs/<run_id>/artifacts/fold/<protein_id>/`.
  - Combine with `--keep-going` for long runs so one failure does not abort all pending proteins.

- **Stable run IDs for retries**
  - Reuse `--run-id` in fold commands when retrying on the same input set.
  - Avoid changing filenames between retries if you expect resume behavior.

## Common failures and fixes

1. **ColabFold binary not found / not executable**
   - Validate the three discovery methods (flag, env var, PATH).
   - Check execute bit:
     ```bash
     ls -l /path/to/colabfold_batch
     chmod +x /path/to/colabfold_batch
     ```

2. **`fold-one` or `fold-batch` returns status 2**
   - Status `2` indicates at least one fold record failed.
   - Read logs:
     - `runs/<run_id>/provenance/*.stderr.log`
     - `runs/<run_id>/provenance/*.stdout.log`
   - Then retry with corrected ColabFold/runtime settings.

3. **No FASTA files found in batch directory**
   - `fold-batch` only picks files ending with `.fasta`, `.fa`, or `.faa`.
   - Confirm input naming and directory.

4. **All proteins skipped in `fold-batch`**
   - Re-check `--min-protein-aa` / `--max-protein-aa` thresholds.
   - Inspect sequence lengths before rerun.

5. **UniProt fetch interrupted or inconsistent output**
   - If resume does not trigger, ensure shard files and manifest are intact and match the expected prefix.
   - Safer fallback: write to a new prefix (e.g., `--prefix hsap_reviewed_retry1`).

6. **Slow or flaky WSL networking to external APIs**
   - Retry command; network/transient API issues can surface during UniProt download.
   - Prefer smaller shard sizes if repeated interruptions occur.

---

## 7) Minimal end-to-end command set (copy/paste)

```bash
# 0) bootstrap
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/core.txt
perceptrome init --config config/stream_config.yaml

# 1) uniprot count/fetch
perceptrome uniprot-count --query 'reviewed:true AND taxonomy_id:9606 AND fragment:false' --json
perceptrome uniprot-fetch \
  --query 'reviewed:true AND taxonomy_id:9606 AND fragment:false' \
  --output-dir cache/fasta/uniprot/human_reviewed \
  --prefix hsap_reviewed \
  --records-per-shard 25000

# 2) split + aa stream
perceptrome split-create --catalog cache/fasta/uniprot/human_reviewed/hsap_reviewed.catalog.txt --name hsap_reviewed_aa
perceptrome stream --catalog cache/fasta/uniprot/human_reviewed/hsap_reviewed.catalog.txt --tokenizer aa --source fasta

# 3) generate protein + fold + inspect/export
perceptrome generate-protein --length-aa 600 --output generated/novel_protein.faa
perceptrome fold-one generated/novel_protein.faa --num-recycle 3 --num-models 5
perceptrome fold-inspect <run_id>
perceptrome fold-export <run_id>
```
