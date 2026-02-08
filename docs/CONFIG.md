# Configuration & CLI Reference

This project is driven by a YAML configuration file (default: `config/stream_config.yaml`) and a CLI wrapper (`stream_train.py`) that dispatches to the same internal command set.

## CLI entrypoint

All commands below are available via:

```bash
python3 stream_train.py <command> [options]
```

You can also pass a custom config file:

```bash
python3 stream_train.py --config path/to/config.yaml <command>
```

## Commands

| Command | Purpose |
| --- | --- |
| `init` | Initialize cache/log/state directories and create a fresh state file. |
| `catalog-show <path>` | Print summary of a catalog file (one accession per line). |
| `fetch-one <accession>` | Fetch a record from NCBI (FASTA or GenBank). |
| `encode-one <accession>` | Encode windows to `.npy` for training. |
| `train-one <accession>` | Train on a single accession. |
| `scope-one <accession>` | Visualize reconstruction error + GC content in a curses UI. |
| `scope-stream <accession>` | Train and update scope UI while streaming steps. |
| `stream --catalog <path>` | Iterate over a catalog file with streaming training. |
| `generate-plasmid` | Sample DNA windows from the model and emit FASTA. |
| `generate-protein` | Sample amino acids from the model and emit FASTA. |

## Common CLI flags

These are shared across several commands.

### Tokenizer + sequence source

- `--tokenizer {base,codon,aa}`: choose tokenization mode.
- `--frame-offset {0,1,2}`: codon frame offset (codon mode only).
- `--source {fasta,genbank}`: record source (defaults to fasta for base/codon, genbank for aa).

### AA/GenBank protein filters

These are only relevant when `--tokenizer aa` and `--source genbank`:

- `--min-orf-aa`: minimum ORF length (aa).
- `--min-protein-aa`: alias for `--min-orf-aa`.
- `--max-protein-aa`: reject proteins longer than this length.
- `--strict-cds`: do not fall back to naive ORFs; fail if no CDS passes filters.
- `--require-translation`: require `/translation` in GenBank CDS.
- `--x-free`: reject proteins containing `X` or stop markers.
- `--require-start-m`: require protein to start with `M`.
- `--reject-partial-cds`: drop CDS locations with `<` or `>`.

### Proteome sampling & curriculum

- `--protein-len-min` / `--protein-len-max`: apply extra protein-length filters.
- `--max-windows-per-protein`: cap windows sampled per protein.
- `--translation-only` / `--allow-translated`: force or relax reliance on `/translation`.
- `--no-curriculum`: disable proteome curriculum phases for the run.

### Loss + masking (AA)

- `--loss-type {mse,ce}`: reconstruction loss override.
- `--mask-prob`: random AA masking probability (input corruption).
- `--span-mask-prob` / `--span-mask-len`: contiguous span mask probability/length.

## Configuration schema

The configuration file mirrors the defaults in `perceptrome/config.py`, with overrides in `config/stream_config.yaml`.

### `ncbi`

| Key | Type | Description |
| --- | --- | --- |
| `email` | string | Your email for NCBI queries. |
| `api_key` | string or null | Optional NCBI API key. |
| `max_retries` | int | Retry count for NCBI calls. |
| `backoff_seconds` | float | Backoff between retries. |

### `training`

| Key | Type | Description |
| --- | --- | --- |
| `steps_per_plasmid` | int | Gradient steps per accession. |
| `batch_size` | int | Minibatch size. |
| `window_size` | int | Genome window length (base/codon). |
| `stride` | int | Genome stride (base/codon). |
| `max_stream_epochs` | int | Upper bound on stream epochs. |
| `shuffle_catalog` | bool | Shuffle catalog entries each epoch. |
| `hidden_dim` | int | MLP latent dimension. |
| `model_type` | string | `mlp` or `transformer`. |
| `transformer_d_model` | int | Transformer model dim. |
| `transformer_nhead` | int | Transformer attention heads. |
| `transformer_layers` | int | Transformer encoder layers. |
| `transformer_dropout` | float | Transformer dropout rate. |
| `learning_rate` | float | Optimizer LR. |
| `beta_kl` | float | KL weight. |
| `kl_warmup_steps` | int | Warmup steps for KL. |
| `max_grad_norm` | float | Gradient clipping threshold. |
| `tokenizer` | string | `base`, `codon`, or `aa`. |
| `frame_offset` | int | Codon frame offset. |
| `protein_window_aa` | int | Proteome window length (aa). |
| `protein_stride_aa` | int | Proteome stride (aa). |
| `min_orf_aa` | int | Minimum ORF length (aa). |
| `strict_cds` | bool | Fail if no CDS passes filters. |
| `require_translation` | bool | Require GenBank `/translation`. |
| `x_free` | bool | Reject proteins with `X` markers. |
| `require_start_m` | bool | Require protein starts with `M`. |
| `reject_partial_cds` | bool | Reject partial CDS locations. |
| `max_protein_aa` | int or null | Optional max protein length. |
| `aa_mask_prob` | float | Random AA mask probability. |
| `aa_span_mask_prob` | float | Span mask probability. |
| `aa_span_mask_len` | int | Span mask length (aa). |
| `max_windows_per_protein` | int | Sampling cap per protein. |
| `protein_len_min` | int or null | Minimum protein length. |
| `protein_len_max` | int or null | Maximum protein length. |
| `translation_only` | bool | Use only `/translation` proteins. |
| `curriculum_enabled` | bool | Enable proteome curriculum. |
| `curriculum_steps` | list | Step boundaries for curriculum. |
| `curriculum_phases` | list | List of phase dicts (filters + masks). |

### `io`

| Key | Type | Description |
| --- | --- | --- |
| `cache_fasta_dir` | string | Where FASTA downloads are cached. |
| `cache_genbank_dir` | string | Where GenBank downloads are cached. |
| `cache_encoded_dir` | string | Where `.npy` encoded windows are stored. |
| `model_dir` | string | Model output directory. |
| `checkpoints_dir` | string | Checkpoint directory. |
| `logs_dir` | string | Training/log output directory. |
| `state_file` | string | State file path for streaming resume. |

## Examples

### AA proteome run (GenBank CDS only)

```bash
python3 stream_train.py train-one CP060383.1 \
  --tokenizer aa \
  --source genbank \
  --strict-cds \
  --require-translation \
  --min-orf-aa 90 \
  --protein-len-max 800
```

### Codon mode

```bash
python3 stream_train.py train-one L09137.2 \
  --tokenizer codon \
  --window-size 510 \
  --stride 255 \
  --frame-offset 0
```
