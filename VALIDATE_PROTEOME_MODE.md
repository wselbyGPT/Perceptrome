# Genostream — Proteome (AA) mode quick validation (GenBank/CDS)

This repo supports `--tokenizer aa` to train the VAE on **protein windows**.

In AA mode, Genostream can derive proteins two ways:

- **GenBank (recommended):** fetches a GenBank flatfile and extracts **CDS proteins** (prefers `/translation`; falls back to translating CDS coordinates from ORIGIN).
- **FASTA fallback:** fetches FASTA and runs a naive 6-frame ORF finder.

You can select the record source with `--source`:

- Default: **FASTA** for `base` / `codon`
- Default: **GenBank** for `aa`

## 0) One-time notes

- **Checkpoint incompatibility**: `model/checkpoints/latest.pt` is shared across tokenizers. If you previously trained `base` or `codon`, delete the checkpoint before training AA:

```bash
rm -f model/checkpoints/latest.pt
```

- AA mode uses **`--window-size` and `--stride` in amino-acid units**.

## 1) Validate on 1 accession

Example uses `NC_002134.1` (replace if needed).

```bash
cd ~/perceptrome
source venv/bin/activate

# Optional: prefetch GenBank (AA mode will auto-fetch if missing)
python3 stream_train.py fetch-one NC_002134.1 --source genbank

# Train AA-VAE on CDS-derived protein windows
python3 stream_train.py train-one NC_002134.1 \
  --tokenizer aa \
  --source genbank \
  --min-orf-aa 90 \
  --window-size 256 \
  --stride 128 \
  --steps 50 \
  --batch-size 16

# Visualize reconstruction error + side metric (hydrophobic fraction in AA mode)
python3 stream_train.py scope-one NC_002134.1 \
  --tokenizer aa \
  --source genbank \
  --min-orf-aa 90 \
  --window-size 256 \
  --stride 128
```

If you want to compare CDS-vs-ORF quickly:

```bash
# ORF-based proteins from FASTA
python3 stream_train.py train-one NC_002134.1 --tokenizer aa --source fasta --min-orf-aa 90 --window-size 256 --stride 128 --steps 50 --batch-size 16
```

## 2) Minimal stream pass (10 plasmids)

```bash
head -n 10 plasmid_accessions.txt > config/plasmids_10.txt

python3 stream_train.py stream \
  --catalog config/plasmids_10.txt \
  --tokenizer aa \
  --source genbank \
  --min-orf-aa 90 \
  --window-size 256 \
  --stride 128 \
  --steps-per-plasmid 5 \
  --batch-size 16 \
  --max-epochs 1 \
  --delete-cache
```

## 3) Generate a novel protein

```bash
python3 stream_train.py generate-protein \
  --length-aa 600 \
  --window-aa 256 \
  --temperature 1.0 \
  --latent-scale 1.0 \
  --output generated/novel_protein.faa
```
