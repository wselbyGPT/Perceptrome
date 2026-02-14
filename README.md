# Perceptrome

Perceptrome is a CLI-first workflow for fetching sequence records, encoding training windows, training models, and generating new plasmid/protein sequences.

## Strict quickstart

Run these steps in order from repository root.

### 0) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Init

```bash
python -m perceptrome.cli_main --config config/stream_config.yaml init
```

### 2) Validate environment before long runs

```bash
python -m perceptrome.cli_main --config config/stream_config.yaml doctor
```

### 3) Fetch

```bash
python -m perceptrome.cli_main --config config/stream_config.yaml fetch-one NC_005816.1 --source fasta
```

### 4) Encode

```bash
python -m perceptrome.cli_main --config config/stream_config.yaml encode-one NC_005816.1 --window-size 512 --stride 256 --tokenizer base
```

### 5) Train

```bash
python -m perceptrome.cli_main --config config/stream_config.yaml train-one NC_005816.1 --steps 200 --batch-size 16 --window-size 512 --stride 256 --tokenizer base
```

### 6) Generate + view output

Generate a plasmid FASTA:

```bash
python -m perceptrome.cli_main --config config/stream_config.yaml generate-plasmid --length-bp 10000 --output generated/novel_plasmid.fasta --tokenizer base
```

View the output file:

```bash
head -n 20 generated/novel_plasmid.fasta
```
