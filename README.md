# perceptrome

A streaming genome/proteome learning playground , built for iterative experiments:
fetch → encode → train → visualize ("scope") → validate.

## Quick start

```bash
# from repo root
./build.sh

# (optional) activate your venv
# source venv/bin/activate

# run the streaming trainer (current stub / evolving)
python3 stream_train.py
```

## Project layout (high level)

- `scripts/` — core Python modules (fetch, encoding, training, scope)
- `config/` — YAML and catalog files
- `cache/` — downloaded + encoded artifacts
- `model/` — checkpoints and model outputs
- `state/` — progress / resume metadata
- `docs/` — project documentation
- `VALIDATE_PROTEOME_MODE.md` — validation notes for proteome mode

## Docs

- `docs/INSTALL.md`
- `docs/CONFIG.md`
- `docs/PIPELINE.md`
- `docs/TROUBLESHOOTING.md`
- `ARCHITECTURE.md`

## License

See `LICENSE`.
