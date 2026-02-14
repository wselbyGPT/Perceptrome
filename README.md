# Perceptrome

Perceptrome provides a sequence-model training pipeline with both a command-line interface and a Qt GUI.

## Requirements

- Python **3.10+**
- PySide6 **6.6+** (for GUI use)

## Installation

### End-user install (standard)

Install into your current environment:

```bash
pip install .
```

This provides two console commands:

- `perceptrome` → CLI entrypoint (`perceptrome.cli_main:main`)
- `perceptrome-gui` → Qt GUI entrypoint (`gui_qt.app:main`)

### Editable/developer install

For local development with editable imports:

```bash
pip install -e .[dev]
```

Use the same commands while developing:

```bash
perceptrome --help
perceptrome-gui
```
