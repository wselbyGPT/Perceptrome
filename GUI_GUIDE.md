# Perceptrome GUI Quick Guide

This project ships with a Qt desktop GUI (`gui_qt`) that wraps common training/generation/view workflows.

## 1) Start the GUI

From the repository root:

```bash
python gui.py
```

Alternative entrypoint:

```bash
python -m gui_qt
```

Both launch the same `PerceptromeQt` window.

## 2) Home / Config tab

Use this tab first.

- **Project dir**: Working directory for Train/Generate commands.
- **stream_config.yaml**: Path to your stream configuration file.
- **Dataset list file**: Path to a list file (for example `config/plasmids_10.txt`).
- **Epochs, Batch size, Learning rate**: Saved UI settings for your run setup.
- Click **Save config** to persist settings via `QSettings`.
- Click **Go to Train tab** to jump directly into training.

## 3) Train tab

- Put your command in **Command** (runs via `bash -lc`).
- Typical first check:

```bash
python stream_train.py --help
```

- Click **Start** to run, **Stop** to terminate.
- Output streams live to the log area.
- Progress bar auto-updates when output includes percentages (e.g. `42%`) or epoch-style lines (`Epoch 2/10`).

## 4) Generate tab

Works like Train tab, but for generation commands.

Suggested first check command:

```bash
python stream_train.py generate --help
```

- Click **Start** to run generation.
- Click **Stop** to terminate.
- View live output and inferred progress in the panel.

## 5) View tab (PDF circular genome)

You can create and preview a circular-genome PDF from either:

- **Genome accession** (downloads FASTA through configured NCBI settings), or
- **FASTA path** (local file).

Then set:

- **Output PDF** path (default `generated/circular_genome.pdf`)
- Optional **Title**

Actions:

- **Generate PDF**: resolves sequence, writes the PDF, and auto-loads it in the embedded PDF viewer.
- **Open PDF**: loads an existing PDF path into the viewer.

## 6) History tab

Tracks key UI actions with timestamp/action/details rows.

- Use **Clear history** to reset the table.

## 7) Practical first-run workflow

1. Start GUI: `python gui.py`.
2. In **Home / Config**, set project directory and config paths.
3. Save config.
4. In **Train**, run a help command first, then your real train command.
5. In **Generate**, run your generation command.
6. In **View**, provide accession or FASTA and generate a PDF.

## Notes

- Train/Generate commands run exactly as typed in a shell context (`bash -lc`), so shell features and environment activation should work.
- Saved values persist between sessions.
- If a process ignores graceful termination, the app escalates from terminate to kill automatically.
