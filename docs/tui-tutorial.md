# Perceptrome TUI Tutorial

A complete walkthrough of the Perceptrome terminal user interface for
interactive genomic/proteomic ML training, generation, and inspection.

---

## 1. Launching the TUI

```bash
# Basic launch
perceptrome tui

# Open directly on a specific panel
perceptrome tui --panel train

# Open with a specific run already selected
perceptrome tui --run-id <RUN_ID>

# Open with a detail drawer already visible
perceptrome tui --detail-surface logs

# Use a custom config file
perceptrome --config config/my_config.yaml tui

# You can also run the module directly
python -m perceptrome.tui.app
```

---

## 2. Global Keyboard Shortcuts

These work from any panel in the application.

| Key            | Action                                         |
|----------------|-------------------------------------------------|
| `Ctrl+P`       | Open the command palette (launcher)            |
| `[`            | Switch to the previous panel                   |
| `]`            | Switch to the next panel                       |
| `Ctrl+L`       | Open the Logs detail drawer (right side)       |
| `Ctrl+D`       | Open the Diagnostics detail drawer             |
| `Ctrl+T`       | Open the Traceback detail drawer               |
| `Tab`          | Move focus to the next widget                  |
| `Shift+Tab`    | Move focus to the previous widget              |
| `Escape`       | Close the active detail drawer                 |
| `q`            | Quit the TUI                                   |

---

## 3. Panel-by-Panel Walkthrough

The TUI has 15 panels accessible via the tab bar at the top or with `[`/`]`.

---

### 3.1 Overview (tab: Overview)

The dashboard. Shows a summary of all job activity at a glance.

**What you see:**
- **Stat cards** -- Active, Completed, Failed, and Total job counts
- **Active Job** -- progress bar and step/stage info for the currently running job
- **Loss Trend** -- sparkline of the most recent loss values
- **Recent Jobs** -- table of the last 12 jobs with status, kind, stage, and message

**How to use it:**
- This is the landing page. Check here for a quick picture of what is running
  and what has finished.
- The sparkline automatically tracks whichever job is active. If none are active
  it shows the most recent job that produced loss data.

---

### 3.2 Config (tab: Config)

Edit and validate the YAML configuration before submitting jobs.

**What you see:**
- **Override input** -- type `key=value` overrides (e.g. `training.batch_size=8`)
- **Config path** -- which YAML file to load (default: `config/stream_config.yaml`)
- **Validation table** -- automated checks (window/stride positive, codon
  divisibility, batch size, checkpoints dir)
- **Active Overrides** -- list of overrides you have applied
- **Effective Config** -- the merged result as JSON

**Buttons:**
| Button           | Action                                            |
|------------------|---------------------------------------------------|
| Apply override   | Add the `key=value` string to the override list   |
| Reset overrides  | Clear all overrides back to the base config       |
| Use run config   | Copy config from the selected run's kind          |
| Set draft: Train | Set the draft job kind to `train_one`             |
| Set draft: Generate | Set the draft job kind to `generate_plasmid`   |

**Workflow:**
1. Set the config path if not using the default.
2. Type overrides one at a time and click **Apply override**.
3. Check the validation table -- all rows should say PASS.
4. Click **Set draft: Train** or **Set draft: Generate** to seed the draft, then
   switch to the Train panel to submit.

---

### 3.3 Catalogs & Data (tab: Catalogs)

Browse accession catalogs and launch data-related jobs directly.

**What you see:**
- **Catalog table** -- auto-discovered `.txt` files from `accessions/` and `config/`
  directories, showing directory, filename, and accession count
- **Preview** -- when you select a catalog row, the first 20 accessions are shown
- **Prepare Job** section -- accession input, source format dropdown (FASTA/GenBank),
  tokenizer dropdown (base/codon/aa)

**Buttons:**
| Button          | Action                                                |
|-----------------|-------------------------------------------------------|
| Train single    | Submit a `train_one` job for the entered accession    |
| Stream catalog  | Submit a `stream` job over the selected catalog       |
| Validate plasmid| Submit a `validate_plasmid` job for the accession     |
| Pretrain        | Submit a `pretrain` job on the selected catalog       |

**Workflow:**
1. Click a row in the catalog table to select it and preview accessions.
2. To train a single sequence: type an accession (e.g. `NC_000913`) and click
   **Train single**.
3. To stream over a whole catalog: select the catalog row, then click
   **Stream catalog**.
4. The status bar at the bottom shows cache stats (FASTA, GenBank, encoded
   files cached).

---

### 3.4 Train (tab: Train)

The full-featured job submission form supporting all 7 job types.

**Form fields:**
| Field          | Description                                          |
|----------------|------------------------------------------------------|
| Job kind       | Dropdown: train_one, stream, generate_plasmid, generate_protein, validate_plasmid, pretrain, design_loop |
| Config         | Path to the YAML config file                         |
| Accession/seed | Accession ID or raw sequence (for train_one, validate, design_loop) |
| Catalog path   | Path to accession catalog (for stream, pretrain)     |
| Output path    | Output file path (for generate jobs)                 |
| Tokenizer      | base, codon, or aa                                   |
| Length          | Sequence length for generation jobs                  |
| Candidates     | Number of candidates to generate                     |
| Temperature    | Sampling temperature / latent scale                  |

**Buttons:**
| Button          | Action                                              |
|-----------------|-----------------------------------------------------|
| Submit Job      | Build a JobSpec from the form and submit it          |
| Reset Form      | Clear all fields to defaults                        |
| Load from draft | Populate the form from the saved draft state        |

**Live Monitor section (below the form):**
- Progress bar and status line for the active job
- Loss curve sparkline with current/min/max/rolling stats
- Running/Recent Jobs table

**Panel-local keyboard shortcuts:**
| Key | Action                    |
|-----|---------------------------|
| `r` | Rerun the selected job    |
| `c` | Cancel the selected job   |
| `l` | Open the Logs drawer      |

**Workflow -- train a single accession:**
1. Set **Job kind** to "Train single accession".
2. Type an accession (e.g. `NC_000913`) in the Accession field.
3. Adjust tokenizer/config if needed.
4. Click **Submit Job**.
5. Watch the progress bar and loss sparkline update in real time.

**Workflow -- generate sequences:**
1. Set **Job kind** to "Generate plasmid" or "Generate protein".
2. Set output path, length, number of candidates, and temperature.
3. Click **Submit Job**.

**Workflow -- load a cloned draft:**
1. Go to the Jobs or History panel and click **Clone to draft** on a past job.
2. Return to the Train panel.
3. Click **Load from draft** to populate all form fields from that job's
   original parameters.
4. Adjust as needed and **Submit Job**.

---

### 3.5 Generate (tab: Generate)

A focused panel for plasmid and protein sequence generation.

**Form fields:**
| Field        | Description                                |
|--------------|--------------------------------------------|
| Kind         | Plasmid or Protein                         |
| Output path  | Where to write the FASTA file              |
| Length        | Sequence length                            |
| Candidates   | Number of candidate sequences to generate  |
| Top-K        | Top-K sampling parameter                   |
| Seed         | Random seed (optional, for reproducibility)|
| Temperature  | Sampling temperature / latent scale        |

**Buttons:**
| Button       | Action                                      |
|--------------|---------------------------------------------|
| Generate     | Submit the generation job                   |
| Reset        | Clear form to defaults                      |
| Open outputs | Switch to the Artifacts panel               |

**Below the form:**
- Generation progress bar for the active generation job
- Recent Generation Jobs table (filtered to generation kinds only)

**Workflow:**
1. Pick Plasmid or Protein.
2. Set length, number of candidates, and temperature.
3. Click **Generate**.
4. When done, click **Open outputs** to view the generated FASTA in the
   Artifacts panel.

---

### 3.6 History (tab: History)

Browse all past runs from disk manifests and persisted job records.

**What you see:**
- Table of all runs with status, run ID, kind, title, artifact count, and
  last-updated timestamp
- Run Detail section below the table

**Buttons:**
| Button          | Action                                            |
|-----------------|---------------------------------------------------|
| Refresh         | Re-scan the `runs/` directory for manifests       |
| Rerun selected  | Re-submit the selected run with its original params |
| Clone to draft  | Copy the run's parameters into the draft form     |
| View manifest   | Open the run's manifest in the Artifact drawer    |
| View artifacts  | Switch to the Artifacts panel                     |

**Panel-local keyboard shortcuts:**
| Key | Action                             |
|-----|------------------------------------|
| `r` | Rerun selected job                 |
| `c` | Clone selected job to draft        |
| `m` | Open manifest for selected job     |

**Workflow:**
1. Click a row to see full run detail (parents, children, artifacts, errors).
2. Click **Rerun selected** to re-launch an identical job.
3. Click **Clone to draft** then switch to the Train panel and click
   **Load from draft** to edit parameters before resubmitting.

---

### 3.7 Outputs & Artifacts (tab: Outputs)

Browse all output files produced by jobs, with inline preview.

**What you see:**
- Role filter dropdown: All, Checkpoints, Outputs, Logs, Manifests, Configs,
  Scorecards, Summaries
- Table with Run ID, Role, Filename, Size, Modified date, and Exists flag
- Preview section showing file contents for text-based formats (JSON, FASTA,
  CSV, YAML, logs)

**Buttons:**
| Button   | Action                              |
|----------|-------------------------------------|
| Refresh  | Re-scan artifacts from all runs     |

**Workflow:**
1. Use the role filter dropdown to narrow to the type you care about
   (e.g. "Checkpoints" to find model weights).
2. Click a row to preview the file contents below.
3. Selecting a file also opens it in the Artifact detail drawer on the right.

---

### 3.8 Models (tab: Models)

Browse model checkpoints and weight files on disk.

**What you see:**
- Table of discovered model files (`.pt`, `.pth`, `.ckpt`, `.npz`,
  `.safetensors`, `.bin`, `.json`) from `model/`, `model/checkpoints/`,
  and `model/pretrain/`
- Details section with file metadata and JSON key preview for `.json` files

**Buttons:**
| Button            | Action                                    |
|-------------------|-------------------------------------------|
| Refresh           | Re-scan model directories                 |
| Open in artifacts | Switch to the Artifacts panel             |

**Workflow:**
1. Click **Refresh** to discover model files.
2. Click a row to see path, size, modification date, and (for JSON files)
   a key-value preview of the contents.

---

### 3.9 Jobs (tab: Jobs)

The active job queue with full job detail and management controls.

**What you see:**
- Action button row
- Job table: Status, Job ID, Kind, Title, Stage, Progress, Message
- Selected Job Detail section with full metadata

**Buttons:**
| Button          | Action                                           |
|-----------------|--------------------------------------------------|
| Cancel selected | Send a cancellation signal to the selected job   |
| Rerun selected  | Re-submit the selected job with original params  |
| Clone to draft  | Copy the job's params into the draft form        |
| View logs       | Open the Logs detail drawer                      |
| View artifacts  | Switch to the Artifacts panel                    |

**Panel-local keyboard shortcuts:**
| Key           | Action                                 |
|---------------|----------------------------------------|
| `j` / `Down`  | Select the next job in the table      |
| `k` / `Up`    | Select the previous job               |
| `Enter`       | Switch to the Train panel             |
| `r`           | Rerun selected job                    |
| `c`           | Cancel selected job                   |

**Job Detail shows:**
- Job ID, Kind, Title, Status, Stage, Reason
- Step/Total steps, Epoch, Progress percentage
- Latest loss, Rolling loss
- Created/Updated timestamps
- Artifact count, last error, last warning

**Workflow:**
1. Use `j`/`k` to navigate between jobs.
2. Click a job row to see its full detail below.
3. Click **Cancel selected** to stop a running job.
4. Click **Rerun selected** to relaunch with identical settings.
5. Click **Clone to draft**, then go to Train > **Load from draft** to tweak
   parameters before re-submitting.

---

### 3.10 Metrics (tab: Metrics)

Loss curve visualization and comparison across jobs.

**What you see:**
- **Selected Job Loss Curve** -- full-width sparkline widget
- **Stat cards** -- Current, Min, Max, Rolling(10), and Samples
- **All Jobs Loss Summary** table -- one row per job with loss stats and
  inline text sparkline
- **Per-Job Sparklines** -- compact text-based sparklines with `>` marker
  on the selected job

**Workflow:**
1. While a job is training, switch here to watch the loss curve in real time.
2. Click a row in the summary table to switch the primary sparkline to that
   job's data.
3. Compare convergence across multiple runs using the per-job sparkline
   section.

---

### 3.11 Pipeline (tab: Pipeline)

Stage timeline visualization for the active or selected job.

**What you see:**
- Active job header with current stage
- Timeline table: Timestamp, Stage (with icon), Message
- Stage Sequence: a visual chain like `> -> E -> T -> T -> M` showing the
  progression of stages

**Stage icons:**
| Icon | Stage        |
|------|--------------|
| `>`  | start        |
| `E`  | encode       |
| `T`  | train        |
| `S`  | stream_step  |
| `P`  | pretrain     |
| `G`  | generate     |
| `V`  | validate     |
| `D`  | design_loop  |
| `M`  | manifest     |
| `!`  | error        |
| `W`  | warn/warning |
| `L`  | log          |
| `.`  | progress     |

**Workflow:**
1. Start a job and switch to the Pipeline panel.
2. Watch stages appear in the timeline as the job progresses through encode,
   train, manifest, etc.
3. The stage sequence at the bottom gives a compact view of the full pipeline
   flow.

---

### 3.12 Troubleshoot (tab: Trouble)

Inspect failed jobs, view tracebacks, and relaunch.

**What you see:**
- Failed Jobs table: Job ID, Kind, Stage, Error message
- Failure Detail: full metadata about the selected failure
- Traceback Preview: first 20 lines of the traceback file

**Buttons:**
| Button         | Action                                          |
|----------------|-------------------------------------------------|
| Open traceback | Open the Traceback detail drawer (right side)   |
| Open logs      | Open the Logs detail drawer                     |
| Rerun          | Re-submit the failed job with original params   |
| Clone to draft | Copy the job's params for editing               |
| View artifacts | Switch to the Artifacts panel                   |

**Workflow:**
1. If a job fails, navigate here (or press `Ctrl+T` for the traceback drawer).
2. Click the failed job row to see the full error and traceback preview.
3. Click **Open traceback** for the full traceback in the side drawer.
4. Click **Rerun** to retry, or **Clone to draft** to adjust params first.

---

### 3.13 Events (tab: Events)

Raw event log showing all TUI and job events.

**What you see:**
- Event table: Timestamp, Kind, Message
- Event count and current display limit

**Buttons:**
| Button     | Action                                  |
|------------|-----------------------------------------|
| Refresh    | Reload events from the event log file   |
| Show more  | Increase the display limit by 40 (max 200) |

**Workflow:**
1. Check here for a chronological record of everything that happened:
   panel switches, job submissions, stage updates, errors, diagnostics
   snapshots, etc.
2. Click **Show more** to page back through older events.

---

### 3.14 Diagnostics (tab: Diag)

Environment health checks and system information.

**What you see:**
- **Environment cards** -- Python version, Platform, CPU count, Memory
  (available / total)
- **Health Checks table** -- config file exists, directories exist, disk free
  space, CUDA availability
- **Disk Usage table** -- size and file counts for `runs/`, `model/`,
  `state/tui/`, `config/`
- **Packages table** -- installed versions of torch, numpy, textual, psutil,
  biopython, h5py, safetensors, pyyaml

**Buttons:**
| Button   | Action                          |
|----------|---------------------------------|
| Refresh  | Re-capture all diagnostics      |

**Workflow:**
1. Check here before starting work to verify GPU availability, sufficient disk
   space, and required packages.
2. Click **Refresh** after installing a package or freeing disk space.

---

### 3.15 Settings (tab: Settings)

Application settings and state management.

**What you see:**
- **Paths** -- default config path input, state directory display
- **Display** -- events panel limit, history panel limit
- **Toggles** -- auto-refresh overview, persist session on exit
- **Cache & State** -- sizes of event log, launcher history, and session files

**Buttons:**
| Button                | Action                                       |
|-----------------------|----------------------------------------------|
| Clear event log       | Truncate the `events.jsonl` file to empty    |
| Clear launcher history| Reset the launcher command history            |
| Reset session         | Delete `session.json` (takes effect on restart)|

**Workflow:**
1. If the event log grows large, click **Clear event log** to reclaim space.
2. If the launcher history feels stale, clear it so command ranking resets.
3. **Reset session** returns the TUI to factory-default panel and selection
   state on next launch.

---

## 4. The Command Palette

Press `Ctrl+P` from anywhere to open the command palette.

**How it works:**
1. A modal appears with a filter input and a ranked list of commands.
2. Type to filter -- it searches command labels, categories, IDs, and keywords.
3. Press `Enter` to execute the highlighted command, or click a row.
4. Press `Escape` to close without running anything.

**Available commands (30 total):**

**Panel jumps (15):**
Jump to any panel by name -- e.g. type "train" and press Enter.

**Job commands:**
| Command         | Action                                |
|-----------------|---------------------------------------|
| Job: Start      | Submit the current draft job spec     |
| Job: Stop Active| Cancel the running job                |
| Job: Rerun Last | Rerun the most recent job             |

**Inspect commands:**
| Command               | Action                                |
|-----------------------|---------------------------------------|
| Inspect: Active Job   | Focus on the running job              |
| Inspect: Open Artifact| Open the latest artifact              |
| Inspect: Show Logs    | Open the logs drawer                  |

**View commands:**
| Command                 | Action                                |
|-------------------------|---------------------------------------|
| View: Logs Drawer       | Toggle the logs side panel            |
| View: Diagnostics Drawer| Toggle the diagnostics side panel     |
| View: Resources Drawer  | Toggle the resources side panel       |
| View: Traceback Drawer  | Toggle the traceback side panel       |
| View: Artifact Details  | Toggle the artifact detail panel      |
| View: Reset Layout      | Close all drawers, go to Overview     |

**Troubleshoot commands:**
| Command                         | Action                          |
|---------------------------------|---------------------------------|
| Troubleshoot: Open Failed Job   | Focus the last failed job       |
| Troubleshoot: Open Traceback    | Show traceback for failed job   |
| Troubleshoot: Reopen Last Failed| Return to the last failure      |

**Smart ranking:**
- Commands are ranked by context. If a job is running, "Job: Stop Active"
  ranks higher. If a job has failed, troubleshoot commands float to the top.
- Commands you use frequently get a history boost.
- Some commands are disabled when not applicable (e.g. "Job: Stop Active" is
  disabled when nothing is running).

---

## 5. Detail Drawers

Detail drawers open on the right side of the screen and provide focused views.

| Drawer        | Shortcut  | Content                                    |
|---------------|-----------|--------------------------------------------|
| Logs          | `Ctrl+L`  | Tails the log file for the selected job    |
| Diagnostics   | `Ctrl+D`  | Live system info (Python, platform, memory, GPU) |
| Traceback     | `Ctrl+T`  | Shows the traceback file for the last failed job |
| Resources     | Launcher  | CPU/GPU/memory resource snapshot           |
| Artifact      | Launcher  | Details for the selected artifact file     |

- Press `Escape` to close the active drawer.
- Only one drawer is visible at a time. Opening a new one replaces the current.
- Logs and Diagnostics drawers auto-refresh on a timer.

---

## 6. Common Workflows

### Train on a single accession end-to-end

```
1. perceptrome tui
2. Press ] to navigate to the Config panel
3. Verify config validates (all checks PASS)
4. Press ] to navigate to the Train panel
5. Set Job kind = "Train single accession"
6. Type accession: NC_000913
7. Click "Submit Job"
8. Watch the progress bar and loss sparkline
9. Press Ctrl+L to open the Logs drawer on the right
10. When complete, press ] ] to go to Artifacts and browse outputs
```

### Stream over a catalog

```
1. Navigate to the Catalogs panel (press [ / ] or Ctrl+P > "data")
2. Click a catalog row (e.g. plasmid_accessions.txt)
3. Verify the preview looks correct
4. Click "Stream catalog"
5. Switch to the Metrics panel to watch loss convergence
6. Switch to the Pipeline panel to see stage progression
```

### Generate plasmid sequences

```
1. Navigate to the Generate panel
2. Set Kind = Plasmid, Length = 512, Candidates = 10
3. Adjust Temperature if desired
4. Click "Generate"
5. When complete, click "Open outputs" to preview the FASTA
```

### Investigate a failed job

```
1. Press Ctrl+T to open the Traceback drawer
   -- or navigate to the Troubleshoot panel
2. Read the traceback preview
3. Check Failure Detail for the stage and error message
4. Click "Clone to draft" to copy the job's params
5. Switch to Train panel > "Load from draft"
6. Adjust params and resubmit
```

### Compare multiple runs

```
1. Run several training jobs (different configs, accessions, etc.)
2. Navigate to the Metrics panel
3. The "All Jobs Loss Summary" table shows one row per job
4. Click different rows to switch the primary sparkline
5. The "Per-Job Sparklines" section shows all runs side by side
```

### Resume from a previous session

```
1. Quit the TUI (press q)
2. Relaunch: perceptrome tui
3. The TUI restores:
   - Last active panel
   - Selected job
   - Open detail drawer
   - Draft job spec (form state)
4. Previously completed jobs appear in the Jobs and History panels
5. Jobs that were active when you quit show as STALLED
```

---

## 7. State and Persistence

All TUI state is stored in `state/tui/` (or the path set by
`PERCEPTROME_TUI_STATE_ROOT`).

| File                   | Contents                                  |
|------------------------|-------------------------------------------|
| `session.json`         | Active panel, selected job, drawer state, draft spec, scroll positions |
| `jobs.json`            | Job registry (status, metrics, artifacts) |
| `events.jsonl`         | Append-only event log                     |
| `launcher_history.json`| Command palette usage history             |
| `tui_jobs.json`        | JobManager's own job snapshot             |

All writes are atomic (write to temp file, then rename) to prevent corruption.

To fully reset TUI state:
```bash
rm -rf state/tui/
```

---

## 8. Job Types Reference

| Kind               | Required Fields        | Description                          |
|--------------------|------------------------|--------------------------------------|
| `train_one`        | accession              | Train VAE on a single sequence       |
| `stream`           | catalog                | Stream train over a catalog of accessions |
| `generate_plasmid` | output, length         | Sample plasmid sequences from the model |
| `generate_protein` | output, length         | Sample protein sequences from the model |
| `validate_plasmid` | accession              | Score generated sequences vs reference |
| `pretrain`         | dataset                | MLM/SME pretraining on a dataset     |
| `design_loop`      | seed_sequence          | Iterative generation with feedback   |

---

## 9. Tips

- **Ctrl+P is your friend.** The command palette has every action and adapts
  its ranking to what you are doing right now.
- **Clone before rerun.** If a job failed, clone to draft, fix the params,
  then submit -- rather than blindly rerunning.
- **Use the Diagnostics panel** before starting GPU-heavy work to verify CUDA
  is available and disk space is sufficient.
- **Watch the Pipeline panel** during long stream jobs to see which accession
  the encoder is working on.
- **Filter artifacts by role** in the Outputs panel to quickly find checkpoints
  vs logs vs manifests.
- **The TUI remembers everything.** Your last panel, selected job, draft form
  state, and drawer position are all restored on relaunch.
