# Concrete Model Stack for Genome-Driven Model Generation

This document proposes a practical, staged model stack for Perceptrome using genomes fetched from NCBI.

## Goal and assumptions

- Input: fetched genomes, then windowed/encoded sequences.
- Primary near-term objective: robust reconstruction + anomaly scoring.
- Secondary objective: controlled sequence generation from a latent space.
- Constraint: keep compatibility with existing training and checkpoint flow.

## Stage 1 (now): Strong baseline with current architecture

Use what is already in the codebase as the production baseline:

- **Tokenizer**
  - `base` for DNA-level experiments.
  - `aa` for protein-level experiments when ORF/proteome encoding is available.
- **Model**
  - `mlp` VAE (`PlasmidVAE`) for fast, low-memory iteration.
- **Loss**
  - `mse` for base tokenization.
  - `ce` for AA tokenization.
- **Training behavior**
  - Keep KL warmup + gradient clipping enabled.
  - For AA mode, keep random masking/span masking as denoising signal.

Why this stage:

- Fastest way to get a trustworthy baseline and a stable anomaly signal.
- Lowest risk because it matches the existing checkpoint and CLI/training flow.

## Stage 2 (next): Transformer VAE as the default sequence model

Promote the existing transformer path to the default for long-range sequence context.

- **Tokenizer**
  - Keep `aa` as preferred default when possible (strong semantic compression).
  - Keep `base` as fallback for raw DNA.
- **Model**
  - `transformer` VAE (`TransformerVAE`).
- **Starting hyperparameters**
  - `d_model=256`
  - `nhead=8`
  - `layers=4`
  - `dropout=0.1`
- **Loss**
  - `ce` for categorical reconstruction when tokenization is categorical.
- **Monitoring**
  - Track: reconstruction loss, KL, per-window error distribution, training throughput.

Why this stage:

- Captures long-range dependencies better than MLP baselines.
- Already implemented, so migration effort is mostly config/training policy.

## Stage 3: Retrieval-augmented latent conditioning (metadata-aware)

Extend the VAE with conditioning vectors (host, taxonomy, molecule type, source metadata).

- Add a metadata encoder (small MLP or embedding table).
- Concatenate or cross-attend metadata embedding with latent `z` before decoding.
- Keep unconditioned mode for unknown metadata.

Why this stage:

- Improves controllability for generation and better partitions latent space.

## Stage 4: Graph-enhanced branch (optional but high-value)

Add a parallel GNN branch when interaction/pathway edges are available.

- Build per-sample or per-taxon graphs from known interaction resources.
- Fuse graph embedding with sequence latent embedding (late fusion first).
- Use multi-task heads (reconstruction + trait/function proxy objectives).

Why this stage:

- Introduces biologically informed relational context absent from sequence-only models.

## Stage 5: Generative upgrade path

After Stage 2 is stable, move to stronger generation quality:

1. **Latent VAE sampling** (already available)
2. **Autoregressive transformer decoder** for sequence quality and controllability
3. **Discrete diffusion** for best fidelity/diversity tradeoff (higher complexity)

## Recommended default stack (what to run first)

For immediate execution in this repository:

- **Primary stack**: `aa` tokenizer + `transformer` VAE + `ce` loss.
- **Baseline control**: `base` tokenizer + `mlp` VAE + `mse` loss.
- **Windowing**: start with moderate windows and scale upward only after throughput is validated.
- **Selection criterion**: choose the model with best validation reconstruction calibration and most stable per-window error ranking.

## Minimal experiment matrix

Run the following first-pass matrix on the same fetched genome set:

1. `mlp/base/mse`
2. `mlp/aa/ce`
3. `transformer/base/ce` (or mse if needed for compatibility)
4. `transformer/aa/ce`

Compare:

- Convergence speed (steps to stable loss)
- Final reconstruction quality
- KL stability (avoid collapse)
- Practical throughput (windows/sec)
- Anomaly ranking consistency across held-out accessions

## Definition of done for stack selection

Adopt the Stage 2 stack as default when:

- It beats Stage 1 on held-out reconstruction and anomaly ranking consistency,
- It trains without frequent divergence across multiple accession batches,
- Throughput remains acceptable for stream training.
