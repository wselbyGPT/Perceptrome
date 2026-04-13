#!/usr/bin/env python3
import argparse
import importlib
from typing import Any


def _lazy_cmd(func_name: str):
    def _run(args):
        mod = importlib.import_module("perceptrome.cli.commands")
        return getattr(mod, func_name)(args)

    return _run


def _run_tui(_args):
    from perceptrome.cli.common import resolve_tui_state_root, write_tui_startup_context
    import os

    state_root = resolve_tui_state_root(_args.config)
    os.environ["PERCEPTROME_TUI_STATE_ROOT"] = state_root
    write_tui_startup_context(
        config_path=str(_args.config),
        run_id=getattr(_args, "run_id", None),
        job_id=getattr(_args, "job_id", None),
        panel=getattr(_args, "panel", None),
        detail_surface=getattr(_args, "detail_surface", None),
    )
    mod = importlib.import_module("perceptrome.tui.app")
    return mod.main()

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="perceptrome", description="Perceptrome streaming VAE trainer + scope.")
    p.add_argument("--config", default="config/stream_config.yaml", help="YAML config path (default: config/stream_config.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init"); s.set_defaults(func=_lazy_cmd("cmd_init"))
    s = sub.add_parser("catalog-show"); s.add_argument("path"); s.set_defaults(func=_lazy_cmd("cmd_catalog_show"))
    s = sub.add_parser("catalog-generate")
    s.add_argument("--schema", required=True, help="Catalog schema YAML/JSON path")
    s.add_argument("--output", required=True, help="Output catalog path")
    s.set_defaults(func=_lazy_cmd("cmd_catalog_generate"))

    s = sub.add_parser("virus-catalog", help="Virus catalog workflow commands")
    virus_sub = s.add_subparsers(dest="virus_catalog_command", required=True)

    vb = virus_sub.add_parser("build", help="Build catalog.txt from NCBI datasets virus summary")
    vb.add_argument("--taxon", default=None, help="Taxon query (mutually exclusive with accession/inputfile)")
    vb.add_argument("--accession", action="append", default=None, help="Explicit accession (repeatable)")
    vb.add_argument("--inputfile", default=None, help="Input catalog text file containing accessions")
    vb.add_argument("--include", action="append", default=None, help="datasets include selection (repeatable)")
    vb.add_argument("--filter", action="append", default=None, help="Raw datasets filter arg to append (repeatable)")
    vb.add_argument("--datasets-bin", default=None, help="Path to datasets executable")
    vb.add_argument("--output", default="catalog.txt", help="Output catalog path (default: catalog.txt)")
    vb.add_argument("--snapshot", action="store_true", help="Create snapshot bundle immediately after build")
    vb.add_argument("--snapshot-dir", default=None, help="Snapshot root dir (default: <catalog_dir>/snapshots)")
    vb.add_argument("--snapshot-metadata", action="append", default=None, help="Optional metadata file to include in snapshot")
    vb.set_defaults(func=_lazy_cmd("cmd_virus_catalog_build"))

    vi = virus_sub.add_parser("inspect", help="Inspect virus catalog manifest + catalog")
    vi.add_argument("--manifest", required=True, help="Path to catalog.manifest.json")
    vi.add_argument("--catalog", default=None, help="Override catalog path (default from manifest)")
    vi.set_defaults(func=_lazy_cmd("cmd_virus_catalog_inspect"))

    vr = virus_sub.add_parser("rebuild", help="Re-run stored query spec from manifest")
    vr.add_argument("--manifest", required=True, help="Path to catalog.manifest.json")
    vr.add_argument("--output", default=None, help="Optional catalog output path (default from manifest)")
    vr.add_argument("--datasets-bin", default=None, help="Path to datasets executable")
    vr.set_defaults(func=_lazy_cmd("cmd_virus_catalog_rebuild"))

    vs = virus_sub.add_parser("snapshot", help="Create immutable snapshot bundle")
    vs.add_argument("--catalog", required=True, help="Catalog path")
    vs.add_argument("--manifest", required=True, help="Manifest path")
    vs.add_argument("--metadata", action="append", default=None, help="Optional metadata file to include (repeatable)")
    vs.add_argument("--snapshot-dir", default=None, help="Snapshot root dir")
    vs.set_defaults(func=_lazy_cmd("cmd_virus_catalog_snapshot"))

    s = sub.add_parser("uniprot-count")
    s.add_argument("--query", help="UniProt query expression")
    s.add_argument("--mode", choices=["all", "reviewed", "unreviewed"], default="all", help="Record subset filter (default: all)")
    s.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    s.set_defaults(func=_lazy_cmd("cmd_uniprot_count"))

    s = sub.add_parser("uniprot-fetch")
    s.add_argument("--query", required=True, help="UniProt query expression")
    s.add_argument("--include-isoforms", action="store_true", help="Include isoform records in FASTA output")
    s.add_argument("--output-dir", default=None, help="Output directory for shard FASTA files (default: <cache_fasta_dir>/uniprot)")
    s.add_argument("--prefix", default="uniprot", help="Output shard filename prefix (default: uniprot)")
    s.add_argument("--records-per-shard", type=int, default=None, help="Max records per output shard (default: uniprot.records_per_shard from config; 50000 in config/stream_config.yaml)")
    s.add_argument("--gzip-output", action="store_true", help="Write .gz compressed shard outputs")
    s.add_argument("--resume", action="store_true", help="Skip work if output shards already exist for this query")
    s.add_argument("--count-only", action="store_true", help="Query count only; do not download FASTA shards")
    s.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    s.set_defaults(func=_lazy_cmd("cmd_uniprot_fetch"))

    s = sub.add_parser("virus-fetch", help="Download and stage NCBI virus dataset package(s)")
    s.add_argument("--catalog", default=None, help="Catalog text file of accessions")
    s.add_argument("--manifest", default=None, help="Virus catalog manifest JSON")
    s.add_argument("--taxon", default=None, help="Taxon query for direct datasets fetch")
    s.add_argument("--include", action="append", default=None, help="datasets include selection (repeatable)")
    s.add_argument("--filter", action="append", default=None, help="Raw datasets filter arg to append (repeatable)")
    s.add_argument("--datasets-bin", default=None, help="Path to datasets executable")
    s.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    s.set_defaults(func=_lazy_cmd("cmd_virus_fetch"))

    s = sub.add_parser("split-create")
    s.add_argument("--catalog", required=True, help="Catalog file used to build splits")
    s.add_argument("--name", default="default", help="Split set name")
    s.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    s.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    s.add_argument("--seed", type=int, default=1337, help="Shuffle seed for deterministic split generation")
    s.add_argument("--out", default=None, help="Optional output json path (default: <state_dir>/splits/<name>.json)")
    s.set_defaults(func=_lazy_cmd("cmd_split_create"))

    s = sub.add_parser("split-show")
    s.add_argument("--name", default="default", help="Split set name")
    s.add_argument("--path", default=None, help="Optional explicit split json path")
    s.set_defaults(func=_lazy_cmd("cmd_split_show"))

    s = sub.add_parser("virus-split-create", help="Create deterministic virus train/val/test splits")
    s.add_argument("--catalog", required=True, help="Catalog file used to build splits")
    s.add_argument("--name", default="default", help="Split set name")
    s.add_argument(
        "--strategy",
        choices=["random", "taxon", "host", "completeness-reference", "family-heldout", "date-based"],
        default="random",
        help="Split strategy",
    )
    s.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    s.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    s.add_argument("--seed", type=int, default=1337, help="Seed for deterministic split generation")
    s.add_argument("--fetch-manifest", action="append", default=None, help="virus-fetch manifest JSON (repeatable)")
    s.add_argument("--snapshot", action="append", default=None, help="Catalog snapshot dir or snapshot.bundle.json (repeatable)")
    s.add_argument("--out", default=None, help="Optional output JSON path (default: <state_dir>/splits/<name>.json)")
    s.set_defaults(func=_lazy_cmd("cmd_virus_split_create"))

    s = sub.add_parser("virus-split-show", help="Show virus split metadata and partition preview")
    s.add_argument("--name", default="default", help="Split set name")
    s.add_argument("--path", default=None, help="Optional explicit split JSON path")
    s.set_defaults(func=_lazy_cmd("cmd_virus_split_show"))

    s = sub.add_parser("fetch-one")
    s.add_argument("accession"); s.add_argument("--force", action="store_true")
    s.add_argument("--source", choices=["fasta","genbank"], default=None, help="Fetch record source (default: fasta)")
    s.set_defaults(func=_lazy_cmd("cmd_fetch_one"))

    s = sub.add_parser("tensorboard")
    s.add_argument("--logdir", default=None, help="TensorBoard log directory (default: <logs_dir>/tensorboard)")
    s.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    s.add_argument("--port", type=int, default=6006, help="Bind port (default: 6006)")
    s.add_argument("--reload-interval", type=float, default=5.0, help="Reload interval in seconds (default: 5.0)")
    s.add_argument("--path-prefix", default=None, help="Optional path prefix for reverse-proxy setups")
    s.add_argument("--dry-run", action="store_true", help="Print launch command and exit")
    s.set_defaults(func=_lazy_cmd("cmd_tensorboard"))

    def add_tok_args(sp):
        sp.add_argument("--tokenizer", choices=["base","codon","aa"], default=None, help="Override tokenizer (default from config)")
        sp.add_argument(
            "--aa-profile",
            choices=["conservative", "balanced", "exploratory"],
            default=None,
            help="AA mode preset for filter/sampling knobs (CLI flags still override).",
        )
        sp.add_argument("--frame-offset", type=int, choices=[0,1,2], default=None, help="Codon frame offset (default from config)")
        sp.add_argument("--min-orf-aa", type=int, default=None, help="AA tokenizer: minimum ORF length in amino acids (default from config)")
        sp.add_argument("--min-protein-aa", type=int, default=None, help="Alias for --min-orf-aa (aa/genbank)")
        sp.add_argument("--max-protein-aa", type=int, default=None, help="AA tokenizer: reject proteins longer than this (aa/genbank)")
        sp.add_argument("--strict-cds", action="store_true", help="AA+genbank: use GenBank CDS only (no ORF fallback).")
        sp.add_argument("--require-translation", action="store_true", help="AA+genbank: require /translation qualifier (skip loc+ORIGIN translation).")
        sp.add_argument("--x-free", action="store_true", help="AA+genbank: drop proteins containing X or stop markers.")
        sp.add_argument("--require-start-m", action="store_true", help="AA+genbank: require protein to start with M.")
        sp.add_argument("--reject-partial-cds", action="store_true", help="AA+genbank: reject partial CDS locations with < or >.")
        sp.add_argument("--source", choices=["fasta","genbank"], default=None, help="Sequence record source. Default: fasta for base/codon, genbank for aa.")
        # Proteome (aa) sampling / filters
        sp.add_argument("--protein-len-min", type=int, default=None, help="AA tokenizer: minimum protein length to include (overrides config/curriculum)")
        sp.add_argument("--protein-len-max", type=int, default=None, help="AA tokenizer: maximum protein length to include (overrides config/curriculum)")
        sp.add_argument("--max-windows-per-protein", type=int, default=None, help="AA tokenizer: cap windows sampled per CDS/protein (balances long proteins)")
        sp.add_argument("--translation-only", action="store_true", default=None, help="AA tokenizer+genbank: use only /translation-provided proteins")
        sp.add_argument("--allow-translated", action="store_true", default=None, help="AA tokenizer+genbank: allow translating from CDS when /translation missing")
        sp.add_argument("--no-curriculum", action="store_true", help="Disable proteome curriculum for this run")

    def add_loss_args(sp):
        sp.add_argument(
            "--loss-type",
            choices=["mse", "ce"],
            default=None,
            help="Override reconstruction loss. Default: ce for aa, mse for base/codon.",
        )
        sp.add_argument(
            "--mask-prob",
            type=float,
            default=None,
            help="Denoising mask probability (AA tokenizer only). Default: 0.05 for aa, 0 for others.",
        )
        sp.add_argument("--span-mask-prob", type=float, default=None, help="AA tokenizer: probability to apply a contiguous span mask per sequence")
        sp.add_argument("--span-mask-len", type=int, default=None, help="AA tokenizer: length (aa) of the contiguous span mask")

    def add_tensorboard_args(sp):
        sp.add_argument("--tb-run-id", default=None, help="TensorBoard run id (default: accession or config)")
        sp.add_argument("--tb-log-every", type=int, default=None, help="TensorBoard logging interval in steps (default from config)")


    def add_ast_template_args(sp):
        sp.add_argument("--ast-template-artifact", default=None, help="Optional Bio-AST template or canonical AST artifact for round-trip validation")
        sp.add_argument("--ast-template-mode", choices=["off", "rescore", "reject"], default="off", help="Template policy: ignore, rescore, or reject structurally mismatched candidates")
        sp.add_argument("--ast-template-span-tolerance", type=int, default=0, help="Allowed absolute span delta when comparing rebuilt Bio-AST nodes")
        sp.add_argument("--ast-template-min-score", type=float, default=1.0, help="Minimum template match score required for acceptance")
        sp.add_argument("--ast-template-max-mismatches", type=int, default=0, help="Maximum total node/order/edge mismatches allowed")
        sp.add_argument("--ast-template-include-semantic-edges", action="store_true", help="Require semantic edges from the template during comparison")

    def add_model_args(sp):
        sp.add_argument(
            "--model-type",
            choices=["mlp", "transformer", "ssm"],
            default=None,
            help="Override model architecture for this command (default from config).",
        )

    s = sub.add_parser("encode-one")
    s.add_argument("accession")
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    add_tok_args(s)
    s.set_defaults(func=_lazy_cmd("cmd_encode_one"))

    s = sub.add_parser("train-one")
    s.add_argument("accession")
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--reencode", action="store_true")
    s.add_argument("--experiment-id", default=None, help="Optional run ID for manifest output (default: auto-generated)")
    add_tok_args(s)
    add_loss_args(s)
    add_tensorboard_args(s)
    add_model_args(s)
    s.set_defaults(func=_lazy_cmd("cmd_train_one"))

    s = sub.add_parser("scope-one")
    s.add_argument("accession")
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--fps", type=float, default=12.0)
    s.add_argument("--reencode", action="store_true")
    s.add_argument("--no-curses", action="store_true", help="Do not launch curses UI; emit structured summary artifact only")
    s.add_argument("--summary-artifact", default=None, help="Optional path for scope summary JSON artifact")
    add_tok_args(s)
    s.add_argument("--loss-type", choices=["mse", "ce"], default=None, help="Override loss used for error metric (default: ce for aa, mse for base/codon)")
    add_model_args(s)
    s.set_defaults(func=_lazy_cmd("cmd_scope_one"))

    s = sub.add_parser("scope-stream")
    s.add_argument("accession")
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--fps", type=float, default=12.0)
    s.add_argument("--update-every", type=int, default=5)
    s.add_argument("--reencode", action="store_true")
    s.add_argument("--no-curses", action="store_true", help="Do not launch curses UI; emit structured summary artifact only")
    s.add_argument("--summary-artifact", default=None, help="Optional path for scope summary JSON artifact")
    add_tok_args(s)
    add_loss_args(s)
    add_tensorboard_args(s)
    add_model_args(s)
    s.set_defaults(func=_lazy_cmd("cmd_scope_stream"))

    s = sub.add_parser("stream")
    s.add_argument("--catalog", required=True)
    s.add_argument("--max-epochs", type=int, default=None)
    s.add_argument("--steps-per-plasmid", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--delete-cache", action="store_true")
    s.add_argument("--experiment-id", default=None, help="Optional run ID for manifest output (default: auto-generated)")
    add_tok_args(s)
    add_loss_args(s)
    add_tensorboard_args(s)
    add_model_args(s)
    s.set_defaults(func=_lazy_cmd("cmd_stream"))

    s = sub.add_parser("generate-plasmid")
    s.add_argument("--length-bp", type=int, default=10000)
    s.add_argument("--num-windows", type=int, default=None)
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--name", default="perceptrome_plasmid_1")
    s.add_argument("--output", default="generated/novel_plasmid.fasta")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--latent-scale", type=float, default=1.0)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--gc-bias", type=float, default=1.0)
    s.add_argument("--num-candidates", type=int, default=1)
    s.add_argument("--top-k", type=int, default=1)
    s.add_argument("--target-gc", type=float, default=0.5)
    s.add_argument("--max-homopolymer", type=int, default=None)
    s.add_argument("--summary-path", default=None, help="Optional JSON path for candidate summary (CSV written alongside)")
    s.add_argument("--scorecard-path", default=None, help="Optional JSON path for scorecard output artifact")
    s.add_argument("--scorecard-format", choices=["json"], default="json", help="Scorecard serialization format (default: json)")
    s.add_argument("--scorecard-risk-profile", default=None, help="Optional risk threshold preset (future extensible)")
    s.add_argument("--top-k-output", default=None, help="Optional FASTA path for ranked top-k candidates")
    s.add_argument("--roundtrip-score", action="store_true", help="Include model round-trip reconstruction score in ranking")
    s.add_argument("--recon-weight", type=float, default=0.1, help="Penalty weight applied to reconstruction score")
    s.add_argument("--ast-artifact", default=None, help="Optional Bio-AST JSON artifact used for generation conditioning")
    s.add_argument("--ast-node-type-prompt", action="append", default=None, help="Preferred AST node type (repeatable)")
    s.add_argument("--ast-region-span", action="append", default=None, help="Conditioning span as START:END in base coordinates (repeatable)")
    s.add_argument("--ast-graph-mask", choices=["none", "neighbors", "successors"], default="none", help="Graph-guided token masking mode")
    s.add_argument("--ast-graph-hop-limit", type=int, default=1, help="Graph hop limit for AST mask expansion")
    s.add_argument("--ast-mask-strength", type=float, default=0.0, help="Mask strength [0..1] for AST-guided token preferences")
    # Advanced sampling strategies
    s.add_argument("--sampling-strategy", choices=["temperature", "top_k", "top_p", "beam", "anneal"], default=None,
                   help="Token sampling strategy (default: temperature)")
    s.add_argument("--top-k-tokens", type=int, default=0, help="Top-k sampling: keep only k highest-probability tokens (0=off)")
    s.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling: cumulative probability threshold (1.0=off)")
    s.add_argument("--beam-width", type=int, default=1, help="Beam search width (1=greedy)")
    s.add_argument("--anneal-start", type=float, default=1.2, help="Starting temperature for anneal strategy")
    s.add_argument("--anneal-end", type=float, default=0.5, help="Ending temperature for anneal strategy")
    # Latent space strategies
    s.add_argument("--latent-strategy", choices=["random", "interpolate", "arithmetic", "gradient", "walk"], default=None,
                   help="Latent vector generation strategy (default: random)")
    s.add_argument("--interpolate-steps", type=int, default=10, help="Number of SLERP interpolation points")
    s.add_argument("--interpolate-method", choices=["slerp", "lerp"], default="slerp", help="Interpolation method")
    s.add_argument("--optimize-property", choices=["gc", "reconstruction"], default="gc", help="Property to optimize via gradient descent")
    s.add_argument("--optimize-target", type=float, default=0.5, help="Target value for gradient optimization")
    s.add_argument("--optimize-steps", type=int, default=50, help="Gradient optimization steps")
    s.add_argument("--walk-steps", type=int, default=10, help="Number of latent walk points")
    s.add_argument("--walk-dim", type=int, default=0, help="Latent dimension to walk along")
    s.add_argument("--walk-range", type=float, default=3.0, help="Walk range in standard deviations")
    add_tok_args(s)
    add_ast_template_args(s)
    add_model_args(s)
    s.set_defaults(func=_lazy_cmd("cmd_generate_plasmid"))


    s = sub.add_parser("validate-plasmid")
    s.add_argument("--generated-fasta", required=True, help="Generated FASTA file to validate")
    s.add_argument("--catalog", required=True, help="Catalog of reference accessions")
    s.add_argument("--top-n", type=int, default=5, help="Show top N matching references (default: 5)")
    s.add_argument("--output-json", default=None, help="Optional JSON report path for top-N matches")
    s.add_argument("--force-fetch", action="store_true", help="Force re-fetching FASTA records from NCBI")
    s.set_defaults(func=_lazy_cmd("cmd_validate_plasmid"))

    s = sub.add_parser("generate-protein")
    s.add_argument("--length-aa", type=int, default=600)
    s.add_argument("--num-windows", type=int, default=None)
    s.add_argument("--window-aa", type=int, default=None)
    s.add_argument("--name", default="perceptrome_protein_1")
    s.add_argument("--output", default="generated/novel_protein.faa")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--latent-scale", type=float, default=1.0)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--reject", action="store_true", help="Rejection-sample until a basic protein-like filter passes")
    s.add_argument("--reject-tries", type=int, default=40)
    s.add_argument("--reject-max-run", type=int, default=10, help="Reject if any AA repeats longer than this")
    s.add_argument("--reject-max-x-frac", type=float, default=0.15, help="Reject if fraction of 'X' exceeds this")
    s.add_argument("--num-candidates", type=int, default=1)
    s.add_argument("--top-k", type=int, default=1)
    s.add_argument("--max-homopolymer", type=int, default=None)
    s.add_argument("--max-x-frac", type=float, default=None)
    s.add_argument("--max-internal-stops", type=int, default=0)
    s.add_argument("--summary-path", default=None, help="Optional JSON path for candidate summary (CSV written alongside)")
    s.add_argument("--scorecard-path", default=None, help="Optional JSON path for scorecard output artifact")
    s.add_argument("--scorecard-format", choices=["json"], default="json", help="Scorecard serialization format (default: json)")
    s.add_argument("--scorecard-risk-profile", default=None, help="Optional risk threshold preset (future extensible)")
    s.add_argument("--top-k-output", default=None, help="Optional FASTA path for ranked top-k candidates")
    s.add_argument("--roundtrip-score", action="store_true", help="Include model round-trip reconstruction score in ranking")
    s.add_argument("--recon-weight", type=float, default=0.1, help="Penalty weight applied to reconstruction score")
    s.add_argument("--ast-artifact", default=None, help="Optional Bio-AST JSON artifact used for generation conditioning")
    s.add_argument("--ast-node-type-prompt", action="append", default=None, help="Preferred AST node type (repeatable)")
    s.add_argument("--ast-region-span", action="append", default=None, help="Conditioning span as START:END in base coordinates (repeatable)")
    s.add_argument("--ast-graph-mask", choices=["none", "neighbors", "successors"], default="none", help="Graph-guided token masking mode")
    s.add_argument("--ast-graph-hop-limit", type=int, default=1, help="Graph hop limit for AST mask expansion")
    s.add_argument("--ast-mask-strength", type=float, default=0.0, help="Mask strength [0..1] for AST-guided token preferences")
    # Advanced sampling strategies
    s.add_argument("--sampling-strategy", choices=["temperature", "top_k", "top_p", "beam", "anneal"], default=None,
                   help="Token sampling strategy (default: temperature)")
    s.add_argument("--top-k-tokens", type=int, default=0, help="Top-k sampling: keep only k highest-probability tokens (0=off)")
    s.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling: cumulative probability threshold (1.0=off)")
    s.add_argument("--beam-width", type=int, default=1, help="Beam search width (1=greedy)")
    s.add_argument("--anneal-start", type=float, default=1.2, help="Starting temperature for anneal strategy")
    s.add_argument("--anneal-end", type=float, default=0.5, help="Ending temperature for anneal strategy")
    # Latent space strategies
    s.add_argument("--latent-strategy", choices=["random", "interpolate", "arithmetic", "gradient", "walk"], default=None,
                   help="Latent vector generation strategy (default: random)")
    s.add_argument("--interpolate-steps", type=int, default=10, help="Number of SLERP interpolation points")
    s.add_argument("--interpolate-method", choices=["slerp", "lerp"], default="slerp", help="Interpolation method")
    s.add_argument("--optimize-property", choices=["gc", "reconstruction"], default="gc", help="Property to optimize via gradient descent")
    s.add_argument("--optimize-target", type=float, default=0.5, help="Target value for gradient optimization")
    s.add_argument("--optimize-steps", type=int, default=50, help="Gradient optimization steps")
    s.add_argument("--walk-steps", type=int, default=10, help="Number of latent walk points")
    s.add_argument("--walk-dim", type=int, default=0, help="Latent dimension to walk along")
    s.add_argument("--walk-range", type=float, default=3.0, help="Walk range in standard deviations")
    add_ast_template_args(s)
    add_model_args(s)
    s.set_defaults(func=_lazy_cmd("cmd_generate_protein"))

    s = sub.add_parser("fold-one")
    s.add_argument("fasta", help="Protein FASTA file to fold (monomer only)")
    s.add_argument("--engine", choices=["colabfold"], default="colabfold")
    s.add_argument("--run-id", default=None, help="Optional run id (default: auto-generated)")
    s.add_argument("--num-recycle", type=int, default=3, help="ColabFold recycle count")
    s.add_argument("--num-models", type=int, default=5, help="ColabFold number of models")
    s.add_argument("--colabfold-bin", default=None, help="Path to colabfold_batch executable")
    s.set_defaults(func=_lazy_cmd("cmd_fold_one"))

    s = sub.add_parser("fold-batch")
    s.add_argument("input_dir", help="Directory containing protein FASTA files")
    s.add_argument("--engine", choices=["colabfold"], default="colabfold")
    s.add_argument("--run-id", default=None, help="Optional run id (default: auto-generated)")
    s.add_argument("--num-recycle", type=int, default=3)
    s.add_argument("--num-models", type=int, default=5)
    s.add_argument("--colabfold-bin", default=None, help="Path to colabfold_batch executable")
    s.add_argument("--min-protein-aa", type=int, default=None, help="Skip proteins shorter than this")
    s.add_argument("--max-protein-aa", type=int, default=None, help="Skip proteins longer than this")
    s.add_argument("--resume", action="store_true", help="Reuse existing successful fold outputs under runs/<run_id>/artifacts/fold/<protein_id>")
    s.add_argument("--keep-going", action="store_true", help="Continue batch when a fold item fails")
    s.set_defaults(func=_lazy_cmd("cmd_fold_batch"))

    s = sub.add_parser("fold-inspect")
    s.add_argument("run_or_path", help="Run id or direct path to fold summary json")
    s.set_defaults(func=_lazy_cmd("cmd_fold_inspect"))

    s = sub.add_parser("fold-export")
    s.add_argument("run_or_path", help="Run id or direct path to fold summary json")
    s.add_argument("--output-dir", default=None, help="Export directory (default: run outputs)")
    s.set_defaults(func=_lazy_cmd("cmd_fold_export"))

    s = sub.add_parser("design-loop")
    s.add_argument("--catalog", required=True, help="Catalog of reference accessions")
    s.add_argument("--rounds", type=int, default=5, help="Number of loop rounds/generations")
    s.add_argument("--generations", type=int, default=None, help="Alias for --rounds")
    s.add_argument("--population-size", type=int, default=8, help="Candidate population size per round")
    s.add_argument("--survivor-count", type=int, default=3, help="Elitism count preserved each round")
    s.add_argument("--mutation-rate", type=float, default=0.05, help="Mutation rate for offspring")
    s.add_argument("--mutation-scale", type=float, default=1.0, help="Mutation scale multiplier")
    s.add_argument("--crossover-rate", type=float, default=0.5, help="Crossover probability")
    s.add_argument("--early-stop-threshold", type=float, default=None, help="Optional score threshold for early stop")
    s.add_argument("--length-bp", type=int, default=4000, help="Designed sequence length in base pairs")
    s.add_argument("--target-gc", type=float, default=0.5, help="Target GC fraction")
    s.add_argument("--max-homopolymer", type=int, default=8, help="Homopolymer penalty threshold")
    s.add_argument("--seed", type=int, default=None, help="RNG seed")
    s.add_argument("--name", default="design_loop_best", help="Best candidate FASTA name")
    s.add_argument("--output", default="design_loop_best.fasta", help="Best candidate output FASTA path")
    s.add_argument("--force-fetch", action="store_true", help="Force fetch of reference records")
    s.add_argument("--ast-artifact", default=None, help="Optional Bio-AST JSON artifact used for generation conditioning")
    s.add_argument("--ast-node-type-prompt", action="append", default=None, help="Preferred AST node type (repeatable)")
    s.add_argument("--ast-region-span", action="append", default=None, help="Conditioning span as START:END in base coordinates (repeatable)")
    s.add_argument("--ast-graph-mask", choices=["none", "neighbors", "successors"], default="none", help="Graph-guided token masking mode")
    s.add_argument("--ast-graph-hop-limit", type=int, default=1, help="Graph hop limit for AST mask expansion")
    s.add_argument("--ast-mask-strength", type=float, default=0.0, help="Mask strength [0..1] for AST-guided token preferences")
    add_ast_template_args(s)
    s.set_defaults(func=_lazy_cmd("cmd_design_loop"))

    s = sub.add_parser("compare-lanes")
    s.add_argument("--split-name", default="default", help="Split set name used to select train/val/test accession lists")
    s.add_argument("--baseline-split-path", default=None, help="Optional explicit split JSON for baseline lane")
    s.add_argument("--ast-split-path", default=None, help="Optional explicit split JSON for AST lane")
    s.add_argument("--baseline-config", default=None, help="Optional config path for baseline lane")
    s.add_argument("--ast-config", default=None, help="Optional config path for AST lane")
    s.add_argument("--baseline-model-type", default="mlp", choices=["mlp", "transformer", "ssm", "tree", "hybrid"], help="Model type for baseline lane")
    s.add_argument("--ast-model-type", default="hybrid", choices=["mlp", "transformer", "ssm", "tree", "hybrid"], help="Model type for AST-enabled lane")
    s.add_argument("--max-epochs", type=int, default=1, help="Epoch count applied to each lane")
    s.add_argument("--seed", type=int, default=1337, help="Seed reused for both lanes")
    s.add_argument("--experiment-id", default=None, help="Optional run id for compare-lanes output")
    s.add_argument("--steps-per-plasmid", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--window-size", type=int, default=None)
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--delete-cache", action="store_true")
    add_tok_args(s)
    add_loss_args(s)
    add_tensorboard_args(s)
    s.set_defaults(func=_lazy_cmd("cmd_compare_lanes"))

    s = sub.add_parser("run-job-spec")
    s.add_argument("--spec-json", required=True, help="Serialized JobSpec JSON payload")
    s.set_defaults(func=_lazy_cmd("cmd_run_job_spec"))

    s = sub.add_parser("pretrain")
    s.add_argument("--dataset", default=None, help="Path to tensorized pretraining .npz archive")
    s.add_argument("--vocab-size", type=int, default=None, help="Tokenizer vocabulary size")
    s.add_argument("--batch-size", type=int, default=None, help="Pretraining batch size")
    s.add_argument("--epochs", type=int, default=None, help="Number of pretraining epochs")
    s.add_argument("--hidden-size", type=int, default=None, help="Backbone hidden size")
    s.add_argument("--learning-rate", type=float, default=None, help="Optimizer learning rate")
    s.add_argument("--output-dir", default=None, help="Checkpoint/log output directory")
    s.add_argument("--seed", type=int, default=None, help="Global RNG seed for pretraining")
    s.add_argument("--disable-mlm", action="store_true", help="Disable masked token objective")
    s.add_argument("--disable-sme", action="store_true", help="Disable masked SME objective")
    s.add_argument("--disable-contrastive", action="store_true", help="Disable contrastive objective")
    s.set_defaults(func=_lazy_cmd("cmd_pretrain"))

    s = sub.add_parser("bio-ast", help="Bio-AST workbench commands")
    bio_ast_sub = s.add_subparsers(dest="bio_ast_command", required=True)

    b = bio_ast_sub.add_parser("build", help="Build Bio-AST transform artifacts for an accession")
    b.add_argument("accession")
    b.add_argument("--source", choices=["fasta", "genbank"], default="genbank")
    b.add_argument("--force", action="store_true", help="Force refetch of source record")
    b.set_defaults(func=_lazy_cmd("cmd_bio_ast_build"))

    e = bio_ast_sub.add_parser("export", help="Export one or all Bio-AST transforms")
    e.add_argument("accession")
    e.add_argument("--transform", choices=["canonical_ast", "motif_features", "tree_tensors", "graph_edges", "tree_json", "graph_json", "storage_map", "summary_json", "all"], default="all")
    e.add_argument("--output", required=True)
    e.set_defaults(func=_lazy_cmd("cmd_bio_ast_export"))

    i = bio_ast_sub.add_parser("inspect", help="Inspect Bio-AST artifact counts")
    i.add_argument("accession")
    i.set_defaults(func=_lazy_cmd("cmd_bio_ast_inspect"))


    m = bio_ast_sub.add_parser("embed-export", help="Export Bio-AST embeddings (.npy + metadata) for accession(s)")
    m.add_argument("--accession", action="append", default=[], help="Accession to embed (repeatable)")
    m.add_argument("--catalog", default=None, help="Optional catalog file with accession list")
    m.add_argument("--source", choices=["fasta", "genbank"], default="genbank")
    m.add_argument("--force", action="store_true", help="Force refetch of source record")
    m.add_argument("--hidden-dim", type=int, default=None, help="Embedding hidden dimension (default from training config)")
    m.add_argument("--ast-tree-layers", type=int, default=4)
    m.add_argument("--ast-motif-kernel-size", type=int, default=7)
    m.add_argument("--ast-motif-channels", type=int, default=64)
    m.set_defaults(func=_lazy_cmd("cmd_bio_ast_embed_export"))

    v = bio_ast_sub.add_parser("visualize", help="Build tree/graph visualization artifacts for an accession")
    v.add_argument("--accession", required=True)
    v.add_argument("--source", choices=["fasta", "genbank"], default="genbank")
    v.add_argument("--force", action="store_true", help="Force refetch of source record")
    v.set_defaults(func=_lazy_cmd("cmd_bio_ast_visualize"))

    s = sub.add_parser("tui", help="Launch the Perceptrome terminal UI")
    s.add_argument("--run-id", default=None, help="Open TUI with run context preselected")
    s.add_argument("--job-id", default=None, help="Open TUI with job context preselected")
    s.add_argument("--panel", default=None, help="Open directly on panel id (for example: overview, train, jobs)")
    s.add_argument("--detail-surface", choices=["logs", "diagnostics", "resources", "traceback", "artifact"], default=None, help="Open detail surface immediately")
    s.set_defaults(func=_run_tui)

    return p

def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
