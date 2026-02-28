#!/usr/bin/env python3
import argparse
import importlib
from typing import Any


def _lazy_cmd(func_name: str):
    def _run(args):
        mod = importlib.import_module("perceptrome.cli.commands")
        return getattr(mod, func_name)(args)

    return _run

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
    s.add_argument("--top-k-output", default=None, help="Optional FASTA path for ranked top-k candidates")
    s.add_argument("--roundtrip-score", action="store_true", help="Include model round-trip reconstruction score in ranking")
    s.add_argument("--recon-weight", type=float, default=0.1, help="Penalty weight applied to reconstruction score")
    add_tok_args(s)
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
    s.add_argument("--top-k-output", default=None, help="Optional FASTA path for ranked top-k candidates")
    s.add_argument("--roundtrip-score", action="store_true", help="Include model round-trip reconstruction score in ranking")
    s.add_argument("--recon-weight", type=float, default=0.1, help="Penalty weight applied to reconstruction score")
    add_model_args(s)
    s.set_defaults(func=_lazy_cmd("cmd_generate_protein"))

    s = sub.add_parser("pretrain")
    s.add_argument("--dataset", default=None, help="Path to tensorized pretraining .npz archive")
    s.add_argument("--vocab-size", type=int, default=None, help="Tokenizer vocabulary size")
    s.add_argument("--batch-size", type=int, default=None, help="Pretraining batch size")
    s.add_argument("--epochs", type=int, default=None, help="Number of pretraining epochs")
    s.add_argument("--hidden-size", type=int, default=None, help="Backbone hidden size")
    s.add_argument("--learning-rate", type=float, default=None, help="Optimizer learning rate")
    s.add_argument("--output-dir", default=None, help="Checkpoint/log output directory")
    s.add_argument("--disable-mlm", action="store_true", help="Disable masked token objective")
    s.add_argument("--disable-sme", action="store_true", help="Disable masked SME objective")
    s.add_argument("--disable-contrastive", action="store_true", help="Disable contrastive objective")
    s.set_defaults(func=_lazy_cmd("cmd_pretrain"))

    return p

def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
