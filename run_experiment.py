#!/usr/bin/env python
"""CLI for preparing data, generating missingness, and running PMAE experiments."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pmae_tabular.data.registry import DATASET_SPECS, canonical_dataset_name
from pmae_tabular.experiments import prepare_one, amputate_one, run_experiment


def _add_dataset_arg(parser):
    parser.add_argument("--dataset", required=True,
                        help="Dataset key (see `list-datasets` for available names).")


def build_parser():
    parser = argparse.ArgumentParser(description="PMAE tabular imputation experiment pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-datasets", help="List supported datasets and their feature sources.")

    prepare = sub.add_parser("prepare", help="Load and preprocess one raw dataset.")
    _add_dataset_arg(prepare)
    prepare.add_argument("--force", action="store_true", help="Rebuild processed files.")

    prepare_all = sub.add_parser("prepare-all", help="Preprocess every registered dataset.")
    prepare_all.add_argument("--force", action="store_true")

    amputate = sub.add_parser("amputate", help="Generate and save an incomplete dataset.")
    _add_dataset_arg(amputate)
    amputate.add_argument("--mechanism", choices=["MCAR", "MAR", "MNAR"], default="MCAR")
    amputate.add_argument("--column-fraction", type=float, default=0.3,
                          help="Fraction of feature columns selected for missingness (default: 0.3).")
    amputate.add_argument("--missing-rate", type=float, default=0.5,
                          help="Missing fraction within selected columns (default: 0.5).")
    amputate.add_argument("--seed", type=int, default=42)
    amputate.add_argument("--force", action="store_true", help="Regenerate if output already exists.")

    run = sub.add_parser("run", help="Run PMAE and/or ReMasker, then evaluate hidden values.")
    _add_dataset_arg(run)
    run.add_argument("--mechanism", choices=["MCAR", "MAR", "MNAR"], default="MCAR")
    run.add_argument("--column-fraction", type=float, default=0.3)
    run.add_argument("--missing-rate", type=float, default=0.5)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--epochs", type=int, default=300)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--models", default="pmae,remasker",
                     help="Comma-separated list: pmae, remasker, or both (default: both).")
    run.add_argument("--embed-dim", type=int, default=32)
    run.add_argument("--depth", type=int, default=4)
    run.add_argument("--decoder-depth", type=int, default=2)
    run.add_argument("--num-heads", type=int, default=4)
    run.add_argument("--mlp-ratio", type=float, default=4.0)
    run.add_argument("--force-prepare", action="store_true")
    run.add_argument("--force-amputation", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "list-datasets":
        for name, spec in DATASET_SPECS.items():
            print(f"{name:22} {spec.description}")
        return 0

    if args.command == "prepare":
        key = canonical_dataset_name(args.dataset)
        payload = prepare_one(key, ROOT, force=args.force)
        print(json.dumps({"dataset": key, "samples": payload["n_samples"],
                          "features": payload["n_features"],
                          "numerical": payload["d_numerical"],
                          "categorical": len(payload["categorical_columns"])}, indent=2))
        return 0

    if args.command == "prepare-all":
        for name in DATASET_SPECS:
            payload = prepare_one(name, ROOT, force=args.force)
            print(f"Prepared {name}: {payload['n_samples']} rows x {payload['n_features']} features")
        return 0

    if args.command == "amputate":
        key = canonical_dataset_name(args.dataset)
        output = amputate_one(key, ROOT, args.mechanism, args.column_fraction,
                              args.missing_rate, args.seed, force=args.force)
        print(f"Saved amputated dataset: {output.relative_to(ROOT)}")
        return 0

    if args.command == "run":
        key = canonical_dataset_name(args.dataset)
        models = tuple(x.strip().lower() for x in args.models.split(",") if x.strip())
        result = run_experiment(
            key, ROOT, mechanism=args.mechanism,
            column_fraction=args.column_fraction, missing_rate=args.missing_rate,
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
            device=args.device, models=models, force_prepare=args.force_prepare,
            force_amputation=args.force_amputation, embed_dim=args.embed_dim,
            depth=args.depth, decoder_depth=args.decoder_depth,
            num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
