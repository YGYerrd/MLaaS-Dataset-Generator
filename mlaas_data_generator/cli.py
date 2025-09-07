"""Command line interface for the MLaaS data generator."""

from __future__ import annotations
from typing import List
from pathlib import Path
import argparse
import json

from .config import CONFIG
from .data_utils import prepare_client_distributions
from .files import combine_data_files
from .path_resolver import resolve_output_path

def add_generate_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "generate",
        help="Run federated training and write results"
    )
    p.add_argument(
        "--clients", type=int, 
        default=CONFIG["num_clients"], 
        help="Number of clients"
        )    
    p.add_argument(
        "--output", type=str, 
        default="clients.csv", 
        help="Output CSV file"
    )
    p.add_argument(
        "--dataset", type=str,
        default= "mnist",
        choices=["fashion_mnist", "mnist"],
        help="Dataset to use"
    )
    p.add_argument(
        "--strategy", type=str, default="iid",
        choices=["iid", "quantity_skew", "dirichlet", "shard", "label_per_client", "custom"],
        help="Data splitting strategy",
    )
    p.add_argument(
        "--distribution-param", type=float, default=None,
        help="Parameter for the chosen data split strategy",
    )
    p.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of samples to draw from the dataset before splitting",
    )
    p.add_argument(
        "--sample-frac", type=float, default=None,
        help="Fraction of the dataset to use before splitting",
    )
    p.add_argument("--distribution", type=str, default=None,
                help="Path to JSON defining per-client label counts (only used with --strategy custom)")
    
    p.set_defaults(_cmd="generate")

    
def add_merge_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "merge",
        help="Merge CSVs into one file"
    )
    p.add_argument("inputs", nargs="+",
                   help="Input CSV files or globs (e.g. data/*.csv)")
    p.add_argument("--output", type=str, default="merged.csv",
                   help="Destination CSV path")
    p.add_argument("--id-col", default="MLaaS_ID",
                   help="Name of the sequential ID column (default: MLaaS_ID)")
    p.add_argument("--start-id", type=int, default=1,
                   help="First ID value (default: 1)")
    p.add_argument("--dedupe", action="store_true",
                   help="Drop duplicate rows before assigning IDs")
    p.set_defaults(_cmd="merge")


def expand_inputs(patterns: List[str]) -> List[str]:
    """Supports global and explicit paths for file merging"""
    paths: list[str] = []
    for pat in patterns:
        matches = sorted(str(p) for p in Path().glob(pat)) if any(ch in pat for ch in "*?[]") else [pat]
        paths.extend(matches)
   
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MLaaS client data")
    subparsers = parser.add_subparsers(dest="command")
    
    add_merge_subparser(subparsers)
    add_generate_subparser(subparsers)

    # If no sub command provided, act like generate
    import sys
    if len(sys.argv) > 1 and sys.argv[1] not in {"generate", "merge"}:
        # legacy style: python -m mlaas_data_generator.cli --clients 5
        sys.argv.insert(1, "generate")

    args = parser.parse_args()

    if args._cmd == "generate":
        from .federated import FederatedDataGenerator

        config = CONFIG.copy()
        config["num_clients"] = args.clients
        config["distribution_type"] = args.strategy
        config["distribution_param"] = args.distribution_param
        if args.sample_size is not None:
            config["sample_size"] = args.sample_size
        if args.sample_frac is not None:
            config["sample_frac"] = args.sample_frac

        if args.strategy == "custom":
            if not args.distribution:
                raise SystemExit("--distribution is required when --strategy custom")
            with open(args.distribution, "r") as f:
                custom_distributions = json.load(f)
                config["custom_distributions"] = custom_distributions
        generator = FederatedDataGenerator(config, dataset=args.dataset)
        df = generator.run()
        out_path = resolve_output_path(args.output, kind="run")
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} client records to {out_path}")
        return
    
    if args._cmd == "merge":
        inputs = expand_inputs(args.inputs)

        if not inputs:
            raise SystemExit("No files match the provide paths")
        
        out_path = resolve_output_path(args.output, kind="merged")
        
        combined = combine_data_files(
            paths=inputs,
            output_path=out_path,
            id_col=args.id_col,
            start_id=args.start_id,
            dedupe=args.dedupe
        )

        print(f"Merged {len(inputs)} files into {out_path} ({len(combined)} rows).")
        return
    
    parser.print_help()
        

if __name__ == "__main__":
    main()