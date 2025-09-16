"""Command line interface for the MLaaS data generator."""

from __future__ import annotations
from typing import List
from pathlib import Path
import argparse
import json
import questionary

from .config import CONFIG
from .files import combine_data_files
from .path_resolver import resolve_output_path
from .federated import FederatedDataGenerator

def add_wizard_subparser(subparsers):
    p = subparsers.add_parser("wizard", help="Interactive setup & run")
    p.add_argument("--save", type=str, default="run_config.json",
                   help="Where to save the chosen config (JSON)")
    p.set_defaults(_cmd="wizard")


def add_generate_subparser(subparsers):
    p = subparsers.add_parser(
        "generate",
        help="Run federated training and write results"
    )
    p.add_argument(
        "--rounds", type=int, default=None,
        help="Number of global federated rounds (overrides config)"
    )
    p.add_argument(
        "--clients", type=int, 
        default=CONFIG["num_clients"], 
        help="Number of clients"
        )
    p.add_argument(
        "--client-dropout-rate", type=float, default=None,
        help="Per-round probability a client skips training (0.0–1.0)"
    )
    p.add_argument(
        "--model-type", type=str, default=None,
        choices=["CNN","MLP","LogReg","ResNet","Custom"], 
        help="Model family recorded in run metadata (and used by create_model if applicable)"
    )
    p.add_argument(
        "--task-type", type=str, default=None,
        choices=["classification","regression"],
        help="Task type metadata"
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
        "--seed", type=int, default=None,
        help="Random seed for shuffling, splitting, and participation"
    )
    p.add_argument(
        "--strategy", type=str, default="iid",
        choices=["iid", "quantity_skew", "dirichlet", "shard", "label_per_client", "custom"],
        help="Data splitting strategy",
    )
    p.add_argument(
        "--no-save-weights", action="store_true",
        help="Do not write per-client and global weight JSONs"
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
    p.add_argument(
        "--distribution", type=str, default=None,
        help="Path to JSON defining per-client label counts (only used with --strategy custom)"
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size per client (overrides config)"
    )
    p.add_argument(
        "--learning-rate", type=float, default=None,
        help="Learning rate (overrides config)"
    )
    p.add_argument(
        "--epochs-per-round", type=int, default=None,
        help="Local epochs per round (overrides config)"
    )
    p.add_argument(
        "--hidden-layers", type=str, default=None,
        help="Comma-separated hidden sizes, e.g. '128,64'"
    )
    p.add_argument(
        "--activation", type=str, default=None,
        choices=["relu","tanh","sigmoid","gelu","elu","selu","softplus","linear"],
        help="Activation for hidden layers"
    )
    p.add_argument(
        "--dropout", type=float, default=None,
        help="Dropout probability (0.0-0.9)"
    )
    p.add_argument(
        "--weight-decay", type=float, default=None,
        help="L2 regularization coefficient"
    )
    p.add_argument(
        "--optimizer", type=str, default=None,
        choices=["adam","adamw","sgd","rmsprop","adagrad"],
        help="Optimizer"
    )

    p.set_defaults(_cmd="generate")

    
def add_merge_subparser(subparsers):
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
    add_wizard_subparser(subparsers)

    # If no sub command provided, act like generate
    import sys
    if len(sys.argv) > 1 and sys.argv[1] not in {"generate", "merge", "wizard"}:
        # legacy style: python -m mlaas_data_generator.cli --clients 5
        sys.argv.insert(1, "generate")

    args = parser.parse_args()

    if args._cmd == "generate":

        config = CONFIG.copy()
        config["num_clients"] = args.clients
        config["distribution_type"] = args.strategy
        if args.rounds is not None:                 config["num_rounds"] = int(args.rounds)
        if args.batch_size is not None:             config["batch_size"] = args.batch_size
        if args.learning_rate is not None:          config["learning_rate"] = args.learning_rate
        if args.epochs_per_round is not None:       config["local_epochs"] = args.epochs_per_round
        if args.sample_size is not None:            config["sample_size"] = args.sample_size
        if args.sample_frac is not None:            config["sample_frac"] = args.sample_frac
        if args.hidden_layers:                      config["hidden_layers"] = [int(x) for x in args.hidden_layers.split(",") if x.strip()]
        if args.activation is not None:             config["activation"] = args.activation
        if args.dropout is not None:                config["dropout"] = float(args.dropout)
        if args.weight_decay is not None:           config["weight_decay"] = float(args.weight_decay)
        if args.optimizer is not None:              config["optimizer"] = args.optimizer
        if args.epochs_per_round is not None:       config["local_epochs"] = int(args.epochs_per_round)
        if args.client_dropout_rate is not None:    config["client_dropout_rate"] = float(args.client_dropout_rate)
        if args.model_type is not None:             config["model_type"] = args.model_type
        if args.task_type is not None:              config["task_type"] = args.task_type
        if args.seed is not None:                   config["seed"] = int(args.seed)
        if args.no_save_weights:                    config["save_weights"] = False
        if args.distribution_param is not None:
            dp_raw = args.distribution_param.strip()
            if args.strategy in {"shard","label_per_client"}:
                config["distribution_param"] = int(float(dp_raw))  # accept "2" or "2.0"
            else:
                config["distribution_param"] = float(dp_raw)
        if args.strategy == "custom":
            if not args.distribution:
                raise SystemExit("--distribution is required when --strategy custom")
            with open(args.distribution, "r") as f:
                custom_distributions = json.load(f)
                config["custom_distributions"] = custom_distributions

        generator = FederatedDataGenerator(config, dataset=args.dataset, task_type=config.get("task_type", "classification"), model_type=config.get("model_type", "CNN"))
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
    
    if args._cmd == "wizard":

        dataset = questionary.select(
            "Select dataset:", choices=["fashion_mnist", "mnist"], default="fashion_mnist"
        ).ask()

        num_clients = int(questionary.text("Number of clients:", default="10").ask())

        strategy = questionary.select(
            "Partition strategy:",
            choices=["iid","quantity_skew","dirichlet","shard","label_per_client","custom"],
            default="iid",
        ).ask()

        dist_param = None
        dist_help = {
            "quantity_skew": "Dirichlet alpha over client sizes (e.g., 0.5, 1.0, 2.0)",
            "dirichlet": "Dirichlet alpha over labels per client (e.g., 0.2, 0.5, 1.0)",
            "shard": "Shards per client (integer, e.g., 2)",
            "label_per_client": "k labels per client (integer, e.g., 2)",
        }
        if strategy in dist_help:
            raw = questionary.text(f"{dist_help[strategy]} (blank = default):", default="").ask()
            dist_param = None if raw.strip()=="" else (float(raw) if strategy in ["quantity_skew","dirichlet"] else int(raw))

        custom_path = None
        if strategy == "custom":
            custom_path = questionary.path("Path to custom distributions JSON:").ask()

        ss_raw  = questionary.text("Sample SIZE (blank to skip):", default="").ask().strip()
        sf_raw  = questionary.text("Sample FRAC [0-1] (blank to skip):", default="").ask().strip()
        sample_size = int(ss_raw) if ss_raw else None
        sample_frac = float(sf_raw) if sf_raw else None
        
        model_type = questionary.select(
            "Model type (metadata + potential model selection):",
            choices=["CNN"],
            default="CNN"
        ).ask()

        task_type = questionary.select(
            "Task type:",
            choices=["classification"],
            default="classification"
        ).ask()

        hidden_layers = questionary.text("Hidden layers (comma-separated, e.g., 128,64):", default="64").ask()
        activation = questionary.select("Activation:", choices=["relu","tanh","sigmoid","gelu","elu","selu","softplus","linear"], default="relu").ask()
        optimizer  = questionary.select("Optimizer:", choices=["adam","adamw","sgd","rmsprop","adagrad"], default="adam").ask()
        dropout    = float(questionary.text("Dropout (0.0-0.9):", default="0.0").ask())
        weight_decay = float(questionary.text("Weight decay L2 (e.g., 0.0, 1e-4):", default="0.0").ask())

        # Training knobs
        learning_rate   = float(questionary.text("Learning rate:", default="0.001").ask())
        batch_size      = int(questionary.text("Batch size:", default="32").ask())
        epochs_per_round= int(questionary.text("Epochs per round:", default="3").ask())
        num_rounds      = int(questionary.text("Global rounds:", default="5").ask())
        cdr_raw = questionary.text(
                "Client dropout rate per round (0.0–1.0):", default="0.0"
            ).ask().strip()
        client_dropout_rate = float(cdr_raw) if cdr_raw else 0.0

        # Reproducibility
        seed_raw = questionary.text("Random seed (blank = none):", default="").ask().strip()
        seed = int(seed_raw) if seed_raw else None

        # Save weights?
        save_weights = questionary.confirm("Save per-round weight files?", default=True).ask()
        
        output = questionary.text("Output CSV filename:", default="clients.csv").ask()

        # 2) Build config
        config = {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": epochs_per_round,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_layers": [int(x) for x in hidden_layers.split(",") if x.strip()],
            "activation": activation,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "distribution_type": strategy,
            "distribution_param": dist_param,
            "sample_size": sample_size,
            "sample_frac": sample_frac,
            "model_type": model_type,
            "task_type": task_type,
            "client_dropout_rate": client_dropout_rate,
            "save_weights": bool(save_weights),
        }

        if seed is not None:
            config["seed"] = seed

        if strategy == "custom":
            import json as _json
            with open(custom_path, "r") as f:
                config["custom_distributions"] = _json.load(f)

        # 3) Confirm summary
        print("\n=== Run Summary ===")
        import pprint; pprint.pprint({"dataset": dataset, **config, "output": output})
        if not questionary.confirm("Proceed?").ask():
            print("Aborted.")
            return

        # 4) Save config for reproducibility
        import json as _json
        with open(args.save, "w") as f:
            _json.dump({"dataset": dataset, **config, "output": output}, f, indent=2)
        print(f"Saved configuration to {args.save}")

        # 5) Run
        gen = FederatedDataGenerator(config=config, dataset=dataset)
        df = gen.run()
        from .path_resolver import resolve_output_path
        out_path = resolve_output_path(output, kind="run")
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} client records to {out_path}")
        return

    
    parser.print_help()
        

if __name__ == "__main__":
    main()