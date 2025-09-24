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

DATASET_CHOICES = [
    "fashion_mnist",
    "mnist",
    "cifar10",
    "digits",
    "iris",
    "wine",
    "california_housing",
]

# ---------------- Helpers ----------------

def expand_inputs(patterns: List[str]) -> List[str]:
    """Supports globs and explicit paths for file merging."""
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


def _coerce_value(raw: str):
    raw = raw.strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_dataset_args(pairs: List[str] | None):
    args = {}
    if not pairs:
        return args
    for raw in pairs:
        if "=" not in raw:
            raise SystemExit(f"Invalid dataset argument '{raw}'. Expected KEY=VALUE format.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid dataset argument '{raw}'. Key cannot be empty.")
        args[key] = _coerce_value(value)
    return args


def infer_dataset_profile(dataset: str) -> dict:
    ds = dataset.lower()
    if ds in {"mnist", "fashion_mnist", "cifar10"}:
        return {
            "task": "classification",
            "ask_split": False,            
            "split_key": None,             
            "ask_scaler": False,           
            "default_model": "CNN",
            "allow_label_splits": True,    
            "allow_quantity_skew": True,
            "allow_custom": True,          
            "ask_target_scaler": False,    
        }

    if ds in {"iris", "wine", "digits"}:
        return {
            "task": "classification",          
            "ask_split": True,             
            "split_key": "test_size",
            "ask_scaler": True,            
            "default_model": "MLP",
            "allow_label_splits": True,    
            "allow_quantity_skew": True,
            "allow_custom": True,
            "ask_target_scaler": False,
        }

    if ds == "california_housing":
        return {
            "task": "regression",
            "ask_split": True,             
            "split_key": "test_size",
            "ask_scaler": True,            
            "default_model": "MLP",
            "allow_label_splits": False,   
            "allow_quantity_skew": True,   
            "allow_custom": False,         
            "ask_target_scaler": True,     
        }

    return {
        "task": "classification",
        "ask_split": False,
        "split_key": None,
        "ask_scaler": False,
        "default_model": "MLP",
        "allow_label_splits": True,
        "allow_quantity_skew": True,
        "allow_custom": True,
        "ask_target_scaler": False,
    }

# ---------------- Subparsers ----------------

def add_wizard_subparser(subparsers):
    p = subparsers.add_parser("wizard", help="Interactive setup & run")
    p.add_argument("--save", type=str, default="run_config.json",
                   help="Where to save the chosen config (JSON)")
    p.set_defaults(_cmd="wizard")


def add_generate_subparser(subparsers):
    p = subparsers.add_parser("generate", help="Run federated training and write results")
    p.add_argument("--rounds", type=int, default=None, help="Number of global federated rounds (overrides config)")
    p.add_argument("--clients", type=int, default=CONFIG["num_clients"], help="Number of clients")
    p.add_argument("--client-dropout-rate", type=float, default=None, help="Per-round probability a client skips training (0.0–1.0)")
    p.add_argument("--model-type", type=str, default=None, choices=["CNN","MLP","LogReg","ResNet","Custom"],
                   help="Model family recorded in run metadata (and used by create_model if applicable)")
    p.add_argument("--task-type", type=str, default=None, choices=["classification","regression", "clustering"], help="Task type metadata")
    p.add_argument("--output", type=str, default="clients.csv", help="Output CSV file")
    p.add_argument("--dataset", type=str, default="mnist", choices=DATASET_CHOICES, help="Dataset to use")
    p.add_argument("--dataset-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="Additional dataset loader arguments (repeatable)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for shuffling, splitting, and participation")
    p.add_argument("--strategy", type=str, default="iid",
                   choices=["iid", "quantity_skew", "dirichlet", "shard", "label_per_client", "custom"],
                   help="Data splitting strategy")
    p.add_argument("--no-save-weights", action="store_true", help="Do not write per-client and global weight JSONs")
    p.add_argument("--distribution-param", type=float, default=None, help="Parameter for the chosen data split strategy")
    p.add_argument("--sample-size", type=int, default=None, help="Number of samples to draw from the dataset before splitting")
    p.add_argument("--sample-frac", type=float, default=None, help="Fraction of the dataset to use before splitting")
    p.add_argument("--distribution", type=str, default=None,
                   help="Path to JSON defining per-client label counts (only used with --strategy custom)")
    p.add_argument("--distribution-bins", type=int, default=None, help="Number of bins when summarising regression targets")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size per client (overrides config)")
    p.add_argument("--learning-rate", type=float, default=None, help="Learning rate (overrides config)")
    p.add_argument("--epochs-per-round", type=int, default=None, help="Local epochs per round (overrides config)")
    p.add_argument("--hidden-layers", type=str, default=None, help="Comma-separated hidden sizes, e.g. '128,64'")
    p.add_argument("--activation", type=str, default=None,
                   choices=["relu","tanh","sigmoid","gelu","elu","selu","softplus","linear"],
                   help="Activation for hidden layers")
    p.add_argument("--dropout", type=float, default=None, help="Dropout probability (0.0-0.9)")
    p.add_argument("--weight-decay", type=float, default=None, help="L2 regularization coefficient")
    p.add_argument("--optimizer", type=str, default=None, choices=["adam","adamw","sgd","rmsprop","adagrad"], help="Optimizer")
    p.set_defaults(_cmd="generate")


def add_merge_subparser(subparsers):
    p = subparsers.add_parser("merge", help="Merge CSVs into one file")
    p.add_argument("inputs", nargs="+", help="Input CSV files or globs (e.g. data/*.csv)")
    p.add_argument("--output", type=str, default="merged.csv", help="Destination CSV path")
    p.add_argument("--id-col", default="MLaaS_ID", help="Name of the sequential ID column (default: MLaaS_ID)")
    p.add_argument("--start-id", type=int, default=1, help="First ID value (default: 1)")
    p.add_argument("--dedupe", action="store_true", help="Drop duplicate rows before assigning IDs")
    p.set_defaults(_cmd="merge")

# ---------------- Main ----------------

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
        config["dataset"] = args.dataset
        config["model_type"] = args.model_type or config.get("model_type", "CNN")

        dataset_args = parse_dataset_args(args.dataset_arg)
        if dataset_args:
            config["dataset_args"] = dataset_args
        if args.distribution_bins is not None:
            config["distribution_bins"] = int(args.distribution_bins)
        if args.rounds is not None:
            config["num_rounds"] = int(args.rounds)
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.learning_rate is not None:
            config["learning_rate"] = args.learning_rate
        if args.epochs_per_round is not None:
            config["local_epochs"] = int(args.epochs_per_round)
        if args.sample_size is not None:
            config["sample_size"] = args.sample_size
        if args.sample_frac is not None:
            config["sample_frac"] = args.sample_frac
        if args.hidden_layers:
            config["hidden_layers"] = [int(x) for x in args.hidden_layers.split(",") if x.strip()]
        if args.activation is not None:
            config["activation"] = args.activation
        if args.dropout is not None:
            config["dropout"] = float(args.dropout)
        if args.weight_decay is not None:
            config["weight_decay"] = float(args.weight_decay)
        if args.optimizer is not None:
            config["optimizer"] = args.optimizer
        if args.client_dropout_rate is not None:
            config["client_dropout_rate"] = float(args.client_dropout_rate)
        if args.model_type is not None:
            config["model_type"] = args.model_type
        if args.task_type is not None:
            config["task_type"] = args.task_type
        if args.seed is not None:
            config["seed"] = int(args.seed)
        if args.no_save_weights:
            config["save_weights"] = False
        if args.distribution_param is not None:
            if args.strategy in {"shard", "label_per_client"}:
                config["distribution_param"] = int(args.distribution_param)
            else:
                config["distribution_param"] = float(args.distribution_param)
        if args.strategy == "custom":
            if not args.distribution:
                raise SystemExit("--distribution is required when --strategy custom")
            with open(args.distribution, "r") as f:
                custom_distributions = json.load(f)
                config["custom_distributions"] = custom_distributions

        generator = FederatedDataGenerator(
            config=config,
            dataset=args.dataset,
            task_type=config.get("task_type"),
            model_type=config.get("model_type"),
        )
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
            dedupe=args.dedupe,
        )
        print(f"Merged {len(inputs)} files into {out_path} ({len(combined)} rows).")
        return

    if args._cmd == "wizard":
        # -- 1) Dataset & profile
        dataset = questionary.select(
            "Select dataset:", choices=DATASET_CHOICES, default="fashion_mnist"
        ).ask()
        profile = infer_dataset_profile(dataset)

        # Ask whether to show advanced questions up front
        show_advanced = questionary.confirm(
            "Show advanced options?", default=False
        ).ask()

        dataset_args = {}

        # Only prompt for split/scaler when relevant (tabular, not vision)
        if profile["ask_split"]:
            # Split
            default_split = "0.2"
            if show_advanced:
                split_raw = questionary.text(
                    f"Test split fraction (default {default_split}):", default=default_split
                ).ask().strip()
                if split_raw:
                    dataset_args[profile["split_key"]] = float(split_raw)
            else:
                dataset_args[profile["split_key"]] = float(default_split)

            # Seed (defaults tuned per dataset)
            default_seed = "113" if dataset == "california_housing" else "42"
            if show_advanced:
                seed_raw = questionary.text(
                    f"Dataset split seed (blank = {default_seed}):", default=default_seed
                ).ask().strip()
                if seed_raw:
                    dataset_args["seed"] = int(seed_raw)
                else:
                    dataset_args["seed"] = int(default_seed)
            else:
                dataset_args["seed"] = int(default_seed)

        # Feature scaler (tabular only). Default to standard; only ask if advanced.
        if profile["ask_scaler"]:
            if show_advanced:
                scaler_choice = questionary.select(
                    "Feature scaler:", choices=["standard", "minmax", "none"], default="standard"
                ).ask()
                if scaler_choice != "none":
                    dataset_args["scaler"] = scaler_choice
            else:
                dataset_args["scaler"] = "standard"

        # Target scaler (regression only). Default to z-score; only ask if advanced.
        if profile["ask_target_scaler"]:
            if show_advanced:
                y_std = questionary.select(
                    "Standardize target y (z-score)?", choices=["yes", "no"], default="yes"
                ).ask()
                dataset_args["y_standardize"] = (y_std == "yes")
            else:
                dataset_args["y_standardize"] = True

        # Regression-only: bins for target summaries (default 10; ask only if advanced)
        distribution_bins = None
        if profile["task"] == "regression":
            default_bins = CONFIG.get("distribution_bins", 10)
            if show_advanced:
                bins_raw = questionary.text(
                    "Histogram bins for regression target summaries:",
                    default=str(default_bins),
                ).ask().strip()
                distribution_bins = int(bins_raw) if bins_raw else default_bins
            else:
                distribution_bins = default_bins

        # -- 2) Clients & partitioning (limit choices to what the dataset supports)
        num_clients = int(questionary.text(
            "Number of clients:", default=str(CONFIG.get("num_clients", 10))
        ).ask())

        strat_choices = ["iid"]
        if profile["allow_quantity_skew"]:
            strat_choices.append("quantity_skew")
        if profile["allow_label_splits"]:
            strat_choices.extend(["dirichlet", "shard", "label_per_client"])
        if profile["allow_custom"]:
            strat_choices.append("custom")

        strategy = questionary.select(
            "Partition strategy:", choices=strat_choices, default="iid",
        ).ask()

        # Only ask for a parameter when the chosen strategy needs one
        dist_param = None
        need_param = strategy in {"quantity_skew", "dirichlet", "shard", "label_per_client"}
        if need_param:
            # Supply good defaults silently unless Advanced
            defaults = {
                "quantity_skew": "1.0",   # alpha over client sizes
                "dirichlet": "0.5",       # alpha over labels
                "shard": "2",             # shards per client
                "label_per_client": "2",  # k labels per client
            }
            if show_advanced:
                help_text = {
                    "quantity_skew": "Dirichlet alpha over client sizes (e.g., 0.5, 1.0, 2.0)",
                    "dirichlet": "Dirichlet alpha over labels per client (e.g., 0.2, 0.5, 1.0)",
                    "shard": "Shards per client (integer, e.g., 2)",
                    "label_per_client": "k labels per client (integer, e.g., 2)",
                }[strategy]
                raw = questionary.text(f"{help_text}:", default=defaults[strategy]).ask().strip()
            else:
                raw = defaults[strategy]
            dist_param = float(raw) if strategy in {"quantity_skew", "dirichlet"} else int(float(raw))

        custom_path = None
        if strategy == "custom":
            custom_path = questionary.path("Path to custom distributions JSON:").ask()

        # Optional dataset subsampling (only ask in Advanced)
        sample_size = None
        sample_frac = None
        if show_advanced:
            ss_raw  = questionary.text("Sample SIZE (blank to skip):", default="").ask().strip()
            sf_raw  = questionary.text("Sample FRAC [0-1] (blank to skip):", default="").ask().strip()
            sample_size = int(ss_raw) if ss_raw else None
            sample_frac = float(sf_raw) if sf_raw else None

        # -- 3) Model & training
        # Auto default model based on dataset profile; only expose full menu if Advanced.
        if show_advanced:
            model_type = questionary.select(
                "Model type (metadata + potential model selection):",
                choices=["CNN", "MLP", "LogReg", "ResNet", "Custom"],
                default=profile["default_model"],
            ).ask()
        else:
            model_type = profile["default_model"]

        task_type = profile["task"]  # auto from dataset

        # Presets for training budget (then allow override in Advanced)
        preset = questionary.select(
            "Training preset:", choices=["Quick", "Balanced", "Thorough"], default="Balanced"
        ).ask()
        if preset == "Quick":
            preset_vals = dict(learning_rate=0.001, batch_size=32, local_epochs=3, num_rounds=5,
                            hidden_layers=[64], dropout=0.0, weight_decay=0.0, optimizer="adam")
        elif preset == "Thorough":
            preset_vals = dict(learning_rate=0.001, batch_size=64, local_epochs=15, num_rounds=100,
                            hidden_layers=[128,128], dropout=0.0, weight_decay=1e-4, optimizer="adam")
        else:  # Balanced
            preset_vals = dict(learning_rate=0.001, batch_size=64, local_epochs=5, num_rounds=20,
                            hidden_layers=[128], dropout=0.0, weight_decay=0.0, optimizer="adam")

        # If Advanced, unlock all knobs; otherwise use sensible defaults
        if show_advanced:
            hidden_layers = questionary.text(
                "Hidden layers (comma-separated, e.g., 128,64):",
                default=",".join(str(u) for u in preset_vals["hidden_layers"])
            ).ask()
            hidden_layers = [int(x) for x in hidden_layers.split(",") if x.strip()]

            activation = questionary.select(
                "Activation:", choices=["relu","tanh","sigmoid","gelu","elu","selu","softplus","linear"],
                default="relu"
            ).ask()
            optimizer  = questionary.select(
                "Optimizer:", choices=["adam","adamw","sgd","rmsprop","adagrad"], default=preset_vals["optimizer"]
            ).ask()
            dropout    = float(questionary.text("Dropout (0.0-0.9):", default=str(preset_vals["dropout"])).ask())
            weight_decay = float(questionary.text("Weight decay L2 (e.g., 0.0, 1e-4):",
                                                default=str(preset_vals["weight_decay"])).ask())
            learning_rate   = float(questionary.text("Learning rate:", default=str(preset_vals["learning_rate"])).ask())
            batch_size      = int(questionary.text("Batch size:", default=str(preset_vals["batch_size"])).ask())
            epochs_per_round= int(questionary.text("Epochs per round:", default=str(preset_vals["local_epochs"])).ask())
            num_rounds      = int(questionary.text("Global rounds:", default=str(preset_vals["num_rounds"])).ask())
        else:
            hidden_layers  = preset_vals["hidden_layers"]
            activation     = "relu"
            optimizer      = preset_vals["optimizer"]
            dropout        = preset_vals["dropout"]
            weight_decay   = preset_vals["weight_decay"]
            learning_rate  = preset_vals["learning_rate"]
            batch_size     = preset_vals["batch_size"]
            epochs_per_round = preset_vals["local_epochs"]
            num_rounds     = preset_vals["num_rounds"]

        # Client dropout rate (ask only in Advanced; default 0.0)
        if show_advanced:
            cdr_raw = questionary.text("Client dropout rate per round (0.0–1.0):", default="0.0").ask().strip()
            client_dropout_rate = float(cdr_raw) if cdr_raw else 0.0
        else:
            client_dropout_rate = 0.0

        # Reproducibility (ask only in Advanced)
        seed = None
        if show_advanced:
            seed_raw = questionary.text("Random seed (blank = none):", default="").ask().strip()
            seed = int(seed_raw) if seed_raw else None

        # Save weights? (ask only in Advanced; default False to keep runs light)
        save_weights = questionary.confirm("Save per-round weight files?", default=False).ask() if show_advanced else False

        # Output filename (always ask)
        output = questionary.text("Output CSV filename:", default="clients.csv").ask()

        # -- 4) Build config
        config = {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": epochs_per_round,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_layers": hidden_layers,
            "activation": activation,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "distribution_type": strategy,
            "distribution_param": dist_param,
            "sample_size": sample_size,
            "sample_frac": sample_frac,
            "model_type": model_type,
            "task_type": profile["task"],           # auto from dataset
            "client_dropout_rate": client_dropout_rate,
            "save_weights": bool(save_weights),
            "dataset": dataset,
            "dataset_args": dataset_args,
        }
        if distribution_bins is not None:
            config["distribution_bins"] = distribution_bins
        if seed is not None:
            config["seed"] = seed
        if strategy == "custom":
            import json as _json
            with open(custom_path, "r") as f:
                config["custom_distributions"] = _json.load(f)

        # -- 5) Confirm & save
        print("\n=== Run Summary ===")
        import pprint; pprint.pprint({**config, "output": output})
        if not questionary.confirm("Proceed?").ask():
            print("Aborted.")
            return

        import json as _json 
        with open(args.save, "w") as f:
            _json.dump({**config, "output": output}, f, indent=2)
        print(f"Saved configuration to {args.save}")

        # -- 6) Run
        gen = FederatedDataGenerator(config=config, dataset=dataset)
        df = gen.run()
        out_path = resolve_output_path(output, kind="run")
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} client records to {out_path}")
        return


    parser.print_help()

if __name__ == "__main__":
    main()
