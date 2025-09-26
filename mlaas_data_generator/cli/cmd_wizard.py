# cli/cmd_wizard.py
from __future__ import annotations
import argparse
from ..config import CONFIG
from ..path_resolver import resolve_output_path
from ..federated.orchestrator import FederatedDataGenerator
from .profiles import DATASET_CHOICES, infer_dataset_profile

def _handle(args: argparse.Namespace) -> None:
    # Lazy import so the rest of CLI doesn’t require questionary
    import questionary

    dataset = questionary.select("Select dataset:", choices=DATASET_CHOICES, default="fashion_mnist").ask()
    profile = infer_dataset_profile(dataset)
    show_advanced = questionary.confirm("Show advanced options?", default=False).ask()

    dataset_args = {}
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
        "task_type": profile["task"],
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

def register_wizard(subparsers):
    p = subparsers.add_parser("wizard", help="Interactive setup & run")
    p.add_argument("--save", type=str, default="run_config.json", help="Where to save the chosen config (JSON)")
    p.set_defaults(_handler=_handle)
