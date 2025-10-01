# cli/cmd_wizard.py
from __future__ import annotations
import argparse
import sqlite3
import pandas as pd
from ..config import CONFIG
from ..path_resolver import resolve_table_output_paths
from ..federated.orchestrator import FederatedDataGenerator
from .profiles import DATASET_CHOICES, infer_dataset_profile

def _handle(args: argparse.Namespace) -> None:
    import questionary
    import json as _json
    import pprint

    # Pretty → canonical model_type mapping used by builders
    MODEL_MAP = {
        "CNN": "cnn",
        "MobileNetV2": "mobilenetv2",
        "MLP": "mlp",
        "Logistic Regression": "logreg",
        "Random Forest": "randomforest",
    }

    dataset = questionary.select("Select dataset:", choices=DATASET_CHOICES, default="fashion_mnist").ask()
    profile = infer_dataset_profile(dataset)
    show_advanced = questionary.confirm("Show advanced options?", default=False).ask()

    # Task choice constrained by dataset profile
    task_type = questionary.select(
        "Select task type:",
        choices=profile["tasks_supported"],
        default=profile["default_task"],
    ).ask()

    dataset_args = {}
    if profile["ask_split"]:
        default_split = "0.2"
        if show_advanced:
            split_raw = questionary.text(
                f"Test split fraction (default {default_split}):", default=default_split
            ).ask().strip()
            if split_raw:
                dataset_args[profile["split_key"]] = float(split_raw)
        else:
            dataset_args[profile["split_key"]] = float(default_split)

        default_seed = "113" if dataset == "california_housing" else "42"
        if show_advanced:
            seed_raw = questionary.text(
                f"Dataset split seed (blank = {default_seed}):", default=default_seed
            ).ask().strip()
            dataset_args["seed"] = int(seed_raw) if seed_raw else int(default_seed)
        else:
            dataset_args["seed"] = int(default_seed)

    if profile["ask_scaler"]:
        if show_advanced:
            scaler_choice = questionary.select(
                "Feature scaler:", choices=["standard", "minmax", "none"], default="standard"
            ).ask()
            if scaler_choice != "none":
                dataset_args["scaler"] = scaler_choice
        else:
            dataset_args["scaler"] = "standard"

    if profile["ask_target_scaler"] and task_type == "regression":
        if show_advanced:
            y_std = questionary.select(
                "Standardize target y (z-score)?", choices=["yes", "no"], default="yes"
            ).ask()
            dataset_args["y_standardize"] = (y_std == "yes")
        else:
            dataset_args["y_standardize"] = True

    # Bins for summaries (always allowed; use default if not shown)
    distribution_bins = CONFIG.get("distribution_bins", 10)
    if show_advanced:
        if task_type == "regression":
            bins_raw = questionary.text(
                "Histogram bins for regression target summaries:",
                default=str(distribution_bins),
            ).ask().strip()
            distribution_bins = int(bins_raw) if bins_raw else distribution_bins

    # Clients & partitioning
    num_clients = int(questionary.text(
        "Number of clients:", default=str(CONFIG.get("num_clients", 10))
    ).ask())

    # Partition strategy (restrict for regression)
    strat_choices = ["iid"]
    if profile["allow_quantity_skew"]:
        strat_choices.append("quantity_skew")
    if task_type != "regression" and profile["allow_label_splits"]:
        strat_choices.extend(["dirichlet", "shard", "label_per_client"])
    if profile["allow_custom"]:
        strat_choices.append("custom")

    strategy = questionary.select(
        "Partition strategy:", choices=strat_choices, default="iid",
    ).ask()

    dist_param = None
    need_param = strategy in {"quantity_skew", "dirichlet", "shard", "label_per_client"}
    if need_param:
        defaults = {
            "quantity_skew": "1.0",
            "dirichlet": "0.5",
            "shard": "2",
            "label_per_client": "2",
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

    # Optional dataset subsampling (Advanced only)
    sample_size = None
    sample_frac = None
    if show_advanced:
        ss_raw  = questionary.text("Sample SIZE (blank to skip):", default="").ask().strip()
        sf_raw  = questionary.text("Sample FRAC [0-1] (blank to skip):", default="").ask().strip()
        sample_size = int(ss_raw) if ss_raw else None
        sample_frac = float(sf_raw) if sf_raw else None

    # -------------------------
    # Model & training knobs
    # -------------------------
    rf_trees = rf_max_depth = mobilenet_trainable = None

    if task_type in {"classification", "regression"}:
        # Decide model choices from profile flags
        is_image = profile.get("is_image", False)

        if is_image and task_type == "classification":
            model_choices_pretty = ["CNN", "MobileNetV2", "MLP"]
        elif is_image and task_type == "regression":
            # uncommon, keep simple
            model_choices_pretty = ["MLP"]
        else:
            # tabular
            if task_type == "classification":
                model_choices_pretty = ["MLP", "Logistic Regression", "Random Forest"]
            else:
                # regression (tabular)
                model_choices_pretty = ["Random Forest", "MLP"]  # RF first; matches profile default

        default_pretty = profile.get("default_model_pretty")
        if default_pretty not in model_choices_pretty:
            # fall back if profile default isn't compatible with task/dataset choice
            default_pretty = model_choices_pretty[0]

        model_pretty = questionary.select(
            "Model family:",
            choices=model_choices_pretty,
            default=default_pretty,
        ).ask()

        model_type = MODEL_MAP[model_pretty]

        # Presets
        preset = questionary.select(
            "Training preset:",
            choices=["Quick", "Balanced", "Thorough"],
            default="Balanced"
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
            dropout      = float(questionary.text("Dropout (0.0-0.9):", default=str(preset_vals["dropout"])).ask())
            weight_decay = float(questionary.text("Weight decay L2 (e.g., 0.0, 1e-4):",
                                                default=str(preset_vals["weight_decay"])).ask())
            learning_rate   = float(questionary.text("Learning rate:", default=str(preset_vals["learning_rate"])).ask())
            batch_size      = int(questionary.text("Batch size:", default=str(preset_vals["batch_size"])).ask())
            epochs_per_round= int(questionary.text("Epochs per round:", default=str(preset_vals["local_epochs"])).ask())
            num_rounds      = int(questionary.text("Global rounds:", default=str(preset_vals["num_rounds"])).ask())
        else:
            hidden_layers   = preset_vals["hidden_layers"]
            activation      = "relu"
            optimizer       = preset_vals["optimizer"]
            dropout         = preset_vals["dropout"]
            weight_decay    = preset_vals["weight_decay"]
            learning_rate   = preset_vals["learning_rate"]
            batch_size      = preset_vals["batch_size"]
            epochs_per_round= preset_vals["local_epochs"]
            num_rounds      = int(questionary.text("Global rounds:", default=str(preset_vals["num_rounds"])).ask())

        # Model-specific extras (advanced only)
        if show_advanced:
            if model_type == "randomforest":
                rf_trees = int(questionary.text("RandomForest: n_estimators:", default="100").ask() or "100")
                md_raw = questionary.text("RandomForest: max_depth (blank = None):", default="").ask().strip()
                rf_max_depth = int(md_raw) if md_raw else None
            if model_type == "mobilenetv2":
                mobilenet_trainable = questionary.confirm("Fine-tune MobileNetV2 backbone (trainable)?", default=False).ask()

        # Client dropout
        client_dropout_rate = 0.0
        if show_advanced:
            cdr_raw = questionary.text("Client dropout rate per round (0.0–1.0):", default="0.0").ask().strip()
            client_dropout_rate = float(cdr_raw) if cdr_raw else 0.0

        save_weights = questionary.confirm("Save per-round weight files?", default=False).ask()

    else:
        # Clustering path
        model_type = profile["default_model_pretty"]  # not used for clustering; kept for metadata
        # Rounds
        preset_num_rounds = CONFIG.get("num_rounds", 20)
        num_rounds = int(questionary.text("Global rounds:", default=str(preset_num_rounds)).ask())
        # KMeans hyperparams
        clustering_k = clustering_init = clustering_n_init = clustering_max_iter = clustering_tol = None
        if show_advanced:
            clustering_k = int(questionary.text("KMeans: number of clusters k:", default="10").ask() or "10")
            clustering_init = questionary.select("KMeans init:", choices=["k-means++","random"], default="k-means++").ask()
            clustering_n_init = int(questionary.text("KMeans n_init:", default="10").ask() or "10")
            clustering_max_iter = int(questionary.text("KMeans max_iter:", default="300").ask() or "300")
            clustering_tol = float(questionary.text("KMeans tol:", default="1e-4").ask() or "1e-4")

        # No-op knobs for supervised path
        learning_rate = None
        batch_size = None
        epochs_per_round = 1
        hidden_layers = []
        activation = "relu"
        optimizer = "adam"
        dropout = 0.0
        weight_decay = 0.0
        client_dropout_rate = 0.0
        save_weights = False

    # Global seed
    seed = None
    if show_advanced:
        seed_raw = questionary.text("Random seed (blank = none):", default="").ask().strip()
        seed = int(seed_raw) if seed_raw else None

    # Build config
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
        "model_type": MODEL_MAP.get(model_type, model_type) if isinstance(model_type, str) else "mlp",
        "task_type": task_type,
        "client_dropout_rate": client_dropout_rate,
        "save_weights": bool(save_weights),
        "dataset": dataset,
        "dataset_args": dataset_args,
        "distribution_bins": distribution_bins,
    }

    if seed is not None:
        config["seed"] = seed
    if strategy == "custom":
        with open(custom_path, "r") as f:
            config["custom_distributions"] = _json.load(f)

    # Model-specific extras
    if rf_trees is not None:             config["rf_trees"] = rf_trees
    if rf_max_depth is not None:         config["rf_max_depth"] = rf_max_depth
    if mobilenet_trainable is not None:  config["mobilenet_trainable"] = mobilenet_trainable

    # Clustering extras
    if task_type == "clustering":
        if show_advanced:
            if clustering_k is not None:        config["clustering_k"] = clustering_k
            if clustering_init is not None:     config["clustering_init"] = clustering_init
            if clustering_n_init is not None:   config["clustering_n_init"] = clustering_n_init
            if clustering_max_iter is not None: config["clustering_max_iter"] = clustering_max_iter
            if clustering_tol is not None:      config["clustering_tol"] = clustering_tol

    print("\n=== Run Summary ===")
    pprint.pprint(config)
    if not questionary.confirm("Proceed?").ask():
        print("Aborted."); return

    with open(args.save, "w") as f:
        _json.dump(config, f, indent=2)
    print(f"Saved configuration to {args.save}")

    gen = FederatedDataGenerator(config=config, dataset=dataset)
    summary = gen.run()
    run_id = summary["run_id"]
    db_path = summary["db_path"]

    con = sqlite3.connect(db_path)
    try:
        runs_df = pd.read_sql_query(
            "SELECT * FROM runs WHERE run_id = ?",
            con, params=[run_id]
        )
        rounds_df = pd.read_sql_query(
            "SELECT * FROM rounds WHERE run_id = ? ORDER BY round",
            con, params=[run_id]
        )
        clients_df = pd.read_sql_query(
            "SELECT * FROM client_rounds WHERE run_id = ? ORDER BY round, client_id",
            con, params=[run_id]
        )
    finally:
        con.close()

    runs_df.to_csv("outputs/runs.csv", index=False)
    rounds_df.to_csv("outputs/rounds.csv", index=False)
    clients_df.to_csv("outputs/client_rounds.csv", index=False)  # fixed extension
    print("Wrote tables:")
    print("  runs -> outputs/runs.csv")
    print("  rounds -> outputs/rounds.csv")
    print("  client_rounds -> outputs/client_rounds.csv")

def register_wizard(subparsers):
    p = subparsers.add_parser("wizard", help="Interactive setup & run")
    p.add_argument("--save", type=str, default="run_config.json", help="Where to save the chosen config (JSON)")
    p.set_defaults(_handler=_handle)
