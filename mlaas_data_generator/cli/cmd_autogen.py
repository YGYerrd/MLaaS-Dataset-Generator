# cli/cmd_autogen.py
from __future__ import annotations
import argparse, math, random, sqlite3, json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from ..config import CONFIG
from ..federated.orchestrator import FederatedDataGenerator

# --------------------------
# Study spec (datasets & compat)
# --------------------------
IMG_CLASS_DATASETS = ["mnist", "fashion_mnist", "cifar10"]
TAB_CLASS_DATASETS = ["digits", "iris", "wine"]
REGRESSION_DATASETS = ["california_housing", "diabetes"]
ALL_DATASETS = IMG_CLASS_DATASETS + TAB_CLASS_DATASETS + REGRESSION_DATASETS

COMPAT = {
    "classification": set(IMG_CLASS_DATASETS + TAB_CLASS_DATASETS),
    "regression": set(REGRESSION_DATASETS),
    "clustering": set(ALL_DATASETS),
}

MODEL_MAP = {
    "cnn": "cnn",
    "mobilenetv2": "mobilenetv2",
    "mlp": "mlp",
    "logreg": "logreg",
    "randomforest": "randomforest",
}

# --------------------------
# Helpers
# --------------------------
def _task_counts(total: int, split_csv: str) -> Dict[str, int]:
    # split_csv like "50,30,20" => classification, regression, clustering %
    c_pct, r_pct, k_pct = [float(x) for x in split_csv.split(",")]
    # normalize in case they don't sum to 100 exactly
    s = c_pct + r_pct + k_pct
    c = round(total * (c_pct / s))
    r = round(total * (r_pct / s))
    k = total - c - r
    return {"classification": c, "regression": r, "clustering": k}

def _rng(seed: int | None):
    return np.random.default_rng(seed if seed is not None else 42)

def _pick(seq, rng, p=None):
    if p is None:
        return seq[int(rng.integers(0, len(seq)))]
    return seq[int(rng.choice(len(seq), p=p))]

def _log_uniform(rng, a, b):
    # sample in log space
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))

# --------------------------
# Dataset allocation
# --------------------------
def allocate_runs(total: int, task_counts: Dict[str, int], rng) -> List[Tuple[str, str]]:
    """
    Return a list of (dataset, task) pairs totaling 'total' runs.
    We aim for roughly even dataset presence while satisfying task counts and compat.
    """
    pairs: List[Tuple[str, str]] = []

    # Build per-task pools of compatible datasets
    pools = {t: sorted(list(COMPAT[t])) for t in task_counts}

    # Start with uniform assignment by task → datasets
    for task, count in task_counts.items():
        ds_list = pools[task]
        base = count // len(ds_list)
        rem = count - base * len(ds_list)
        # everyone gets 'base', then distribute the remainder randomly
        for ds in ds_list:
            pairs.extend([(ds, task)] * base)
        if rem > 0:
            idxs = rng.choice(len(ds_list), size=rem, replace=False)
            for i in idxs:
                pairs.append((ds_list[int(i)], task))

    # We now have exact per-task totals; dataset totals may be a bit uneven (good enough).
    rng.shuffle(pairs)
    assert len(pairs) == sum(task_counts.values()) == total
    return pairs

# --------------------------
# Samplers for knobs
# --------------------------
def sample_distribution_knobs(task: str, rng):
    # Regression: avoid label-based partitions
    if task == "regression":
        strat = _pick(["iid", "quantity_skew"], rng, p=[0.8, 0.2])
        if strat == "quantity_skew":
            return strat, float(_pick([0.5, 1.0, 2.0], rng))
        return strat, None

    # Classification/Clustering
    strat = _pick(["iid", "dirichlet", "shard", "quantity_skew"], rng, p=[0.4, 0.3, 0.2, 0.1])
    if strat == "dirichlet":
        return strat, float(_pick([0.1, 0.5, 1.0], rng))
    if strat == "shard":
        return strat, int(_pick([2, 3, 4, 5], rng))
    if strat == "quantity_skew":
        return strat, float(_pick([0.5, 1.0, 2.0], rng))
    return strat, None

def sample_common_federated_knobs(task: str, dataset: str, rng):
    # clients
    num_clients = int(_pick([5, 10, 20, 50], rng, p=[0.15, 0.45, 0.35, 0.05]))
    # rounds
    if task == "clustering":
        num_rounds = int(_pick([5, 10], rng, p=[0.5, 0.5]))
    else:
        num_rounds = int(_pick([5, 10, 20], rng, p=[0.3, 0.45, 0.25]))
    # epochs per round
    local_epochs = int(_pick([1, 2, 3, 5], rng, p=[0.15, 0.4, 0.3, 0.15]))
    # batch size depends on dataset type
    if dataset in ["cifar10"]:
        batch_size = int(_pick([32, 64, 128], rng, p=[0.2, 0.6, 0.2]))
        local_epochs = min(local_epochs, 5)
    elif dataset in IMG_CLASS_DATASETS:
        batch_size = int(_pick([32, 64, 128], rng, p=[0.3, 0.5, 0.2]))
    else:
        batch_size = int(_pick([16, 32, 64], rng, p=[0.3, 0.5, 0.2]))

    # learning rate & optimizer later adjusted by model; seed omitted here (assigned in dataset_args)
    client_dropout_rate = float(np.clip(rng.uniform(0.0, 0.3), 0.0, 0.3))
    save_weights = bool(rng.uniform() < 0.1)  # only sometimes to save space
    return num_clients, num_rounds, local_epochs, batch_size, client_dropout_rate, save_weights

def sample_model_and_training(task: str, dataset: str, rng):
    # Defaults
    optimizer = _pick(["adam", "rmsprop", "sgd"], rng, p=[0.5, 0.2, 0.3])
    activation = _pick(["relu", "tanh", "gelu"], rng, p=[0.8, 0.1, 0.1])
    hidden_layers = [128]
    dropout = _pick([0.0, 0.2, 0.5], rng, p=[0.5, 0.35, 0.15])
    weight_decay = _pick([0.0, 1e-4, 1e-3], rng, p=[0.6, 0.3, 0.1])

    if task == "classification" and dataset in IMG_CLASS_DATASETS:
        # images
        if dataset == "cifar10":
            model_type = _pick(["mobilenetv2", "cnn", "mlp"], rng, p=[0.6, 0.35, 0.05])
        else:
            model_type = _pick(["cnn", "mlp", "mobilenetv2"], rng, p=[0.7, 0.2, 0.1])
        if model_type in {"adam", "rmsprop"}:
            pass  # no-op; typo guard
        lr = _log_uniform(rng, 1e-4, 3e-3) if optimizer in {"adam", "rmsprop"} else _log_uniform(rng, 3e-4, 3e-2)
        if model_type == "mlp":
            depth = int(_pick([1, 2, 3], rng, p=[0.5, 0.35, 0.15]))
            hidden_layers = [int(_pick([64, 128, 256], rng)) for _ in range(depth)]
        return MODEL_MAP[model_type], float(lr), optimizer, activation, hidden_layers, float(dropout), float(weight_decay)

    if task == "classification" and dataset in TAB_CLASS_DATASETS:
        model_type = _pick(["mlp", "logreg", "randomforest"], rng, p=[0.6, 0.2, 0.2])
        lr = _log_uniform(rng, 1e-4, 1e-2) if optimizer in {"adam", "rmsprop"} else _log_uniform(rng, 3e-4, 3e-2)
        if model_type == "mlp":
            depth = int(_pick([1, 2, 3], rng, p=[0.5, 0.35, 0.15]))
            hidden_layers = [int(_pick([32, 64, 128, 256], rng)) for _ in range(depth)]
        return MODEL_MAP[model_type], float(lr), optimizer, activation, hidden_layers, float(dropout), float(weight_decay)

    if task == "regression":
        model_type = _pick(["randomforest", "mlp"], rng, p=[0.5, 0.5])
        if model_type == "mlp":
            lr = _log_uniform(rng, 1e-4, 1e-1)
            depth = int(_pick([1, 2, 3], rng, p=[0.5, 0.35, 0.15]))
            hidden_layers = [int(_pick([64, 128, 256], rng)) for _ in range(depth)]
        else:
            lr = None  # not used for RF adapter
        return MODEL_MAP[model_type], (None if model_type == "randomforest" else float(lr)), optimizer, activation, hidden_layers, float(dropout), float(weight_decay)

    # clustering: model_type is metadata-only; keep "mlp" to satisfy schema
    model_type = "mlp"
    # training knobs for clustering are ignored in your strategies; keep reasonable defaults
    lr = 1e-3
    return MODEL_MAP[model_type], float(lr), optimizer, activation, [128], 0.0, 0.0

def sample_dataset_args(task: str, dataset: str, rng):
    args = {}
    # tabular datasets use explicit split; keras image datasets ignore and provide their own
    if dataset in TAB_CLASS_DATASETS + REGRESSION_DATASETS:
        args["test_size"] = 0.2
        args["seed"] = int(rng.integers(1, 10_000))
        # scaling
        if task == "regression":
            args["scaler"] = _pick(["standard", "minmax"], rng, p=[0.9, 0.1])
            args["y_standardize"] = bool(rng.uniform() < 0.7)
        else:
            # classification/clustering tabular: often standardize
            args["scaler"] = _pick(["standard", "minmax", "none"], rng, p=[0.8, 0.15, 0.05])
    return args

def sample_clustering_knobs(dataset: str, rng):
    # choose k around ground-truth where known
    gt = {"mnist":10, "fashion_mnist":10, "cifar10":10, "digits":10, "iris":3, "wine":3}.get(dataset, None)
    if gt is None:
        k = int(_pick([3, 5, 8, 10], rng))
    else:
        cand = [max(2, gt + d) for d in [-2, -1, 0, 1, 2]]
        k = int(_pick(cand, rng))
    init = _pick(["k-means++", "random"], rng, p=[0.8, 0.2])
    n_init = int(_pick([10, 20], rng))
    max_iter = int(_pick([200, 300, 500], rng, p=[0.2, 0.6, 0.2]))
    tol = float(_pick([1e-4, 1e-3], rng, p=[0.7, 0.3]))
    return dict(clustering_k=k, clustering_init=init, clustering_n_init=n_init,
                clustering_max_iter=max_iter, clustering_tol=tol)

def compose_config(dataset: str, task: str, rng):
    # federated knobs
    num_clients, num_rounds, local_epochs, batch_size, client_dropout_rate, save_weights = \
        sample_common_federated_knobs(task, dataset, rng)

    # distribution
    distribution_type, distribution_param = sample_distribution_knobs(task, rng)

    # model & training
    model_type, learning_rate, optimizer, activation, hidden_layers, dropout, weight_decay = \
        sample_model_and_training(task, dataset, rng)

    # dataset args
    dataset_args = sample_dataset_args(task, dataset, rng)

    # optional sampling
    sample_size = None
    sample_frac = (0.25 if rng.uniform() < 0.08 else (0.5 if rng.uniform() < 0.04 else None))

    cfg = {
        "dataset": dataset,
        "task_type": task,
        "model_type": model_type,
        "dataset_args": dataset_args,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "optimizer": optimizer,
        "distribution_type": distribution_type,
        "distribution_param": distribution_param,
        "client_dropout_rate": client_dropout_rate,
        "sample_size": sample_size,
        "sample_frac": sample_frac,
        "save_weights": bool(save_weights),
        "distribution_bins": CONFIG.get("distribution_bins", 10),
    }

    if task == "clustering":
        cfg.update(sample_clustering_knobs(dataset, rng))

    # seed: let orchestrator accept/run; keep here for reproducibility
    if rng.uniform() < 0.9:
        cfg["seed"] = int(rng.integers(1, 10_000_000))
    return cfg

# --------------------------
# Driver
# --------------------------
def _handle(args: argparse.Namespace) -> None:
    total_runs = int(args.runs)
    task_counts = _task_counts(total_runs, args.task_split)
    rng = _rng(args.seed)

    # Build (dataset, task) plan
    plan = allocate_runs(total_runs, task_counts, rng)

    manifest: List[dict] = []
    for idx, (dataset, task) in enumerate(plan, start=1):
        cfg = compose_config(dataset, task, rng)
        # run it
        gen = FederatedDataGenerator(config=cfg, dataset=dataset)
        summary = gen.run()

        # collect quick outputs (like wizard)
        run_id = summary["run_id"]
        db_path = summary["db_path"]
        con = sqlite3.connect(db_path)
        try:
            runs_df   = pd.read_sql_query("SELECT * FROM runs WHERE run_id = ?", con, params=[run_id])
            rounds_df = pd.read_sql_query("SELECT * FROM rounds WHERE run_id = ? ORDER BY round", con, params=[run_id])
            clients_df= pd.read_sql_query("SELECT * FROM client_rounds WHERE run_id = ? ORDER BY round, client_id", con, params=[run_id])
        finally:
            con.close()

        # persist per-run CSVs (optional); also keep/append a global manifest
        runs_df.to_csv(f"outputs/runs_{idx:04d}.csv", index=False)
        rounds_df.to_csv(f"outputs/rounds_{idx:04d}.csv", index=False)
        clients_df.to_csv(f"outputs/client_rounds_{idx:04d}.csv", index=False)

        manifest.append({
            "idx": idx, "run_id": run_id, "dataset": dataset, "task_type": task,
            "config": cfg
        })

    # write a manifest so the study is reproducible/inspectable
    with open("outputs/study_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Wrote manifest for {len(manifest)} runs to outputs/study_manifest.json")

def register_autogen(subparsers):
    p = subparsers.add_parser("autogen", help="Automatically generate many runs")
    p.add_argument("--runs", type=int, default=50, help="Total runs to generate")
    p.add_argument("--task-split", type=str, default="50,30,20", help="classification,regression,clustering percentages")
    p.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    p.set_defaults(_handler=_handle)
