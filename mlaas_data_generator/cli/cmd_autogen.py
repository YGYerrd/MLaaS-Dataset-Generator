# cli/cmd_autogen.py
import argparse, sqlite3, json
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
def _task_counts(total, split_csv):
    c_pct, r_pct, k_pct = [float(x) for x in split_csv.split(",")]
    s = c_pct + r_pct + k_pct
    c = round(total * (c_pct / s))
    r = round(total * (r_pct / s))
    k = total - c - r
    return {"classification": c, "regression": r, "clustering": k}

def _rng(seed):
    return np.random.default_rng(seed if seed is not None else 42)

def _pick(seq, rng, p=None):
    return seq[int(rng.choice(len(seq), p=p))] if p else seq[int(rng.integers(0, len(seq)))]

def _log_uniform(rng, a, b):
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))

# --------------------------
# Structured allocation
# --------------------------
def _dataset_modality(dataset):
    return "image" if dataset in IMG_CLASS_DATASETS else "tabular"

def _allowed_models(task, dataset):
    modality = _dataset_modality(dataset)
    if task == "clustering":
        return ["mlp"]  # clustering uses adapter; metadata-only
    if task == "regression":
        return ["mlp", "randomforest"] if modality == "tabular" else ["mlp"]
    if modality == "image":
        return ["cnn", "mobilenetv2", "mlp"]
    return ["mlp", "randomforest", "logreg"]

def _task_distributions(task):
    return ["iid", "quantity_skew"] if task == "regression" else ["iid", "dirichlet", "quantity_skew", "shard"]

def _distribute_evenly(total, items, rng):
    if not items or total <= 0:
        return {item: 0 for item in items}
    base = total // len(items)
    counts = {item: base for item in items}
    rem = total - base * len(items)
    if rem > 0:
        order = list(rng.permutation(len(items)))
        for idx in order[:rem]:
            counts[items[int(idx)]] += 1
    return counts

def _structured_distribution_param(distribution, rng):
    if distribution == "dirichlet":
        return float(_pick([0.1, 0.5, 1.0], rng))
    if distribution == "quantity_skew":
        return float(_pick([0.5, 1.0, 2.0], rng))
    if distribution == "shard":
        return int(_pick([2, 3, 4, 5], rng))
    return None

def build_structured_plan(total_runs, task_counts, rng):
    plan = []
    for task, count in task_counts.items():
        if count <= 0:
            continue
        datasets = sorted(list(COMPAT[task]))
        dataset_counts = _distribute_evenly(count, datasets, rng)
        for dataset, ds_count in dataset_counts.items():
            if ds_count <= 0:
                continue
            models = _allowed_models(task, dataset)
            model_counts = _distribute_evenly(ds_count, models, rng)
            distributions = _task_distributions(task)
            for model, model_count in model_counts.items():
                if model_count <= 0:
                    continue
                dist_choices = list(distributions)
                if len(dist_choices) > 1:
                    perm = list(rng.permutation(len(dist_choices)))
                    dist_choices = [dist_choices[int(i)] for i in perm]
                dist_choices = dist_choices[:min(2, len(dist_choices))]
                optimizers = ["adam", "sgd"]
                for i in range(model_count):
                    dist = dist_choices[i % len(dist_choices)]
                    plan.append({
                        "task": task,
                        "dataset": dataset,
                        "model_type": model,
                        "distribution": dist,
                        "optimizer": optimizers[i % 2],
                        "distribution_param": _structured_distribution_param(dist, rng),
                        "group_id": f"{model}_{dataset}_{dist}",
                    })
    assert len(plan) == total_runs, f"Planned {len(plan)} runs, expected {total_runs}"
    return plan

# --------------------------
# Samplers for knobs (unchanged)
# --------------------------
def sample_distribution_knobs(task, rng):
    if task == "regression":
        strat = _pick(["iid", "quantity_skew"], rng, p=[0.8, 0.2])
        if strat == "quantity_skew":
            return strat, float(_pick([0.5, 1.0, 2.0], rng))
        return strat, None
    strat = _pick(["iid", "dirichlet", "shard", "quantity_skew"], rng, p=[0.4, 0.3, 0.2, 0.1])
    if strat == "dirichlet":
        return strat, float(_pick([0.1, 0.5, 1.0], rng))
    if strat == "shard":
        return strat, int(_pick([2, 3, 4, 5], rng))
    if strat == "quantity_skew":
        return strat, float(_pick([0.5, 1.0, 2.0], rng))
    return strat, None

def sample_common_federated_knobs(task, dataset, rng):
    num_clients = int(_pick([5, 10, 20, 50], rng, p=[0.15, 0.45, 0.35, 0.05]))
    if task == "clustering":
        num_rounds = int(_pick([5, 10], rng, p=[0.5, 0.5]))
    else:
        num_rounds = int(_pick([5, 10, 20], rng, p=[0.3, 0.45, 0.25]))
    local_epochs = int(_pick([1, 2, 3, 5], rng, p=[0.15, 0.4, 0.3, 0.15]))
    if dataset in ["cifar10"]:
        batch_size = int(_pick([32, 64, 128], rng, p=[0.2, 0.6, 0.2]))
        local_epochs = min(local_epochs, 5)
    elif dataset in IMG_CLASS_DATASETS:
        batch_size = int(_pick([32, 64, 128], rng, p=[0.3, 0.5, 0.2]))
    else:
        batch_size = int(_pick([16, 32, 64], rng, p=[0.3, 0.5, 0.2]))
    client_dropout_rate = float(np.clip(rng.uniform(0.0, 0.3), 0.0, 0.3))
    save_weights = bool(rng.uniform() < 0.1)
    return num_clients, num_rounds, local_epochs, batch_size, client_dropout_rate, save_weights

def sample_model_and_training(task, dataset, rng):
    optimizer = _pick(["adam", "rmsprop", "sgd"], rng, p=[0.5, 0.2, 0.3])
    activation = _pick(["relu", "tanh", "gelu"], rng, p=[0.8, 0.1, 0.1])
    hidden_layers = [128]
    dropout = _pick([0.0, 0.2, 0.5], rng, p=[0.5, 0.35, 0.15])
    weight_decay = _pick([0.0, 1e-4, 1e-3], rng, p=[0.6, 0.3, 0.1])

    if task == "classification" and dataset in IMG_CLASS_DATASETS:
        if dataset == "cifar10":
            model_type = _pick(["mobilenetv2", "cnn", "mlp"], rng, p=[0.6, 0.35, 0.05])
        else:
            model_type = _pick(["cnn", "mlp", "mobilenetv2"], rng, p=[0.7, 0.2, 0.1])
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
            lr = None
        return MODEL_MAP[model_type], (None if model_type == "randomforest" else float(lr)), optimizer, activation, hidden_layers, float(dropout), float(weight_decay)

    model_type = "mlp"  # clustering metadata
    lr = 1e-3
    return MODEL_MAP[model_type], float(lr), optimizer, activation, [128], 0.0, 0.0

def sample_dataset_args(task, dataset, rng):
    args = {}
    if dataset in TAB_CLASS_DATASETS + REGRESSION_DATASETS:
        args["test_size"] = 0.2
        args["seed"] = int(rng.integers(1, 10_000))
        if task == "regression":
            args["scaler"] = _pick(["standard", "minmax"], rng, p=[0.9, 0.1])
            args["y_standardize"] = bool(rng.uniform() < 0.7)
        else:
            args["scaler"] = _pick(["standard", "minmax", "none"], rng, p=[0.8, 0.15, 0.05])
    return args

def sample_clustering_knobs(dataset, rng):
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

def compose_config(dataset, task, rng):
    num_clients, num_rounds, local_epochs, batch_size, client_dropout_rate, save_weights = \
        sample_common_federated_knobs(task, dataset, rng)

    distribution_type, distribution_param = sample_distribution_knobs(task, rng)

    model_type, learning_rate, optimizer, activation, hidden_layers, dropout, weight_decay = \
        sample_model_and_training(task, dataset, rng)

    dataset_args = sample_dataset_args(task, dataset, rng)

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

    if rng.uniform() < 0.9:
        cfg["seed"] = int(rng.integers(1, 10_000_000))
    return cfg

# --------------------------
# Driver
# --------------------------
def _handle(args):
    total_runs = int(args.runs)
    task_counts = _task_counts(total_runs, args.task_split)
    rng = _rng(args.seed)

    plan = build_structured_plan(total_runs, task_counts, rng)

    manifest = []
    for idx, plan_item in enumerate(plan, start=1):
        dataset = plan_item["dataset"]
        task = plan_item["task"]

        cfg = compose_config(dataset, task, rng)
        cfg.update({
            "model_type": plan_item["model_type"],
            "optimizer": plan_item["optimizer"],
            "distribution_type": plan_item["distribution"],
            "distribution_param": plan_item.get("distribution_param"),
        })

        gen = FederatedDataGenerator(config=cfg, dataset=dataset)
        summary = gen.run()

        run_id = summary["run_id"]
        db_path = summary["db_path"]
        con = sqlite3.connect(db_path)
        try:
            runs_df   = pd.read_sql_query("SELECT * FROM runs WHERE run_id = ?", con, params=[run_id])
            rounds_df = pd.read_sql_query("SELECT * FROM rounds WHERE run_id = ? ORDER BY round", con, params=[run_id])
            clients_df= pd.read_sql_query("SELECT * FROM client_rounds WHERE run_id = ? ORDER BY round, client_id", con, params=[run_id])
        finally:
            con.close()

        runs_df.to_csv(f"outputs/runs_{idx:04d}.csv", index=False)
        rounds_df.to_csv(f"outputs/rounds_{idx:04d}.csv", index=False)
        clients_df.to_csv(f"outputs/client_rounds_{idx:04d}.csv", index=False)

        manifest.append({
            "idx": idx,
            "run_id": run_id,
            "dataset": dataset,
            "task_type": task,
            "group_id": plan_item.get("group_id"),
            "config": cfg
        })

    with open("outputs/study_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Wrote manifest for {len(manifest)} runs to outputs/study_manifest.json")

def register_autogen(subparsers):
    p = subparsers.add_parser("autogen", help="Automatically generate many runs")
    p.add_argument("--runs", type=int, default=50, help="Total runs to generate")
    p.add_argument("--task-split", type=str, default="50,30,20", help="classification,regression,clustering percentages")
    p.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    p.set_defaults(_handler=_handle)
