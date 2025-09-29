# records_metrics.py
from __future__ import annotations
import json
import numpy as np
# ---------- record builders ----------

def _nan_or(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except Exception:
        return None
    
def build_run_meta(run_id, dataset, task_type, model_type, split_meta, knobs, params_count, metric_name):
    return {
        "run_id": run_id,
        "dataset": dataset,
        "task_type": task_type,
        "model_type": model_type,
        "distribution_type": split_meta["strategy"],
        "distribution_param": split_meta["distribution_param"],
        "hidden_layers": ",".join(str(u) for u in (knobs["hidden_layers"] or [])),
        "activation": knobs["activation"],
        "dropout": knobs["dropout"],
        "weight_decay": knobs["weight_decay"],
        "params_count": int(params_count),
        "optimizer": knobs["optimizer"],
        "learning_rate": knobs["learning_rate"],
        "batch_size": knobs["batch_size"],
        "epochs_per_round": knobs["local_epochs"],
        "metric_name": metric_name,
        "distribution_bins": knobs["distribution_bins"],
    }

def build_run_record(run_meta, knobs, config):
    return {
        "run_id": run_meta["run_id"],
        "dataset": run_meta["dataset"],
        "task_type": run_meta["task_type"],
        "model_type": run_meta["model_type"],
        "split_strategy": run_meta["distribution_type"],
        "distribution_param": str(run_meta["distribution_param"]),
        "num_clients": int(knobs["num_clients"]),
        "num_rounds": int(knobs["num_rounds"]),
        "seed": config.get("seed"),

        # training knobs
        "learning_rate": _nan_or(knobs.get("learning_rate")),
        "batch_size": _nan_or(knobs.get("batch_size")),
        "local_epochs": _nan_or(knobs.get("local_epochs")),
        "hidden_layers": ",".join(str(u) for u in (knobs["hidden_layers"] or [])),
        "activation": knobs.get("activation"),
        "dropout": _nan_or(knobs.get("dropout")),
        "weight_decay": _nan_or(knobs.get("weight_decay")),
        "optimizer": knobs.get("optimizer"),

        # model/meta
        "params_count": int(run_meta.get("params_count", 0)),
        "metric_name": run_meta.get("metric_name"),
        "distribution_bins": _nan_or(run_meta.get("distribution_bins")),

        # clustering passthroughs (safe to be NULL)
        "clustering_k": config.get("clustering_k"),
        "clustering_init": config.get("clustering_init"),
        "clustering_n_init": config.get("clustering_n_init"),
        "clustering_max_iter": config.get("clustering_max_iter"),
        "clustering_tol": config.get("clustering_tol"),

        "save_weights": 1 if config.get("save_weights", False) else 0,
        "dataset_args_json": json.dumps(config.get("dataset_args", {}) or {}),
    }


def build_round_record(run_meta, round_idx, loss, global_metric, global_score, global_extra):
    return {
        "run_id": run_meta["run_id"],
        "round": int(round_idx),
        "global_loss": _nan_or(loss),
        "global_metric": _nan_or(global_metric),
        "global_metric_name": run_meta.get("metric_name"),   # 'accuracy'|'rmse'|'silhouette'
        "global_aux_metric": _nan_or(global_extra),          # e.g. f1 or inertia
        "global_score": _nan_or(global_score),
        "frontier_json": None,
    }

def build_client_record(run_meta, round_idx, client_id, distribution, metric_key, outcome, task_type):
    """
    Produces a row for client_rounds (participating client).
    """
    # task convenience fields
    accuracy = None
    f1 = None
    rmse = None
    rmse_orig = None
    silhouette = None
    inertia = None

    if task_type == "classification":
        accuracy = outcome.metric_value
        f1 = outcome.extra_metric
    elif task_type == "regression":
        rmse = outcome.metric_value
        rmse_orig = (outcome.extras or {}).get("rmse_original_units")
    elif task_type == "clustering":
        silhouette = outcome.metric_value
        inertia = outcome.extra_metric

    # extras (safe if empty)
    extras = outcome.extras or {}

    return {
        "run_id": run_meta["run_id"],
        "round": int(round_idx),
        "client_id": client_id,

        "participated": 1 if outcome.participated else 0,
        "round_fail_reason": outcome.fail_reason,
        "rounds_participated_so_far": outcome.rounds_so_far,

        "data_distribution_json": json.dumps(distribution or {}),

        "samples_count": outcome.samples_count,
        "computation_time_s": _nan_or(outcome.duration),

        "comm_bytes_up": int(outcome.comm_up or 0),
        "comm_bytes_down": int(outcome.comm_down or 0),

        "loss": _nan_or(outcome.loss),
        "accuracy": _nan_or(accuracy),
        "f1": _nan_or(f1),
        "rmse": _nan_or(rmse),
        "rmse_original_units": _nan_or(rmse_orig),
        "silhouette": _nan_or(silhouette),
        "inertia": _nan_or(inertia),

        "metric_score": _nan_or(outcome.metric_score),
        "extra_metric": _nan_or(outcome.extra_metric),

        "ari": _nan_or(extras.get("ari")),
        "nmi": _nan_or(extras.get("nmi")),
        "clustering_k": extras.get("clustering_k"),
        "clustering_agg": extras.get("clustering_agg"),

        "availability_flag": 1 if outcome.participated else 0,
        "throughput_eps": _nan_or(extras.get("throughput_eps")),
        "inference_latency_ms_mean": _nan_or(extras.get("inference_latency_ms_mean")),
        "inference_latency_ms_p95": _nan_or(extras.get("inference_latency_ms_p95")),
        "compute_cost_usd": _nan_or(extras.get("compute_cost_usd")),
        "total_cost_usd": _nan_or(extras.get("total_cost_usd")),
        "reliability_score": _nan_or(extras.get("reliability_score")),
    }

def build_skip_record(run_meta, round_idx, client_id, distribution, samples_count, rounds_so_far, comm_down):
    """
    Produces a row for client_rounds (dropout client).
    Signature kept the same as your existing call site.
    """
    return {
        "run_id": run_meta["run_id"],
        "round": int(round_idx),
        "client_id": client_id,

        "participated": 0,
        "round_fail_reason": "dropped_out",
        "rounds_participated_so_far": rounds_so_far,

        "data_distribution_json": json.dumps(distribution or {}),

        "samples_count": int(samples_count),
        "computation_time_s": 0.0,

        "comm_bytes_up": 0,
        "comm_bytes_down": int(comm_down or 0),

        "loss": None,
        "accuracy": None,
        "f1": None,
        "rmse": None,
        "rmse_original_units": None,
        "silhouette": None,
        "inertia": None,

        "metric_score": None,
        "extra_metric": None,

        "ari": None,
        "nmi": None,
        "clustering_k": None,
        "clustering_agg": None,

        "availability_flag": 0,
        "throughput_eps": None,
        "inference_latency_ms_mean": None,
        "inference_latency_ms_p95": None,
        "compute_cost_usd": None,
        "total_cost_usd": None,
        "reliability_score": None,
    }

def build_global_record(round_idx, metric_key, loss, metric_value, metric_score, aux_metric, target_scaler=None):
    rec = {
        "round": round_idx,
        metric_key: float(metric_value),
        "loss": float(loss),
        "score": float(metric_score),
    }
    if target_scaler and metric_key == "rmse" and target_scaler.get("type") == "standard":
        rec["rmse_original_units"] = float(metric_value) * float(target_scaler["std"])
    if aux_metric is not None:
        rec["aux_metric"] = float(aux_metric)
    return rec

def attach_reliability(df, source_col="metric_score"):
    reliability = (
        df.groupby("client_id")[source_col]
          .mean()
          .reset_index()
          .rename(columns={source_col: "reliability_Score"})
    )
    return df.merge(reliability, on="client_id", how="left")

def save_global_metrics_json(metric_key: str, records: list[dict], task_type: str, save: bool):
    if not save:
        return
    with open("weights/global_metrics.json", "w") as f:
        json.dump({"metric": metric_key, "records": records}, f, indent=4)
    if task_type == "classification":
        with open("weights/global_accuracies.json", "w") as f:
            json.dump(records, f, indent=4)

