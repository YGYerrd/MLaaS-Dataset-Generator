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
    
def build_run_meta(run_id, dataset, task_type, model_type, split_meta, knobs, params_count, metric_name, hardware_snapshot):
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
        "hardware_snapshot": hardware_snapshot
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
        "hardware_snapshot_json": json.dumps(run_meta.get("hardware_snapshot") or {})
    }


def build_round_record(run_meta, round_idx, loss, global_metric, global_score, global_extra, resource_summary=None):
    summary = resource_summary or {}
    return {
        "run_id": run_meta["run_id"],
        "round": int(round_idx),
        "global_loss": _nan_or(loss),
        "global_metric": _nan_or(global_metric),
        "global_metric_name": run_meta.get("metric_name"),
        "global_aux_metric": _nan_or(global_extra),
        "global_score": _nan_or(global_score),
        "frontier_json": None,
        "scheduled_clients": summary.get("scheduled_clients"),
        "attempted_clients": summary.get("attempted_clients"),
        "participating_clients": summary.get("participating_clients"),
        "dropped_clients": summary.get("dropped_clients"),
        "avg_client_duration": _nan_or(summary.get("avg_client_duration")),
        "max_client_duration": _nan_or(summary.get("max_client_duration")),
        "avg_cpu_util": _nan_or(summary.get("avg_cpu_util")),
        "max_cpu_util": _nan_or(summary.get("max_cpu_util")),
        "avg_memory_util": _nan_or(summary.get("avg_memory_util")),
        "max_memory_util": _nan_or(summary.get("max_memory_util")),
        "avg_memory_used_mb": _nan_or(summary.get("avg_memory_used_mb")),
        "max_memory_used_mb": _nan_or(summary.get("max_memory_used_mb")),
        "avg_gpu_util": _nan_or(summary.get("avg_gpu_util")),
        "max_gpu_util": _nan_or(summary.get("max_gpu_util")),
        "avg_gpu_memory_util": _nan_or(summary.get("avg_gpu_memory_util")),
        "max_gpu_memory_util": _nan_or(summary.get("max_gpu_memory_util")),
        "avg_gpu_memory_used_mb": _nan_or(summary.get("avg_gpu_memory_used_mb")),
        "max_gpu_memory_used_mb": _nan_or(summary.get("max_gpu_memory_used_mb")),
        "avg_cpu_time_s": _nan_or(summary.get("avg_cpu_time_s")),
        "max_cpu_time_s": _nan_or(summary.get("max_cpu_time_s")),
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
        
        "cpu_time_s": _nan_or(outcome.cpu_time_s),
        "cpu_utilization": _nan_or(outcome.cpu_utilization),
        "memory_used_mb": _nan_or(outcome.memory_used_mb),
        "memory_utilization": _nan_or(outcome.memory_utilization),
        "gpu_utilization": _nan_or(outcome.gpu_utilization),
        "gpu_memory_used_mb": _nan_or(outcome.gpu_memory_used_mb),
        "gpu_memory_utilization": _nan_or(outcome.gpu_memory_utilization),

        "availability_flag": 1 if outcome.participated else 0,
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
        
        "cpu_time_s": None,
        "cpu_utilization": None,
        "memory_used_mb": None,
        "memory_utilization": None,
        "gpu_utilization": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_utilization": None,

        "availability_flag": 0,
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


def save_global_metrics_json(metric_key: str, records: list[dict], task_type: str, save: bool):
    if not save:
        return
    with open("weights/global_metrics.json", "w") as f:
        json.dump({"metric": metric_key, "records": records}, f, indent=4)
    if task_type == "classification":
        with open("weights/global_accuracies.json", "w") as f:
            json.dump(records, f, indent=4)

