# records_metrics.py
from __future__ import annotations
import json
import numpy as np
from ..model_utils import evaluate_model, aggregate_weights

# ---------- metrics / scoring ----------

def metric_score_value(task_type: str, metric_value: float | None) -> float:
    """Map raw metric to a [0,1]-ish score. For classification/clustering, identity; for regression, 1/(1+rmse)."""
    mv = float(metric_value) if metric_value is not None else np.nan
    if mv != mv:  # NaN
        return np.nan
    if task_type == "regression":
        return 1.0 / (1.0 + mv)
    return mv

def weights_size(weights_dict_or_list) -> int:
    if not weights_dict_or_list:
        return 0
    arrays = weights_dict_or_list.values() if isinstance(weights_dict_or_list, dict) else weights_dict_or_list
    return int(sum(np.asarray(w).nbytes for w in arrays))

# ---------- record builders ----------

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

def build_client_record(run_meta, round_idx, client_id, distribution, metric_key, outcome, task_type: str):
    rec = {
        **run_meta,
        "round": round_idx,
        "client_id": client_id,
        "participated": outcome.participated,
        "round_fail_reason": outcome.fail_reason,
        "Data_Distribution": distribution,           # Standardised column name
        "samples_count": outcome.samples_count,
        "Computation_Time": outcome.duration,
        # QoS
        "loss": outcome.loss,
        metric_key: outcome.metric_value,
        "metric_score": outcome.metric_score,
        "extra_metric": outcome.extra_metric,
        "reliability": np.nan,
        "rounds_participated_so_far": outcome.rounds_so_far,
        "comm_bytes_up": outcome.comm_up,
        "comm_bytes_down": outcome.comm_down,
    }

    # Convenience columns by task
    if task_type == "classification":
        rec["accuracy"] = outcome.metric_value
        rec["f1"] = outcome.extra_metric
    elif task_type == "clustering":
        rec["silhouette"] = outcome.metric_value
        rec["inertia"] = outcome.extra_metric
    else:
        rec.setdefault("accuracy", np.nan)
        rec.setdefault("f1", np.nan)

    # Additional task-specific extras (e.g., ARI/NMI, clustering_k...)
    if outcome.extras:
        rec.update(outcome.extras)

    return rec

def build_skip_record(run_meta, round_idx, client_id, distribution, samples_count, rounds_so_far, comm_down):
    return {
        **run_meta,
        "round": round_idx,
        "client_id": client_id,
        "participated": False,
        "round_fail_reason": "dropped_out",
        "Data_Distribution": distribution,
        "samples_count": samples_count,
        "Computation_Time": 0.0,
        "loss": np.nan,
        "metric_score": np.nan,
        "extra_metric": np.nan,
        "reliability": np.nan,
        "rounds_participated_so_far": rounds_so_far,
        "comm_bytes_up": 0,
        "comm_bytes_down": comm_down,
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

# ---------- aggregation & evaluation ----------

def aggregate_nn_and_eval(global_model, client_weights: list, round_idx: int, task_type: str, x_test, y_test, save: bool):
    """FedAvg aggregation (via your model_utils.aggregate_weights) + evaluate_model."""
    if client_weights:
        new_global_weights = aggregate_weights(client_weights)
        # Keep parity with your existing set_weights(list_ordered)
        global_model.set_weights([new_global_weights[f"layer_{i}"] for i in range(len(new_global_weights))])
        if save:
            with open(f"weights/global_round_{round_idx}.json", "w") as f:
                json.dump({k: v.tolist() for k, v in new_global_weights.items()}, f, indent=4)
    else:
        print("No participating clients this round; keeping previous global weights.")

    loss, metric_value, extra_metric = evaluate_model(global_model, x_test, y_test, task_type=task_type)
    mscore = metric_score_value(task_type, metric_value)
    return loss, metric_value, mscore, extra_metric

def eval_clustering_global(build_model_fn, x_train, x_test):
    """Proxy-global metric for clustering: fit a fresh adapter on full train and eval on test."""
    try:
        tmp = build_model_fn()
        tmp.fit(x_train)
        _, sil, inertia = tmp.evaluate(x_test)
    except Exception:
        sil, inertia = (np.nan, np.nan)
    mscore = metric_score_value("clustering", sil)
    return np.nan, sil, mscore, inertia
