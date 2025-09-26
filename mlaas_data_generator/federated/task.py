# task.py
from __future__ import annotations
from dataclasses import dataclass
import json, time, numpy as np

from ..model_utils import create_model, train_local_model, evaluate_model
from .records import (
    metric_score_value,
    weights_size,
    aggregate_nn_and_eval,   # FedAvg + evaluate for NN tasks
    eval_clustering_global,  # proxy-global fit+eval for clustering
)

@dataclass
class ClientOutcome:
    participated: bool
    fail_reason: str
    samples_count: int
    duration: float
    loss: float
    metric_value: float 
    metric_score: float 
    extra_metric: float 
    rounds_so_far: int
    comm_down: int
    comm_up: int
    payload: dict | list | None   # weights for NN, None for clustering
    extras: dict                  # extra columns per task

class TaskStrategy:
    """Base class: thin wrapper around your existing per-task logic."""
    def __init__(self, meta, knobs, config, x_test, y_test, metric_key, save_weights: bool):
        self.meta = meta
        self.knobs = knobs
        self.config = config
        self.x_test = x_test
        self.y_test = y_test
        self.metric_key = metric_key
        self.save_weights = save_weights

    # ---- shared helpers
    def build_model(self):
        return create_model(
            input_shape=tuple(self.meta["input_shape"]),
            num_classes=self.meta.get("num_classes"),
            hidden_layers=self.knobs["hidden_layers"],
            learning_rate=self.knobs["learning_rate"],
            activation=self.knobs["activation"],
            dropout=self.knobs["dropout"],
            weight_decay=self.knobs["weight_decay"],
            optimizer=self.knobs["optimizer"],
            task_type=self.task_type()
        )

    def comm_down_bytes(self, global_model):
        # number of bytes when broadcasting global weights (0 for clustering adapter if no weights)
        try:
            return weights_size(global_model.get_weights())
        except Exception:
            return 0
        
    def task_type(self) -> str: ...
    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome: ...
    def aggregate_and_eval(self, global_model, client_payloads, round_idx, x_train, x_test, y_test):
        """Return (loss, metric_value, metric_score, extra_metric)."""
        ...

# -------------------- Classification --------------------

class ClassificationStrategy(TaskStrategy):
    def task_type(self) -> str: return "classification"

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome:
        local_model = self.build_model()
        local_model.set_weights(global_model.get_weights())
        samples_count = len(y)
        start = time.time()
        try:
            weights = train_local_model(
                local_model, x, y,
                epochs=self.knobs["local_epochs"],
                batch_size=self.knobs["batch_size"],
            )
            duration = time.time() - start
            loss, metric_value, extra_metric = evaluate_model(local_model, self.x_test, self.y_test, task_type="classification")
            mscore = metric_score_value("classification", metric_value)

            if self.save_weights:
                with open(f"weights/{client_id}_round_{round_idx}.json", "w") as f:
                    json.dump({k: v.tolist() for k, v in weights.items()}, f, indent=4)

            return ClientOutcome(
                participated=True, fail_reason="", samples_count=samples_count, duration=duration,
                loss=loss, metric_value=metric_value, metric_score=mscore, extra_metric=extra_metric,
                rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=weights_size(weights),
                payload=weights, extras={},  # accuracy/f1 added in records builder
            )
        except Exception:
            duration = time.time() - start
            return ClientOutcome(
                participated=False, fail_reason="error", samples_count=samples_count, duration=duration,
                loss=np.nan, metric_value=np.nan, metric_score=np.nan, extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1, comm_down=comm_down, comm_up=0,
                payload=None, extras={},
            )

    def aggregate_and_eval(self, global_model, client_payloads, round_idx, x_train, x_test, y_test):
        return aggregate_nn_and_eval(
            global_model=global_model,
            client_weights=client_payloads,
            round_idx=round_idx,
            task_type="classification",
            x_test=x_test, y_test=y_test,
            save=self.save_weights,
        )

# -------------------- Regression --------------------

class RegressionStrategy(TaskStrategy):
    def task_type(self) -> str: return "regression"

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome:
        local_model = self.build_model()
        local_model.set_weights(global_model.get_weights())
        samples_count = len(y)
        start = time.time()
        try:
            weights = train_local_model(
                local_model, x, y,
                epochs=self.knobs["local_epochs"],
                batch_size=self.knobs["batch_size"],
            )
            duration = time.time() - start
            loss, metric_value, extra_metric = evaluate_model(local_model, self.x_test, self.y_test, task_type="regression")
            mscore = metric_score_value("regression", metric_value)

            if self.save_weights:
                with open(f"weights/{client_id}_round_{round_idx}.json", "w") as f:
                    json.dump({k: v.tolist() for k, v in weights.items()}, f, indent=4)

            return ClientOutcome(
                participated=True, fail_reason="", samples_count=samples_count, duration=duration,
                loss=loss, metric_value=metric_value, metric_score=mscore, extra_metric=extra_metric,
                rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=weights_size(weights),
                payload=weights, extras={},
            )
        except Exception:
            duration = time.time() - start
            return ClientOutcome(
                participated=False, fail_reason="error", samples_count=samples_count, duration=duration,
                loss=np.nan, metric_value=np.nan, metric_score=np.nan, extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1, comm_down=comm_down, comm_up=0,
                payload=None, extras={},
            )

    def aggregate_and_eval(self, global_model, client_payloads, round_idx, x_train, x_test, y_test):
        return aggregate_nn_and_eval(
            global_model=global_model,
            client_weights=client_payloads,
            round_idx=round_idx,
            task_type="regression",
            x_test=x_test, y_test=y_test,
            save=self.save_weights,
        )

# -------------------- Clustering --------------------

class ClusteringStrategy(TaskStrategy):
    def task_type(self) -> str: return "clustering"

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome:
        # local-only KMeans adapter path
        X = x
        t0 = time.time()
        local_model = self.build_model()  # returns KMeansAdapter
        local_model.fit(X)
        duration = time.time() - t0

        # evaluate on test set
        try:
            loss, sil, inertia = local_model.evaluate(self.x_test)
        except Exception:
            loss, sil, inertia = (np.nan, np.nan, np.nan)

        # optional ARI/NMI when labels exist
        ari = nmi = np.nan
        try:
            if self.y_test is not None:
                from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                preds = local_model.predict(self.x_test)
                if preds.shape[0] == self.y_test.shape[0]:
                    ari = float(adjusted_rand_score(self.y_test, preds))
                    nmi = float(normalized_mutual_info_score(self.y_test, preds))
        except Exception:
            pass

        try:
            comm_up = weights_size(local_model.get_weights())
        except Exception:
            comm_up = 0

        mscore = metric_score_value("clustering", sil)
        return ClientOutcome(
            participated=True, fail_reason="", samples_count=len(X), duration=duration,
            loss=np.nan, metric_value=sil, metric_score=mscore, extra_metric=inertia,
            rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=comm_up,
            payload=None,
            extras={
                "silhouette": sil,
                "inertia": inertia,
                "ari": ari,
                "nmi": nmi,
                "clustering_k": getattr(local_model, "k", np.nan),
                "clustering_agg": "local_only",
            },
        )

    def aggregate_and_eval(self, global_model, client_payloads, round_idx, x_train, x_test, y_test):
        # no aggregation; compute a proxy global metric by fitting on full train
        return eval_clustering_global(build_model_fn=self.build_model, x_train=x_train, x_test=x_test)

# -------------------- factory --------------------

def make_task_strategy(task_type: str, meta: dict, knobs: dict, config: dict, x_test, y_test, metric_key: str, save_weights: bool) -> TaskStrategy:
    if task_type == "classification":
        return ClassificationStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)
    if task_type == "regression":
        return RegressionStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)
    if task_type == "clustering":
        return ClusteringStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)
    raise ValueError(f"Unknown task type: {task_type}")
