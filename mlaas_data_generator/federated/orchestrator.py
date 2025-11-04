# orchestrator.py
from __future__ import annotations
import os, uuid
import numpy as np

from ..config import CONFIG
from ..data.loaders import load_dataset
from ..data.splitters import split_data
from ..data.distributions import get_data_distribution
from ..storage.writer import make_writer
from .task import make_task_strategy
from .system_metrics import capture_hardware_snapshot, summarize_round_usage

from .records import (
    build_run_meta,
    build_run_record,
    build_client_record,
    build_skip_record,
    build_round_record
)

class FederatedDataGenerator:
    """Generate MLaaS client records using a simple federated-learning loop."""
    def __init__(
        self,
        config: dict | None = None,
        dataset: str | None = None,
        task_type: str | None = None,
        model_type: str | None = None,
        dataset_args: dict | None = None,
    ):
        # --- config & identity
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)

        self.dataset = dataset or self.config.get("dataset", "fashion_mnist")
        self.model_type = model_type or self.config.get("model_type", "CNN")
        self.config["dataset"] = self.dataset
        self.config["model_type"] = self.model_type

        # --- dataset args
        self.dataset_args = {}
        config_dataset_args = self.config.get("dataset_args") or {}
        if isinstance(config_dataset_args, dict):
            self.dataset_args.update(config_dataset_args)
        if dataset_args:
            self.dataset_args.update(dataset_args)
        if self.dataset_args:
            self.config["dataset_args"] = dict(self.dataset_args)

        # --- load data
        train, test, meta = load_dataset(self.dataset, **self.dataset_args)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = train, test
        self.meta = meta
        self.input_shape = tuple(meta["input_shape"])
        self.num_classes = meta.get("num_classes")

        # --- task & metric labels
        requested_task = task_type or self.config.get("task_type")
        meta_task = meta.get("task_type", "classification")
        self.task_type = requested_task or meta_task
        if requested_task != meta_task:
            print(f"Warning: overriding dataset task type '{meta_task}' with requested '{self.task_type}'.")

        if self.task_type == "clustering":
            self.metric_key = "silhouette"
            self.metric_label = "Silhouette"
        elif self.task_type == "classification":
            self.metric_key = "accuracy"
            self.metric_label = "Accuracy"
        else:
            self.metric_key = "rmse"
            self.metric_label = "RMSE"

        self.target_scaler = meta.get("target_scaler")
        self.save_weights = bool(self.config.get("save_weights", True))
        self.distribution_bins = int(self.config.get("distribution_bins", 10) or 10)

        # Regression: set value range for distribution summaries
        if self.task_type == "regression" or self.num_classes is None:
            if len(self.y_train) > 0:
                y_min = float(np.min(self.y_train))
                y_max = float(np.max(self.y_train))
                if y_min == y_max:
                    y_min -= 0.5; y_max += 0.5
                self.distribution_range = (y_min, y_max)
            else:
                self.distribution_range = (0.0, 1.0)
        else:
            self.distribution_range = None

        # --- knobs
        hidden_layers = self.config.get("hidden_layers", [self.config.get("reduced_neurons", 64)])
        if hidden_layers is None:
            hidden_layers = [self.config.get("reduced_neurons", 64)]
        self.hidden_layers = list(hidden_layers)

        self.knobs = {
            "num_clients": int(self.config["num_clients"]),
            "num_rounds": int(self.config["num_rounds"]),
            "local_epochs": int(self.config["local_epochs"]),
            "batch_size": self.config["batch_size"],
            "learning_rate": self.config["learning_rate"],
            "hidden_layers": self.hidden_layers,
            "activation": self.config.get("activation", "relu"),
            "dropout": float(self.config.get("dropout", 0.0) or 0.0),
            "weight_decay": float(self.config.get("weight_decay", 0.0) or 0.0),
            "optimizer": self.config.get("optimizer", "adam"),
            "distribution_type": self.config.get("distribution_type", "iid"),
            "distribution_param": self.config.get("distribution_param", None),
            "custom_distributions": self.config.get("custom_distributions", None),
            "sample_size": self.config.get("sample_size", None),
            "sample_frac": self.config.get("sample_frac", None),
            "distribution_bins": self.distribution_bins,
            "early_stopping_patience": self.config.get("early_stopping_patience")
        }

        self.rng = np.random.default_rng(self.config.get("seed", 42))

        # Strategy encapsulates build/train/eval details for this task (thin wrapper around your current logic)
        self.strategy = make_task_strategy(
            task_type=self.task_type,
            meta=self.meta,
            knobs=self.knobs,
            config=self.config,
            x_test=self.x_test,
            y_test=self.y_test,
            metric_key=self.metric_key,
            save_weights=self.save_weights,
        )

        # --- Disable multi-round training for non-federated models
        if self.task_type == "clustering" or self.model_type.lower() == "randomforest":
            print(f"Non-federated model detected ({self.model_type}); forcing single-round training.")
            self.knobs["num_rounds"] = 1

    def run(self):
        os.makedirs("weights", exist_ok=True)
        patience_cfg = self.config.get("early_stopping_patience")
        if patience_cfg in (None, "", False):
            early_stopping_patience = None
        else:
            try:
                early_stopping_patience = int(patience_cfg)
            except (TypeError, ValueError):
                early_stopping_patience = None
            else:
                if early_stopping_patience <= 0:
                    early_stopping_patience = None
        if self.task_type == "regression" and self.knobs["distribution_type"] in {"dirichlet", "shard", "label_per_client"}:
            print("Warning: label-based partitioning not supported for regression; using 'iid'.")
            self.knobs["distribution_type"] = "iid"

        clients, split_info = split_data(
            self.x_train, self.y_train, self.knobs["num_clients"],
            strategy=self.knobs["distribution_type"],
            distribution_param=self.knobs["distribution_param"],
            custom_distributions=self.knobs["custom_distributions"],
            sample_size=self.knobs["sample_size"],
            sample_frac=self.knobs["sample_frac"],
            rng=self.rng
        )
        print("\n========== RUN CONFIGURATION SUMMARY ==========")
        for key, val in sorted(self.config.items()):
            print(f"{key:25}: {val}")
        print("------------------------------------------------")
        print("Derived knobs:")
        for key, val in sorted(self.knobs.items()):
            print(f"{key:25}: {val}")
        print("------------------------------------------------")
        print(f"Dataset:         {self.dataset}")
        print(f"Model Type:      {self.model_type}")
        print(f"Task Type:       {self.task_type}")
        print(f"Input Shape:     {self.input_shape}")
        print(f"Num Classes:     {self.num_classes}")
        print(f"Save Weights:    {self.save_weights}")
        print(f"Distribution Bins: {self.distribution_bins}")
        print("================================================\n")
        print("Client data distributions before training:")
        for client_id, data in clients.items():
            print(f"{client_id}: {get_data_distribution(data['y'], self.num_classes, bins=self.knobs.get('distribution_bins'), value_range=self.distribution_range)}")

        # build global model via strategy
        global_model = self.strategy.build_model()

        hardware_snapshot = capture_hardware_snapshot()

        run_meta = build_run_meta(
            run_id=str(uuid.uuid4()),
            dataset=self.dataset,
            task_type=self.task_type,
            model_type=self.model_type,
            split_meta=split_info,
            knobs=self.knobs,
            params_count=int(global_model.count_params()),
            metric_name=self.metric_key,
            hardware_snapshot=hardware_snapshot
        )

        db_path = self.config.get("db_path", "federated.db")
        writer = make_writer("sqlite", db_path=db_path)
        writer.start()
        try:
            writer.write_run(build_run_record(run_meta, self.knobs, self.config))

            participated_counts = {cid: 0 for cid in clients.keys()}

            for round_num in range(self.knobs["num_rounds"]):
                round_idx = round_num + 1
                print(f"--- Round {round_idx} ---")
                client_payloads = []
                client_outcomes = []
                round_metrics = []
                skipped_clients = 0

                # bytes to broadcast down: size of global weights (strategy computes)
                down_bytes = self.strategy.comm_down_bytes(global_model)

                for client_id, data in clients.items():

                    distribution = get_data_distribution(
                        data["y"], self.num_classes, bins=self.knobs.get("distribution_bins"), value_range=self.distribution_range
                    )
                    n_samples = len(data["y"])

                    # dropout
                    if self.rng.random() < self.config.get("client_dropout_rate", 0.0):
                        row = build_skip_record(
                            run_meta, round_idx, client_id, distribution, n_samples,
                            rounds_so_far=participated_counts[client_id],
                            comm_down=down_bytes
                        )
                        writer.write_client_round(row)
                        print(f"{client_id} dropped out ")
                        skipped_clients += 1
                        continue

                    # train client via strategy
                    next_rounds_so_far = participated_counts[client_id] + 1

                    print(f"{client_id} training...")
                    
                    outcome = self.strategy.train_client(
                        client_id=client_id,
                        x=data["x"], y=data["y"],
                        global_model=global_model,
                        round_idx=round_idx,
                        rounds_so_far=next_rounds_so_far,
                        comm_down=down_bytes,
                    )
                    # record
                    row = build_client_record(
                        run_meta=run_meta,
                        round_idx=round_idx,
                        client_id=client_id,
                        distribution=distribution,
                        metric_key=self.metric_key,
                        outcome=outcome,
                        task_type=self.task_type,
                    )
                    writer.write_client_round(row)
                    client_outcomes.append(outcome)
                    round_metrics.append(
                        {
                            "participated": bool(outcome.participated),
                            "duration": outcome.duration,
                            "cpu_utilization": outcome.cpu_utilization,
                            "memory_utilization": outcome.memory_utilization,
                            "memory_used_mb": outcome.memory_used_mb,
                            "gpu_utilization": outcome.gpu_utilization,
                            "gpu_memory_utilization": outcome.gpu_memory_utilization,
                            "gpu_memory_used_mb": outcome.gpu_memory_used_mb,
                            "cpu_time_s": outcome.cpu_time_s,
                        }
                    )
                    
                    if outcome.participated:
                        participated_counts[client_id] = next_rounds_so_far

                    if outcome.payload is not None:
                        client_payloads.append(outcome.payload)

                # aggregate & evaluate globally
                loss, global_metric, global_score, global_extra = self.strategy.aggregate_and_eval(
                    global_model=global_model,
                    client_payloads=client_payloads,
                    client_outcomes=client_outcomes,
                    round_idx=round_idx,
                    x_train=self.x_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                )

                round_usage_summary = summarize_round_usage(
                    round_metrics,
                    scheduled_clients=len(clients),
                    skipped_clients=skipped_clients,
                )

                writer.write_round(
                    build_round_record(
                        run_meta,
                        round_idx=round_idx,
                        loss=loss,
                        global_metric=global_metric,
                        global_score=global_score,
                        global_extra=global_extra,
                        resource_summary=round_usage_summary
                    )
                )

                if self.task_type == "regression" and self.target_scaler and self.target_scaler.get("type") == "standard":
                    rmse_std = float(global_metric)
                    rmse_orig = rmse_std * float(self.target_scaler["std"])
                    print(f"Global model {self.metric_label}: {rmse_std:.6f} (standardized) | {rmse_orig:.2f} (original units)")
                else:
                    print(f"Global model {self.metric_label}: {global_metric}")

                if self.task_type == "regression":
                    print(f"Global metric score: {global_score}")
                if global_extra is not None:
                    print(f"Global auxiliary metric: {global_extra}")
        finally:
            writer.finish()

        print("Federated Learning Process Complete!\n")
        return {
            "run_id": run_meta["run_id"],
            "db_path": db_path,
            "rounds": self.knobs["num_rounds"],
            "clients": self.knobs["num_clients"],
        }
