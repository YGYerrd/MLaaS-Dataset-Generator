"""Federated learning based data generator."""

from __future__ import annotations
import json
import os
import time
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import uuid


from .config import CONFIG
from .data_utils import (
    load_dataset,
    split_data,
    get_data_distribution,
)
from .model_utils import create_model, train_local_model, evaluate_model, aggregate_weights


class FederatedDataGenerator:
    """Generate MLaaS client records using a simple federated-learning loop."""

    def __init__(
        self, 
        config: dict | None = None, 
        dataset: str = "fashion_mnist",
        task_type: str = "classification",
        model_type: str = "CNN"
        ):
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.dataset = dataset
        self.model_type = model_type

        seed = self.config.get("seed", 42)
        self.rng = np.random.default_rng(seed)
 
        (train, test, meta) = load_dataset(dataset)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = train, test
        self.meta = meta 
        self.input_shape = tuple(meta["input_shape"])
        self.num_classes = meta.get("num_classes")
        self.task_type = meta["task_type"]
        self.save_weights = bool(self.config.get("save_weights", True))

        self.knobs = {
            "num_clients": int(self.config["num_clients"]),
            "num_rounds": int(self.config["num_rounds"]),
            "local_epochs": int(self.config["local_epochs"]),
            "batch_size": int(self.config["batch_size"]),
            "learning_rate": float(self.config["learning_rate"]),
            "hidden_layers": self.config.get("hidden_layers", [self.config.get("reduced_neurons", 64)]),
            "activation": self.config.get("activation", "relu"),
            "dropout": float(self.config.get("dropout", 0.0) or 0.0),
            "weight_decay": float(self.config.get("weight_decay", 0.0) or 0.0),
            "optimizer": self.config.get("optimizer", "adam"),
            "distribution_type": self.config.get("distribution_type", "iid"),
            "distribution_param": self.config.get("distribution_param", None),
            "custom_distributions": self.config.get("custom_distributions", None),
            "sample_size": self.config.get("sample_size", None),
            "sample_frac": self.config.get("sample_frac", None),
        }


    def run(self) -> pd.DataFrame:
        os.makedirs("weights", exist_ok=True)

        clients, split_info = split_data(
            self.x_train, self.y_train, self.knobs["num_clients"],
            strategy=self.knobs["distribution_type"],
            distribution_param=self.knobs["distribution_param"],
            custom_distributions=self.knobs["custom_distributions"],
            sample_size=self.knobs["sample_size"],
            sample_frac=self.knobs["sample_frac"],
            rng=self.rng
        )
        

        print(f"\nUsing split strategy: {split_info['strategy']} | params: {split_info['distribution_param']}\n")
        print(f"Using Dataset: {self.dataset}\n")

        print("Client data distributions before training:")
        for client_id, data in clients.items():
            print(f"{client_id}: {get_data_distribution(data['y'], self.num_classes)}")

        global_model = self._build_model()
        run_meta = self._build_run_meta(global_model, split_info)

        records = []
        global_accuracies = []
        participated_counts = {cid: 0 for cid in clients.keys()}

        for round_num in range(self.knobs["num_rounds"]):
            round_idx = round_num + 1
            print(f"--- Round {round_idx} ---")
            client_weights = []

            down_bytes = self._weights_size(global_model.get_weights())

            for client_id, data in clients.items():
                distribution = get_data_distribution(data["y"], self.num_classes)
                n_samples = len(data["y"])

                if self.rng.random() < self.config.get("client_dropout_rate", 0.0):
                    records.append(self._make_skip_record(
                        run_meta, round_idx, client_id, distribution, n_samples, rounds_so_far= participated_counts[client_id],
                        comm_down=down_bytes
                    ))
                    print(f"{client_id} dropped out ")
                    continue
                
                
                next_rounds_so_far = participated_counts[client_id] + 1

                print(f"{client_id} training...")
                
                record, weights = self._train_client(
                    client_id, data["x"], data["y"], global_model, run_meta, round_idx,
                    rounds_so_far=next_rounds_so_far, comm_down=down_bytes
                )
                records.append(record)
                if weights is not None:
                    client_weights.append(weights)
                    participated_counts[client_id] = next_rounds_so_far

            global_loss, global_accuracy, global_f1 = self._aggregate_and_eval(global_model, client_weights, round_idx)
            print(f"Global model accuracy: {global_accuracy}")
            global_accuracies.append({
                "Round:": round_idx, 
                "Global_accuracy": global_accuracy,
                "Global_loss": global_loss,
                "Global_f1": global_f1
            })
        if self.save_weights:
            with open("weights/global_accuracies.json", "w") as f:
                json.dump(global_accuracies, f, indent=4)
            
        df = pd.DataFrame(records)

        reliability = (
            df.groupby("client_id")["accuracy"]
            .mean()
            .reset_index()
            .rename(columns={"accuracy": "reliability_Score"})
        )
        df = df.merge(reliability, on="client_id", how="left")

        print("Federated Learning Process Complete!\n")
        return df
    
    def _build_model(self):
            return create_model(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                hidden_layers=self.knobs["hidden_layers"],
                learning_rate=self.knobs["learning_rate"],
                activation=self.knobs["activation"],
                dropout=self.knobs["dropout"],
                weight_decay=self.knobs["weight_decay"],
                optimizer=self.knobs["optimizer"],
                task_type=self.task_type
            )
    
    def _build_run_meta(self, global_model, split_meta):
        return {
            "run_id": str(uuid.uuid4()),
            "dataset": self.dataset,
            "task_type": self.task_type,
            "model_type": self.model_type,
            "distribution_type": split_meta["strategy"],
            "distribution_param": split_meta["distribution_param"],
            "hidden_layers": ",".join(str(u) for u in (self.knobs["hidden_layers"] or [])),
            "activation": self.knobs["activation"],
            "dropout": self.knobs["dropout"],
            "weight_decay": self.knobs["weight_decay"],
            "params_count": int(global_model.count_params()),
            "optimizer": self.knobs["optimizer"],
            "learning_rate": self.knobs["learning_rate"],
            "batch_size": self.knobs["batch_size"],
            "epochs_per_round": self.knobs["local_epochs"],
        }
    
    def _train_client(self, client_id, x, y, global_model, run_meta, round_idx, rounds_so_far, comm_down):
        local_model = self._build_model()
        local_model.set_weights(global_model.get_weights())

        distribution = get_data_distribution(y, self.num_classes)
        samples_count = len(y)

        start = time.time()
        try:
            weights = train_local_model(
                local_model, x, y,
                epochs=self.knobs["local_epochs"],
                batch_size=self.knobs["batch_size"],
            )
            duration = time.time() - start
            loss, acc, f1 = evaluate_model(local_model, self.x_test, self.y_test)
            
            if self.save_weights:
                with open(f"weights/{client_id}_round_{round_idx}.json", "w") as f:
                    json.dump({k: v.tolist() for k, v in weights.items()}, f, indent=4)

            rec = self._make_record(
                run_meta=run_meta,
                round_idx=round_idx,
                client_id=client_id,
                participated=True,
                fail_reason="",
                distribution=distribution,
                samples_count=samples_count,
                duration=duration,
                loss=loss, acc=acc, f1=f1,
                rounds_so_far=rounds_so_far,
                comm_down=comm_down,
                comm_up=self._weights_size(weights),
            )
            return rec, weights

        except Exception:
            duration = time.time() - start
            rec = self._make_record(
                run_meta=run_meta,
                round_idx=round_idx,
                client_id=client_id,
                participated=False,
                fail_reason="error",
                distribution=distribution,
                samples_count=samples_count,
                duration=duration,
                loss=np.nan, acc=np.nan, f1=np.nan,
                rounds_so_far=rounds_so_far - 1,
                comm_down=comm_down,
                comm_up=0,
            )
            return rec, None

    def _aggregate_and_eval(self, global_model, client_weights, round_idx):
        if client_weights:
            new_global_weights = aggregate_weights(client_weights)
            global_model.set_weights([new_global_weights[f"layer_{i}"] for i in range(len(new_global_weights))])
            with open(f"weights/global_round_{round_idx}.json", "w") as f:
                json.dump({k: v.tolist() for k, v in new_global_weights.items()}, f, indent=4)
        else:
            print("No participating clients this round; keeping previous global weights.")
        return evaluate_model(global_model, self.x_test, self.y_test)

    def _make_record(
    self, run_meta, round_idx, client_id, participated, fail_reason,
    distribution, samples_count, duration, loss, acc, f1,
    rounds_so_far, comm_down, comm_up
    ):
        # single source of truth for CSV rows
        return {
            **run_meta,
            "round": round_idx,
            "client_id": client_id,
            "participated": participated,
            "round_fail_reason": fail_reason,
            "data_Distribution": distribution,
            "samples_count": samples_count,
            "Computation_Time": duration,
            # QoS
            "accuracy": acc,
            "loss": loss,
            "f1": f1,
            "reliability": np.nan, 
            "rounds_participated_so_far": rounds_so_far,
            "comm_bytes_up": comm_up,
            "comm_bytes_down": comm_down,
        }

    def _make_skip_record(self, run_meta, round_idx, client_id, distribution, samples_count, rounds_so_far, comm_down):
        # dropout (no training)
        return self._make_record(
            run_meta=run_meta,
            round_idx=round_idx,
            client_id=client_id,
            participated=False,
            fail_reason="dropped_out",
            distribution=distribution,
            samples_count=samples_count,
            duration=0.0,
            loss=np.nan, acc=np.nan, f1=np.nan,
            rounds_so_far=rounds_so_far,
            comm_down=comm_down,
            comm_up=0,
        )

    def _weights_size(self, weights_dict_or_list):
        if isinstance(weights_dict_or_list, dict):
            arrays = weights_dict_or_list.values()
        else:
            arrays = weights_dict_or_list
        return int(sum(np.asarray(w).nbytes for w in arrays))
