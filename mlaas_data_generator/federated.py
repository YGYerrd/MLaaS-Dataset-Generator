"""Federated learning based data generator."""

from __future__ import annotations
import json
import os
import time
from typing import Dict, Optional
import pandas as pd
import numpy as np


from .config import CONFIG
from .data_utils import load_dataset, split_data, split_custom_data, get_data_distribution
from .model_utils import create_model, train_local_model, evaluate_model, aggregate_weights
from .metrics import compute_binary_vector


class FederatedDataGenerator:
    """Generate MLaaS client records using a simple federated-learning loop."""

    def __init__(
        self, 
        config: dict | None = None, 
        dataset: str = "fashion_mnist",
        client_distributions: Dict[str, Dict[int, int]] = None
        ):
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)
        self.dataset = dataset
        self.client_distributions = client_distributions

        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_dataset(dataset)
        self.input_shape = self.x_train.shape[1:]
        self.num_classes = len(np.unique(self.y_train))

    def run(self) -> pd.DataFrame:
        os.makedirs("weights", exist_ok=True)

        if self.client_distributions:
            clients = split_custom_data(self.x_train, self.y_train, self.client_distributions)
        else:
            clients = split_data(self.x_train, self.y_train, self.config["num_clients"])


        global_model = create_model(
            self.input_shape,
            self.num_classes,
            self.config["reduced_neurons"],
            self.config["learning_rate"],
        )
        
        records = []

        for round_num in range(CONFIG["num_rounds"]):
            print(f"--- Round {round_num + 1} ---")

            client_weights = []
            for client_id, data in clients.items():
                print(f"{client_id} training...")
                local_model = create_model(
                    self.input_shape,
                    self.num_classes,
                    self.config["reduced_neurons"],
                    self.config["learning_rate"],
                )
                local_model.set_weights(global_model.get_weights())

                start = time.time()
                weights = train_local_model(
                    local_model,
                    data["x"],
                    data["y"],
                    epochs=self.config["local_epochs"],
                    batch_size=self.config["batch_size"],
                )
                duration = time.time() - start

                with open(f"weights/{client_id}_round_{round_num+1}.json", "w") as f:
                    json.dump({k: v.tolist() for k, v in weights.items()}, f, indent=4)

                accuracy = evaluate_model(local_model, self.x_test, self.y_test)
                distribution = get_data_distribution(data["y"], self.num_classes)

                records.append(
                    {
                        "Client": client_id,
                        "Round": round_num + 1,
                        "Computation_Time": duration,
                        "Quality_Factor": accuracy,
                        "Data_Distribution": distribution,
                    }
                )
                client_weights.append(weights)

            print("Aggregating client weights...")
            new_global_weights = aggregate_weights(client_weights)
            global_model.set_weights([new_global_weights[f"layer_{i}"] for i in range(len(new_global_weights))])
            with open(f"weights/global_round_{round_num+1}.json", "w") as f:
                json.dump({k: v.tolist() for k, v in new_global_weights.items()}, f, indent=4)

        df = pd.DataFrame(records)

        mean_acc = (
            df.groupby("Client")["Quality_Factor"].mean().reset_index().rename(
                columns={"Quality_Factor": "Reliability_Score"}
            )
        )
        df = df.merge(mean_acc, on="Client")
        metrics_df = compute_binary_vector(df)
        return pd.concat([df, metrics_df], axis=1)