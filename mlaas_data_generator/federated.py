"""Federated learning based data generator."""

from __future__ import annotations

import time
import pandas as pd
import numpy as np

from .config import CONFIG
from .data_utils import load_dataset, split_data, get_data_distribution
from .model_utils import create_model, train_local_model, evaluate_model
from .metrics import compute_binary_vector


class FederatedDataGenerator:
    """Generate MLaaS client records using a simple federated-learning loop."""

    def __init__(self, config: dict | None = None, dataset: str = "fashion_mnist"):
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)
        self.dataset = dataset

        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_dataset(dataset)
        self.input_shape = self.x_train.shape[1:]
        self.num_classes = len(np.unique(self.y_train))

    def run(self) -> pd.DataFrame:
        clients = split_data(self.x_train, self.y_train, self.config["num_clients"])
        records = []

        for client_id, data in clients.items():
            model = create_model(
                self.input_shape,
                self.num_classes,
                self.config["reduced_neurons"],
                self.config["learning_rate"],
            )
            start = time.time()
            train_local_model(
                model,
                data["x"],
                data["y"],
                epochs=self.config["local_epochs"],
                batch_size=self.config["batch_size"],
            )
            duration = time.time() - start
            accuracy = evaluate_model(model, self.x_test, self.y_test)
            distribution = get_data_distribution(data["y"], self.num_classes)
            records.append(
                {
                    "Client": client_id,
                    "Computation_Time": duration,
                    "Quality_Factor": accuracy,
                    "Reliability_Score": 1.0,
                    "Data_Distribution": distribution,
                }
            )

        df = pd.DataFrame(records)
        metrics_df = compute_binary_vector(df)
        return pd.concat([df, metrics_df], axis=1)