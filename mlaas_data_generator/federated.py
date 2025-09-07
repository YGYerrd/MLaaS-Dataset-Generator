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
        dataset: str = "mnist",
        task_type: str = "classification",
        model_type: str = "CNN"
        ):
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.dataset = dataset
        self.task_type = task_type
        self.model_type = model_type

        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_dataset(dataset)
        
        self.x_train, self.y_train = shuffle(self.x_train,  self.y_train, random_state=42)
        
        self.input_shape = self.x_train.shape[1:]
        self.num_classes = len(np.unique(self.y_train))


    def run(self) -> pd.DataFrame:
        os.makedirs("weights", exist_ok=True)

        sample_size = self.config.get("sample_size")
        sample_frac = self.config.get("sample_frac") 
        distribution_type = self.config.get("distribution_type", "iid")
        distribution_param = self.config.get("distribution_param", None)
        custom_distributions = self.config.get("custom_distributions", None)

        clients, split_info = split_data(
            self.x_train,
            self.y_train,
            self.config["num_clients"],
            strategy=distribution_type,
            distribution_param=distribution_param,
            custom_distributions=custom_distributions,
            sample_size=sample_size,
            sample_frac=sample_frac
        )
        distribution_type = split_info["strategy"]
        distribution_param = split_info["distribution_param"]
        
        run_id = str(uuid.uuid4())
        dataset_name = self.dataset
        task_type = self.task_type
        model_type = self.model_type

        print(f"\nUsing split strategy: {distribution_type} | params: {distribution_param}\n")
        print(f"Using Dataset:{dataset_name}\n")

        print("Client data distributions before training:")
        client_data_distributions = {}
        for client_id, data in clients.items():
            distribution = get_data_distribution(data["y"], self.num_classes)
            client_data_distributions[client_id] = distribution
            print(f"{client_id}: {distribution}")

        global_model = create_model(
            self.input_shape,
            self.num_classes,
            self.config["reduced_neurons"],
            self.config["learning_rate"],
        )

        records = []
        global_accuracies = []

        for round_num in range(self.config["num_rounds"]):
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

                participated = True
                round_fail_reason = ""

                start = time.time()

                try:
                    weights = train_local_model(
                        local_model,
                        data["x"],
                        data["y"],
                        epochs=self.config["local_epochs"],
                        batch_size=self.config["batch_size"],
                    )
                
                except Exception:
                    participated = False
                    round_fail_reason = "error"
                    duration = time.time() - start
                    accuracy = np.nan
                    distribution = client_data_distributions[client_id]
                    samples_count = len(data["y"])

                    records.append(
                        {
                            "run_id": run_id,
                            "round": round_num + 1,
                            "client_id": client_id,
                            "dataset": dataset_name,
                            "task_type": task_type,
                            "model_type": model_type,
                            "participated": participated,
                            "round_fail_reason": round_fail_reason,
                            "distribution_type": distribution_type,
                            "distribution_param": distribution_param,
                            "data_Distribution": distribution,
                            "samples_count": samples_count,
                            "Computation_Time": duration,
                            "quality_Factor": accuracy,
                        }
                    )
                    continue

                duration = time.time() - start

                with open(f"weights/{client_id}_round_{round_num+1}.json", "w") as f:
                    json.dump({k: v.tolist() for k, v in weights.items()}, f, indent=4)

                accuracy = evaluate_model(local_model, self.x_test, self.y_test)
                distribution = client_data_distributions[client_id]
                samples_count = len(data["y"])

                actual_distribution = get_data_distribution(data["y"], self.num_classes)
                expected_distribution = client_data_distributions.get(client_id)

                if actual_distribution != expected_distribution:
                    print(f"Warning: distribution mismatch for {client_id}")

                records.append(
                    {
                        "run_id": run_id,
                        "round": round_num + 1,
                        "client_id": client_id,
                        "dataset": dataset_name,
                        "task_type": task_type,
                        "model_type": model_type,
                        "participated": participated,
                        "round_fail_reason": round_fail_reason,
                        "distribution_type": distribution_type,
                        "distribution_param": distribution_param,
                        "data_Distribution": distribution,
                        "samples_count": samples_count,
                        "Computation_Time": duration,
                        "quality_Factor": accuracy,
                    }
                )
                client_weights.append(weights)

            print("Aggregating client weights...")
            new_global_weights = aggregate_weights(client_weights)
            global_model.set_weights(
                [new_global_weights[f"layer_{i}"] for i in range(len(new_global_weights))]
            )
            global_accuracy = evaluate_model(
                global_model, self.x_test, self.y_test
            )
            print(f"Global model accuracy: {global_accuracy}")
            global_accuracies.append(
                {"Round": round_num + 1, "Global_Accuracy": global_accuracy}
            )
            with open(f"weights/global_round_{round_num+1}.json", "w") as f:
                json.dump({k: v.tolist() for k, v in new_global_weights.items()}, f, indent=4)
 
        with open("weights/global_accuracies.json", "w") as f:
            json.dump(global_accuracies, f, indent=4)

        self.global_accuracies = global_accuracies
        
        df = pd.DataFrame(records)

        mean_acc = (
            df.groupby("client_id")["quality_Factor"].mean().reset_index().rename(
                columns={"quality_Factor": "reliability_Score"}
            )
        )
        df = df.merge(mean_acc, on="client_id")
        print("Federated Learning Process Complete!\n")
       
        return df
