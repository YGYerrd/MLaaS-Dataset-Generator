# cli/profiles.py
from __future__ import annotations

DATASET_CHOICES = [
    "fashion_mnist",
    "mnist",
    "cifar10",
    "digits",
    "iris",
    "wine",
    "california_housing",
]

def infer_dataset_profile(dataset: str) -> dict:
    ds = dataset.lower()

    # Vision datasets: default to classification, but clustering is fine (unsupervised on features)
    if ds in {"mnist", "fashion_mnist", "cifar10"}:
        return {
            "default_task": "classification",
            "tasks_supported": ["classification", "clustering"],
            "ask_split": False,
            "split_key": None,
            "ask_scaler": False,
            "default_model": "CNN",
            "allow_label_splits": True,
            "allow_quantity_skew": True,
            "allow_custom": True,
            "ask_target_scaler": False,
        }

    # Small tabular classification: can also do clustering
    if ds in {"iris", "wine", "digits"}:
        return {
            "default_task": "classification",
            "tasks_supported": ["classification", "clustering"],
            "ask_split": True,
            "split_key": "test_size",
            "ask_scaler": True,
            "default_model": "MLP",
            "allow_label_splits": True,
            "allow_quantity_skew": True,
            "allow_custom": True,
            "ask_target_scaler": False,
        }

    # Regression dataset: can also do clustering (unsupervised on features)
    if ds == "california_housing":
        return {
            "default_task": "regression",
            "tasks_supported": ["regression", "clustering"],
            "ask_split": True,
            "split_key": "test_size",
            "ask_scaler": True,
            "default_model": "MLP",
            "allow_label_splits": False,   # even if you pick clustering, we keep this conservative
            "allow_quantity_skew": True,
            "allow_custom": False,
            "ask_target_scaler": True,
        }

    # Fallback
    return {
        "default_task": "classification",
        "tasks_supported": ["classification", "clustering"],
        "ask_split": False,
        "split_key": None,
        "ask_scaler": False,
        "default_model": "MLP",
        "allow_label_splits": True,
        "allow_quantity_skew": True,
        "allow_custom": True,
        "ask_target_scaler": False,
    }
