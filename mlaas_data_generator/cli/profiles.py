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
    """
    Return a profile describing UX hints for the wizard.

    Fields:
      - default_task: "classification"/"regression"/"clustering"
      - tasks_supported: list of allowed tasks in the UI
      - is_image: bool → controls which model families are shown
      - default_model_pretty: human-facing default model label
      - ask_split: whether to ask for a train/test split
      - split_key: key name in dataset_args for the split fraction
      - ask_scaler: whether to ask for feature scaling
      - ask_target_scaler: whether to ask about y standardization (regression)
      - allow_label_splits: allow label-based partitioning strategies
      - allow_quantity_skew: allow quantity-skew strategy
      - allow_custom: allow custom distribution file
    """
    ds = dataset.lower()

    # Vision datasets: default classification, allow clustering as unsupervised
    if ds in {"mnist", "fashion_mnist"}:
        return {
            "default_task": "classification",
            "tasks_supported": ["classification", "clustering"],
            "is_image": True,
            "default_model_pretty": "CNN",     # lightweight conv is a sensible default
            "ask_split": False,
            "split_key": None,
            "ask_scaler": False,
            "ask_target_scaler": False,
            "allow_label_splits": True,
            "allow_quantity_skew": True,
            "allow_custom": True,
        }

    if ds == "cifar10":
        return {
            "default_task": "classification",
            "tasks_supported": ["classification", "clustering"],
            "is_image": True,
            "default_model_pretty": "MobileNetV2",  # better default backbone for color images
            "ask_split": False,
            "split_key": None,
            "ask_scaler": False,
            "ask_target_scaler": False,
            "allow_label_splits": True,
            "allow_quantity_skew": True,
            "allow_custom": True,
        }

    # Small tabular classification (can also try clustering)
    if ds in {"iris", "wine", "digits"}:
        return {
            "default_task": "classification",
            "tasks_supported": ["classification", "clustering"],
            "is_image": False,
            "default_model_pretty": "MLP",  # RF is strong, but MLP is a safe default across tasks
            "ask_split": True,
            "split_key": "test_size",
            "ask_scaler": True,
            "ask_target_scaler": False,
            "allow_label_splits": True,
            "allow_quantity_skew": True,
            "allow_custom": True,
        }

    # Regression (tabular). Clustering is still possible (unsupervised on features).
    if ds == "california_housing":
        return {
            "default_task": "regression",
            "tasks_supported": ["regression", "clustering"],
            "is_image": False,
            "default_model_pretty": "Random Forest",  # solid default for tabular regression
            "ask_split": True,
            "split_key": "test_size",
            "ask_scaler": True,
            "ask_target_scaler": True,
            "allow_label_splits": False,  # keep conservative even if user later picks clustering
            "allow_quantity_skew": True,
            "allow_custom": False,
        }

    # Fallback (treat as tabular classification)
    return {
        "default_task": "classification",
        "tasks_supported": ["classification", "clustering"],
        "is_image": False,
        "default_model_pretty": "MLP",
        "ask_split": False,
        "split_key": None,
        "ask_scaler": False,
        "ask_target_scaler": False,
        "allow_label_splits": True,
        "allow_quantity_skew": True,
        "allow_custom": True,
    }
