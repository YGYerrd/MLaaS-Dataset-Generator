import importlib, numpy as np
from sklearn.model_selection import train_test_split
from .scaling import apply_feature_scaler, apply_target_scaler

KERAS_DATASETS = {
    "mnist": "tensorflow.keras.datasets.mnist.load_data",
    "fashion_mnist": "tensorflow.keras.datasets.fashion_mnist.load_data",
    "cifar10": "tensorflow.keras.datasets.cifar10.load_data",
}

SKLEARN_DATASETS = {
    "iris": "sklearn.datasets.load_iris",
    "wine": "sklearn.datasets.load_wine",
    "digits": "sklearn.datasets.load_digits",
    "california_housing": "sklearn.datasets.fetch_california_housing",
    "diabetes": "sklearn.datasets.load_diabetes",
}

# Default task per sklearn dataset (caller can override with task=...)
SKLEARN_DEFAULT_TASK = {
    "iris": "classification",
    "wine": "classification",
    "digits": "classification",
    "california_housing": "regression",
    "diabetes": "regression",
}

REGRESSION_DATASETS = {"california_housing", "diabetes"}
CLASSIFICATION_DATASETS = {"iris", "wine", "digits"}  # can also be clustered if requested

def _import(path):
    mod, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), attr)

def _meta(task, input_shape, num_classes=None, feature_names=None, scaler=None):
    return {
        "task_type": task,
        "input_shape": tuple(input_shape),
        "num_classes": None if num_classes is None else num_classes,
        "feature_names": feature_names,
        "scaler": scaler,
    }

def _load_keras(name: str):
    loader = _import(KERAS_DATASETS[name])
    (x_train, y_train), (x_test, y_test) = loader()
    y_train = y_train.squeeze().astype(int)
    y_test = y_test.squeeze().astype(int)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    if x_train.ndim == 3:  # grayscale to (H,W,1)
        x_train = x_train[..., None]
        x_test = x_test[..., None]
    meta = _meta("classification", x_train.shape[1:], num_classes=int(np.max(y_train) + 1))
    return (x_train, y_train), (x_test, y_test), meta

def _load_sklearn(
    name: str,
    task: str,
    test_size=0.2,
    seed=42,
    scaler="standard",
    y_standardize=True,
):
    loader = _import(SKLEARN_DATASETS[name])
    bunch = loader()

    # --- REGRESSION ---
    if task == "regression":
        if name not in REGRESSION_DATASETS:
            raise ValueError(f"Dataset '{name}' does not provide a regression target.")
        X = bunch.data.astype("float32")
        y = bunch.target.astype("float32")

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=None
        )
        x_train, x_test, scaler_used = apply_feature_scaler(x_train, x_test, scaler)
        y_train, y_test, y_scaler = apply_target_scaler(
            y_train, y_test, "standard" if y_standardize else None
        )

        meta = {
            **_meta(
                "regression",
                (x_train.shape[1],),
                num_classes=None,
                feature_names=getattr(bunch, "feature_names", None),
                scaler=scaler_used,
            ),
            "target_scaler": y_scaler,
        }
        return (x_train, y_train), (x_test, y_test), meta

    # --- CLASSIFICATION ---
    if task == "classification":
        if name not in CLASSIFICATION_DATASETS:
            raise ValueError(f"Dataset '{name}' is not a classification dataset.")
        X = bunch.data.astype("float32")
        y = bunch.target.astype(int)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        x_train, x_test, scaler_used = apply_feature_scaler(x_train, x_test, scaler)
        meta = _meta(
            "classification",
            (x_train.shape[1],),
            num_classes=int(np.unique(y).size),
            feature_names=getattr(bunch, "feature_names", None),
            scaler=scaler_used,
        )
        return (x_train, y_train), (x_test, y_test), meta

    # --- CLUSTERING (default alternative for classification datasets) ---
    if task == "clustering":
        # We still return y so downstream can compute ARI/NMI, but do NOT stratify splits.
        X = bunch.data.astype("float32")
        y = bunch.target.astype(int)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=None
        )
        x_train, x_test, scaler_used = apply_feature_scaler(x_train, x_test, scaler)
        meta = _meta(
            "clustering",
            (x_train.shape[1],),
            num_classes=int(np.unique(y).size),
            feature_names=getattr(bunch, "feature_names", None),
            scaler=scaler_used,
        )
        return (x_train, y_train), (x_test, y_test), meta

    raise ValueError(f"Unknown task '{task}'. Use 'regression'|'classification'|'clustering'.")

def _load_csv(
    csv_path: str,
    target: str,
    task: str = "regression",
    test_size=0.2,
    seed=42,
    scaler="standard",
):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in CSV columns: {list(df.columns)}")
    y = df[target].to_numpy(dtype="float32" if task == "regression" else "int32")
    X = df.drop(columns=[target]).to_numpy(dtype="float32")
    stratify = None if task == "regression" else y
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    x_train, x_test, scaler_used = apply_feature_scaler(
        x_train, x_test, scaler if task == "regression" else None
    )
    meta = _meta(
        task,
        (x_train.shape[1],),
        num_classes=None if task == "regression" else int(np.max(y) + 1),
        feature_names=list(df.drop(columns=[target]).columns),
        scaler=scaler_used,
    )
    return (x_train, y_train), (x_test, y_test), meta

def load_dataset(name: str, **kwargs):
    """
    Families:
      A) keras: mnist, fashion_mnist, cifar10  (task = classification)
      B) sklearn: iris, wine, digits, california_housing, diabetes
         -> supply task=('classification'|'clustering'|'regression'), or omit to use per-dataset defaults:
            iris/wine/digits -> classification
            california_housing/diabetes -> regression
      C) csv:   pass csv_path=..., target=..., task=('regression'|'classification')
    """
    key = name.lower()

    if key in KERAS_DATASETS:
        return _load_keras(key)

    if key in SKLEARN_DATASETS:
        # Resolve task: explicit kwarg wins, else default for that dataset.
        task = kwargs.get("task", SKLEARN_DEFAULT_TASK.get(key, "classification"))
        return _load_sklearn(
            key,
            task=task,
            test_size=kwargs.get("test_size", 0.2),
            seed=kwargs.get("seed", 42),
            scaler=kwargs.get("scaler", "standard"),
            y_standardize=kwargs.get("y_standardize", True),
        )

    if key == "csv":
        return _load_csv(
            csv_path=kwargs["csv_path"],
            target=kwargs["target"],
            task=kwargs.get("task", "regression"),
            test_size=kwargs.get("test_size", 0.2),
            seed=kwargs.get("seed", 42),
            scaler=kwargs.get("scaler", "standard"),
        )

    raise KeyError(
        f"Unknown dataset '{name}'. Choices: {list(KERAS_DATASETS) + list(SKLEARN_DATASETS) + ['csv']}"
    )
