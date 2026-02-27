import importlib
import numpy as np
from sklearn.model_selection import train_test_split

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

SKLEARN_DEFAULT_TASK = {
    "iris": "classification",
    "wine": "classification",
    "digits": "classification",
    "california_housing": "regression",
    "diabetes": "regression",
}

REGRESSION_DATASETS = {"california_housing", "diabetes"}
CLASSIFICATION_DATASETS = {"iris", "wine", "digits"}


def _import(path):
    mod, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), attr)


def _meta(task, input_shape, num_classes=None, feature_names=None):
    return {
        "task_type": task,
        "input_shape": tuple(input_shape),
        "num_classes": None if num_classes is None else int(num_classes),
        "feature_names": feature_names,
        "x_format": "array",
        "source_family": "generic",
    }


def load_keras_source(name):
    if name not in KERAS_DATASETS:
        raise ValueError(f"Unknown keras dataset: {name}")

    loader = _import(KERAS_DATASETS[name])
    (x_train, y_train), (x_test, y_test) = loader()

    # keep raw-ish; just enforce predictable dtypes/shapes
    y_train = y_train.squeeze().astype("int32")
    y_test = y_test.squeeze().astype("int32")

    # keep raw pixel scale (0..255) under strict option A
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    if x_train.ndim == 3:  # grayscale to (H,W,1)
        x_train = x_train[..., None]
        x_test = x_test[..., None]

    meta = _meta("classification", x_train.shape[1:], num_classes=int(np.max(y_train) + 1))
    meta.update({"dataset_name": name, "source": "keras"})
    return (x_train, y_train), (x_test, y_test), meta


def load_sklearn_source(name, task, test_size=0.2, seed=42):
    if name not in SKLEARN_DATASETS:
        raise ValueError(f"Unknown sklearn dataset: {name}")

    loader = _import(SKLEARN_DATASETS[name])
    bunch = loader()

    if task == "regression":
        if name not in REGRESSION_DATASETS:
            raise ValueError(f"Dataset '{name}' does not provide a regression target.")
        X = bunch.data.astype("float32")
        y = bunch.target.astype("float32")
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=None
        )
        meta = _meta(
            "regression",
            (x_train.shape[1],),
            num_classes=None,
            feature_names=getattr(bunch, "feature_names", None),
        )
        meta.update({"dataset_name": name, "source": "sklearn"})
        return (x_train, y_train), (x_test, y_test), meta

    if task == "classification":
        if name not in CLASSIFICATION_DATASETS:
            raise ValueError(f"Dataset '{name}' is not a classification dataset.")
        X = bunch.data.astype("float32")
        y = bunch.target.astype("int32")
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        meta = _meta(
            "classification",
            (x_train.shape[1],),
            num_classes=int(np.unique(y).size),
            feature_names=getattr(bunch, "feature_names", None),
        )
        meta.update({"dataset_name": name, "source": "sklearn"})
        return (x_train, y_train), (x_test, y_test), meta

    if task == "clustering":
        X = bunch.data.astype("float32")
        y = bunch.target.astype("int32")
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=None
        )
        meta = _meta(
            "clustering",
            (x_train.shape[1],),
            num_classes=int(np.unique(y).size),
            feature_names=getattr(bunch, "feature_names", None),
        )
        meta.update({"dataset_name": name, "source": "sklearn"})
        return (x_train, y_train), (x_test, y_test), meta

    raise ValueError(f"Unknown task '{task}'. Use 'regression'|'classification'|'clustering'.")


def load_csv_source(csv_path, target, task="regression", test_size=0.2, seed=42):
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

    meta = _meta(
        task,
        (x_train.shape[1],),
        num_classes=None if task == "regression" else int(np.max(y) + 1),
        feature_names=list(df.drop(columns=[target]).columns),
    )
    meta.update({"dataset_name": csv_path, "source": "csv", "target_column": target})
    return (x_train, y_train), (x_test, y_test), meta