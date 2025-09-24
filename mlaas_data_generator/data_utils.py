"""Utility functions for loading and splitting data."""

from __future__ import annotations

import importlib, numpy as np
from numpy.random import default_rng, Generator
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
    "california_housing": "sklearn.datasets.fetch_california_housing"
}

def _import(path):
    mod, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), attr)

def _apply_scaler(x_train, x_test, scaler):
    if not scaler:
        return x_train.astype("float32"), x_test.astype("float32"), None
    if scaler == "standard":
        from sklearn.preprocessing import StandardScaler as S
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler as S
    else:
        raise ValueError(f"Unknown scaler: {scaler}")
    s = S()
    x_train = s.fit_transform(x_train).astype("float32")
    x_test  = s.transform(x_test).astype("float32")
    return x_train, x_test, scaler


def _apply_target_scaler(y_train, y_test, method: str | None):
    if not method or method == "none":
        return y_train.astype("float32"), y_test.astype("float32"), None
    if method == "standard":
        mean = float(np.mean(y_train))
        std  = float(np.std(y_train)) if np.std(y_train) > 0 else 1.0
        y_train = ((y_train - mean) / std).astype("float32")
        y_test  = ((y_test  - mean) / std).astype("float32")
        return y_train, y_test, {"type": "standard", "mean": mean, "std": std}
    raise ValueError(f"Unknown target scaler: {method}")


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

def _load_sklearn(name: str, test_size=0.2, seed=42, scaler="standard", y_standardize=True):
    loader = _import(SKLEARN_DATASETS[name])
    bunch = loader()
    
    if name == "california_housing":
        X = bunch.data.astype("float32")
        y = bunch.target.astype("float32")

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=None
        )
        x_train, x_test, scaler_used = _apply_scaler(x_train, x_test, scaler)

        y_train, y_test, y_scaler = _apply_target_scaler(y_train, y_test, "standard" if y_standardize else None)

        meta = {
            **_meta("regression", (x_train.shape[1],), num_classes=None,
                    feature_names=getattr(bunch, "feature_names", None), scaler=scaler_used),
            "target_scaler": y_scaler
        }
        return (x_train, y_train), (x_test, y_test), meta

    X = bunch.data.astype("float32")
    y = bunch.target.astype(int)
    stratify = y
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    x_train, x_test, scaler_used = _apply_scaler(x_train, x_test, scaler)
    meta = _meta(
        "clustering",
        (x_train.shape[1],),
        num_classes=int(np.unique(y).size),
        feature_names=getattr(bunch, "feature_names", None),
        scaler=scaler_used,
    )
    return (x_train, y_train), (x_test, y_test), meta


#Ignore for now
"""def _load_csv(csv_path: str, target: str, task: str = "regression",
              test_size=0.2, seed=42, scaler="standard"):
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
    x_train, x_test, scaler_used = _apply_scaler(x_train, x_test, scaler if task == "regression" else None)
    meta = _meta(
        task,
        (x_train.shape[1],),
        num_classes=None if task == "regression" else int(np.max(y) + 1),
        feature_names=list(df.drop(columns=[target]).columns),
        scaler=scaler_used,
    )
    return (x_train, y_train), (x_test, y_test), meta
"""


def load_dataset(name: str, **kwargs):
    """
    Families:
      A) keras: mnist, fashion_mnist, cifar10
      B) sklearn: iris, wine, digits, california_housing
      C) csv:   pass csv_path=..., target=..., task=('regression'|'classification')
    """
    key = name.lower()

    if key in KERAS_DATASETS:
        return _load_keras(key)

    if key in SKLEARN_DATASETS:
        return _load_sklearn(
            key,
            test_size=kwargs.get("test_size", 0.2),
            seed=kwargs.get("seed", 42),
            scaler=kwargs.get("scaler", "standard"),
            y_standardize=kwargs.get("y_standardize", True)
        )
    """if key == "csv":
        return _load_csv(
            csv_path=kwargs["csv_path"],
            target=kwargs["target"],
            task=kwargs.get("task", "regression"),
            test_size=kwargs.get("test_size", 0.2),
            seed=kwargs.get("seed", 42),
            scaler=kwargs.get("scaler", "standard"),
        )
    """
    raise KeyError(
        f"Unknown dataset '{name}'. Choices: {list(KERAS_DATASETS) + list(SKLEARN_DATASETS) + ['csv']}"
    )


def _build_clients_from_indices(x, y, indices_by_client: dict):
    clients = {}
    for cid, idx in indices_by_client.items():
        clients[cid] = {"x": x[idx], "y": y[idx]}
    return clients


def _seed(rng):
    return rng if isinstance(rng, Generator) else default_rng()



def split_iid(x, y, num_clients, rng=None):
    n = len(x)
    seed = _seed(rng)

    idx = seed.permutation(n)
    data_per_client = n // num_clients
    indices_by_client = {
        f"client_{i+1}" : idx[i*data_per_client: (i+1)*data_per_client] for i in range(num_clients)
    }
    return _build_clients_from_indices(x, y, indices_by_client)


def split_quantity_skew(x, y, num_clients, alpha, rng=None):
    """Split data with IID label distribution but uneven sample counts.
    `alpha` controls the difference in client sizes.
    Larger `alpha` results in more balanced client sizes.
    """
    n = len(x)
    seed = _seed(rng)

    proportions = seed.dirichlet([alpha] * num_clients)
    counts = (proportions * n).astype(int)

    # Fix rounding
    diff = n - counts.sum()
    for i in range(abs(diff)):
        counts[i % num_clients] += 1 if diff > 0 else -1

    idx = seed.permutation(n)
    indices_by_client = {}
    start = 0
    for i, count in enumerate(counts):
        end = start + count
        indices_by_client[f"client_{i+1}"] = idx[start:end]
        start = end
    return _build_clients_from_indices(x, y, indices_by_client)


def split_dirichlet_label_skew(x, y, num_clients, alpha, rng=None):
    """Split data so that each class is distributed via Dirichlet over clients."""
    seed = _seed(rng)
    num_classes = len(np.unique(y))
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]

    indices_by_client = {f"client_{i+1}": [] for i in range(num_clients)}

    for indices in class_indices:
        seed.shuffle(indices)
        proportions = seed.dirichlet([alpha] * num_clients)
        counts = (proportions * len(indices)).astype(int)
        diff = len(indices) - counts.sum()
        for i in range(abs(diff)):
            counts[i % num_clients] += 1 if diff > 0 else -1

        start = 0
        for i, count in enumerate(counts):
            end = start + max(count, 0)
            if end > start:
                indices_by_client[f"client_{i+1}"].extend(indices[start:end].tolist())
            start = end

    indices_by_client = {cid: np.asarray(idxs, dtype=int) for cid, idxs in indices_by_client.items()}
    return _build_clients_from_indices(x, y, indices_by_client)


def split_shard_based(x, y, num_clients, shards_per_client, rng=None):
    """Sort by label, split into shards, randomly assign shards to clients."""
    seed = _seed(rng)
    num_shards = num_clients * shards_per_client
    idx_sorted = np.argsort(y, kind="stable")
    shards = np.array_split(idx_sorted, num_shards)
    seed.shuffle(shards)
    indices_by_client = {}
    for i in range(num_clients):
        shard_indices = np.concatenate(shards[i * shards_per_client : (i + 1) * shards_per_client])
        indices_by_client[f"client_{i+1}"] = shard_indices
    return _build_clients_from_indices(x, y, indices_by_client)


def split_label_per_client(x, y, num_clients, k, rng=None):
    """Each client receives data from only k labels (chosen uniformly without replacement)."""
    seed = _seed(rng)
    num_classes = int(np.max(y)) + 1
    class_indices = {c: np.where(y == c)[0] for c in range(num_classes)}
    clients_labels = {i: seed.choice(num_classes, k, replace=False) for i in range(num_clients)}

    indices_by_client = {f"client_{i+1}": [] for i in range(num_clients)}
    for label, idxs in class_indices.items():
        recipients = [cid for cid, labels in clients_labels.items() if label in labels]
        if not recipients:
            continue
        seed.shuffle(idxs)
        splits = np.array_split(idxs, len(recipients))
        for cid, split in zip(recipients, splits):
            if len(split) > 0:
                indices_by_client[f"client_{cid+1}"].extend(split.tolist())

    indices_by_client = {cid: np.asarray(idxs, dtype=int) for cid, idxs in indices_by_client.items()}
    return _build_clients_from_indices(x, y, indices_by_client)


def split_custom_data(x, y, client_distributions: dict, rng=None):
    """Split ``(x, y)`` according to ``client_distributions``."""
    seed = _seed(rng)
    num_classes = int(np.max(y)) + 1

    # Build a mutable pool of available indices per label
    pool_by_label = {}
    for lbl in range(num_classes):
        idxs = np.where(y == lbl)[0]
        seed.shuffle(idxs)
        pool_by_label[lbl] = idxs

    # Allocate indices to clients based on requested counts
    indices_by_client = {cid: [] for cid in client_distributions.keys()}
    for cid, dist in client_distributions.items():
        for label_raw, count in dist.items():
            lbl = int(label_raw)
            if lbl not in pool_by_label:
                continue
            pool = pool_by_label[lbl]
            if len(pool) == 0 or count <= 0:
                continue
            take = min(int(count), len(pool))
            chosen, remaining = pool[:take], pool[take:]
            indices_by_client[cid].extend(chosen.tolist())
            pool_by_label[lbl] = remaining  # shrink pool

    indices_by_client = {cid: np.asarray(idxs, dtype=int) for cid, idxs in indices_by_client.items()}
    return _build_clients_from_indices(x, y, indices_by_client)



def _shrink_dataset(x, y, sample_size=None, sample_frac=None, rng=None):
    seed = _seed(rng)
    n = len(x)
    if sample_size is None and sample_frac is None:
        return x, y
    if sample_frac is not None:
        sample_size = int(round(n * float(sample_frac)))
    sample_size = max(0, min(n, int(sample_size)))
    idx = seed.choice(n, size=sample_size, replace=False)
    return x[idx], y[idx]



def split_data(x, y, num_clients, strategy = "iid", distribution_param = None, custom_distributions=None, sample_size=None, sample_frac=None, rng=None):
    strategy = strategy.lower()
    if sample_size or sample_frac:
        x,y = _shrink_dataset(x=x, y=y, sample_frac=sample_frac, sample_size=sample_size, rng=rng)
    
    resolved = {"strategy": strategy, "distribution_param": None}

    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    if strategy == "iid":
        return split_iid(x, y, num_clients, rng=rng), resolved

    if strategy == "quantity_skew":
        alpha = float(distribution_param) if distribution_param is not None else 1.0
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for quantity_skew.")
        resolved["distribution_param"] = alpha
        return split_quantity_skew(x, y, num_clients, alpha, rng=rng), resolved

    if strategy == "dirichlet":
        alpha = float(distribution_param) if distribution_param is not None else 0.5
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for dirichlet.")
        resolved["distribution_param"] = alpha
        return split_dirichlet_label_skew(x, y, num_clients, alpha, rng=rng), resolved

    if strategy == "shard":
        shards_per_client = int(distribution_param) if distribution_param is not None else 2
        if shards_per_client <= 0:
            raise ValueError("shards_per_client must be > 0 for shard.")
        resolved["distribution_param"] = shards_per_client
        return split_shard_based(x, y, num_clients, shards_per_client, rng=rng), resolved

    if strategy == "label_per_client":
        k = int(distribution_param) if distribution_param is not None else 1
        if not (1 <= k <= int(np.max(y)) + 1):
            raise ValueError("k must be in [1, num_classes] for label_per_client.")
        resolved["distribution_param"] = k
        return split_label_per_client(x, y, num_clients, k, rng=rng), resolved

    if strategy == "custom":
        if not custom_distributions:
                raise ValueError("custom_distributions must be provided for 'custom' strategy.'")
        adjusted = prepare_client_distributions(custom_distributions, num_clients)
        return split_custom_data(x, y, adjusted, rng=rng), resolved

    raise ValueError(f"Unknown data split strategy: {strategy}")


def get_data_distribution(
    y,
    num_classes: int | None,
    bins: int | None = None,
    value_range: tuple[float, float] | None = None,
):
    """Return the target distribution for a client dataset.

    For classification tasks ``num_classes`` should be provided and the return
    value is a mapping of class index to count. For regression tasks
    ``num_classes`` can be ``None`` and a histogram with ``bins`` buckets will
    be produced over ``value_range``.
    """

    if num_classes is None:
        if bins is None:
            bins = 10
        if value_range is not None:
            hist, _ = np.histogram(y, bins=bins, range=value_range)
        else:
            hist, _ = np.histogram(y, bins=bins)
        return {
            f"bin_{i}": int(hist[i])
            for i in range(len(hist))
        }

    distribution = {i: 0 for i in range(num_classes)}
    for label in y:
        distribution[int(label)] += 1
    return distribution


def generate_regular_distribution(num_clients: int, start_client: int = 1, num_labels: int = 10, samples_per_label: int = 100):
    regular_distributions = {}
    for i in range(start_client, num_clients + 1):
        regular_distributions[f"client_{i}"] = {
            label: samples_per_label for label in range(num_labels)
        }
    return regular_distributions


def prepare_client_distributions(custom_distributions: dict | None, num_clients: int):
    """Validate and extend custom distributions to match num_clients.

    If fewer distributions are provided than num_clients, regular distributions
    are generated for the remaining clients. If more are provided, the extra
    distributions are discarded. A warning is printed in both cases.
    """
    if custom_distributions is None:
        return None

    custom_distributions = {
        client: {int(label): count for label, count in dist.items()}
        for client, dist in custom_distributions.items()
    }

    num_custom = len(custom_distributions)
    if num_custom != num_clients:
        print(
            f"Warning: Provided distributions for {num_custom} clients, "
            f"but {num_clients} clients expected."
        )
        if num_custom < num_clients:
            start = num_custom + 1
            regular = generate_regular_distribution(num_clients, start)
            custom_distributions.update(regular)
        else:
            allowed = sorted(custom_distributions.keys())[:num_clients]
            custom_distributions = {k: custom_distributions[k] for k in allowed}

    return custom_distributions