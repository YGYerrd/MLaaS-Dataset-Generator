"""Utility functions for loading and splitting data."""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng, Generator

from tensorflow.keras.datasets import fashion_mnist, mnist


def load_dataset(name: str = "fashion_mnist"):
    """Load a dataset by name.
    Parameters
    ----------
    name: str
        Either ``"fashion_mnist"`` or ``"mnist"``.
    """
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test)



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
    for i in range(diff):
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
        g.shuffle(idxs)
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
    
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    if strategy == "iid":
        return split_iid(x, y, num_clients, rng=rng)

    if strategy == "quantity_skew":
        alpha = float(distribution_param) if distribution_param is not None else 1.0
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for quantity_skew.")
        return split_quantity_skew(x, y, num_clients, alpha, rng=rng)

    if strategy == "dirichlet":
        alpha = float(distribution_param) if distribution_param is not None else 0.5
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for dirichlet.")
        return split_dirichlet_label_skew(x, y, num_clients, alpha, rng=rng)

    if strategy == "shard":
        shards_per_client = int(distribution_param) if distribution_param is not None else 2
        if shards_per_client <= 0:
            raise ValueError("shards_per_client must be > 0 for shard.")
        return split_shard_based(x, y, num_clients, shards_per_client, rng=rng)

    if strategy == "label_per_client":
        k = int(distribution_param) if distribution_param is not None else 1
        if not (1 <= k <= int(np.max(y)) + 1):
            raise ValueError("k must be in [1, num_classes] for label_per_client.")
        return split_label_per_client(x, y, num_clients, k, rng=rng)

    if strategy == "custom":
        if custom_distributions is None or len(custom_distributions) == 0:
            raise ValueError("client_distributions must be provided for 'custom' strategy.")
        
        if len(custom_distributions) != num_clients:
            raise ValueError(
                f"client_distributions specifies {len(custom_distributions)} clients, "
                f"but num_clients={num_clients} was requested. "
                "These must match exactly."
            )

        return split_custom_data(x, y, custom_distributions, rng=rng)

    raise ValueError(f"Unknown data split strategy: {strategy}")



def get_data_distribution(y, num_classes: int):
    """Return the label distribution as a dictionary."""
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