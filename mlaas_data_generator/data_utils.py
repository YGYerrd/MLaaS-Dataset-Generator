"""Utility functions for loading and splitting data."""

from __future__ import annotations

import numpy as np
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

def split_data(x, y, num_clients: int):
    """Split ``(x, y)`` equally among ``num_clients``."""
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]

    data_per_client = len(x) // num_clients
    clients = {}
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client
        clients[f"client_{i+1}"] = {
            "x": x[start:end],
            "y": y[start:end],
        }
    return clients


def split_custom_data(x, y, client_distributions: dict):
    """Split ``(x, y)`` according to ``client_distributions``."""
    clients = {}
    for client, distribution in client_distributions.items():
        client_x, client_y = [], []
        for label, count in distribution.items():
            label = int(label)
            indices = np.where(y == label)[0]
            if len(indices) == 0:
                continue
            chosen = np.random.choice(indices, size=min(count, len(indices)), replace=False)
            client_x.extend(x[chosen])
            client_y.extend(y[chosen])

        client_x = np.array(client_x)
        client_y = np.array(client_y)
        shuffle_idx = np.random.permutation(len(client_y))

        clients[client] = {
            "x": client_x[shuffle_idx],
            "y": client_y[shuffle_idx],
        }
    return clients


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