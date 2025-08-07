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


def get_data_distribution(y, num_classes: int):
    """Return the label distribution as a dictionary."""
    distribution = {i: 0 for i in range(num_classes)}
    for label in y:
        distribution[int(label)] += 1
    return distribution