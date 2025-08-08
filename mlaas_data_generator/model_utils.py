"""Model helpers for the MLaaS data generator."""

from __future__ import annotations

import numpy as np
from tensorflow.keras import layers, models, optimizers


def create_model(input_shape, num_classes, reduced_neurons: int = 64, learning_rate: float = 0.01):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(reduced_neurons, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    opt = optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_local_model(model, x, y, epochs: int = 1, batch_size: int = 32):
    """Train model on data and return weights"""
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return {f"layer_{i}": w for i, w in enumerate(model.get_weights())}

def aggregate_weights(client_weights):
    "aggregate a list of weight dictionaries by averaging"
    num_layers = len(client_weights[0])
    aggregated = {}
    for i in range(num_layers):
        aggregated[f"layer_{i}"] = np.mean([w[f"layer_{i}"] for w in client_weights], axis = 0)
    return aggregated

def get_data_distribution(y):
    distribution = {i: 0 for i in range(10)}
    for label in y:
        distribution[label] += 1
    return distribution

def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return acc