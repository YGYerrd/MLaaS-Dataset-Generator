"""Model helpers for the MLaaS data generator."""

from __future__ import annotations

from tensorflow.keras import layers, models, optimizers


def create_model(input_shape, num_classes, reduced_neurons: int = 64, learning_rate: float = 0.01):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(reduced_neurons, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    opt = optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_local_model(model, x, y, epochs: int = 1, batch_size: int = 32):
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return {f"layer_{i}": w for i, w in enumerate(model.get_weights())}


def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return acc