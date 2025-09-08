"""Model helpers for the MLaaS data generator."""

from __future__ import annotations

import numpy as np
from keras import layers, models, optimizers, regularizers
from sklearn.metrics import f1_score

def _make_optimizer(name, lr):
    name = name.lower()
    if name == "sgd":     return optimizers.SGD(learning_rate=lr, momentum=0.0)
    if name == "rmsprop": return optimizers.RMSprop(learning_rate=lr)
    if name == "adagrad": return optimizers.Adagrad(learning_rate=lr)
    if name == "adamw":   return optimizers.AdamW(learning_rate=lr)
    return optimizers.Adam(learning_rate=lr)

def create_model(input_shape, num_classes, hidden_layers=64, learning_rate=0.01, activation="relu", dropout=0.0, weight_decay=0.0, optimizer="adam"):
    l2 = regularizers.l2(weight_decay) if weight_decay > 0 else None
    
    model = models.Sequential(name="mlaas_cnn")

    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=l2))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=l2))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())

    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation, kernel_regularizer=l2))
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2))

    opt = _make_optimizer(optimizer, learning_rate)
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
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = y_pred.argmax(axis=1)
    f1 = f1_score(y_test, y_pred_classes, average="macro")
    return loss, acc, f1