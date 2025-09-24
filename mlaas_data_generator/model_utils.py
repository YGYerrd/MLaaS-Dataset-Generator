"""Model helpers for the MLaaS data generator."""

from __future__ import annotations

import numpy as np
from keras import layers, models, optimizers, regularizers


def _make_optimizer(name, lr):
    name = name.lower()
    if name == "sgd":     return optimizers.SGD(learning_rate=lr, momentum=0.0)
    if name == "rmsprop": return optimizers.RMSprop(learning_rate=lr)
    if name == "adagrad": return optimizers.Adagrad(learning_rate=lr)
    if name == "adamw":   return optimizers.AdamW(learning_rate=lr)
    return optimizers.Adam(learning_rate=lr)

def create_model(input_shape, num_classes, hidden_layers=64, learning_rate=0.01, activation="relu", dropout=0.0, weight_decay=0.0, optimizer="adam", task_type="classification"):
    l2 = regularizers.l2(weight_decay) if weight_decay > 0 else None
    rank = len(input_shape)

    is_regression = (task_type == "regression")
    out_units = 1 if is_regression else int(num_classes)
    out_activation = "linear" if is_regression else "softmax"
    loss = "mse" if is_regression else "sparse_categorical_crossentropy"
    metrics = ["mse"] if is_regression else ["accuracy"]
    

    if rank == 3:
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

        model.add(layers.Dense(out_units, activation=out_activation, kernel_regularizer=l2))
    
    elif rank == 1:
        model = models.Sequential(name="mlaas_mlp")
        model.add(layers.Input(shape=input_shape))
        for units in hidden_layers:
            model.add(layers.Dense(units, activation=activation, kernel_regularizer=l2))
            if dropout and dropout > 0:
                model.add(layers.Dropout(dropout))
        model.add(layers.Dense(out_units, activation=out_activation, kernel_regularizer=l2))

    else:
        raise ValueError(f"Unsupported input_shape {input_shape}; rank {rank} not handled.")

    opt = _make_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
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


def evaluate_model(model, x_test, y_test, task_type="classification"):
    results = model.evaluate(x_test, y_test, verbose=0)
    if isinstance(results, (list, tuple)):
        loss = float(results[0])
        metric = float(results[1]) if len(results) > 1 else float(results[0])
    else:
        loss = float(results)
        metric = float(results)

    if task_type == "regression":
        return loss, metric, None
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = y_pred.argmax(axis=1)
    f1 = _macro_f1(y_test, y_pred_classes)
    return loss, metric, f1


def _macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if labels.size == 0:
        return 0.0
    scores = []
    for lbl in labels:
        tp = np.sum((y_true == lbl) & (y_pred == lbl))
        fp = np.sum((y_true != lbl) & (y_pred == lbl))
        fn = np.sum((y_true == lbl) & (y_pred != lbl))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))