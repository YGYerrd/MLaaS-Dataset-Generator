from __future__ import annotations
from typing import Sequence
from keras import layers, models, optimizers, regularizers
from .adapters import KMeansAdapter

def _make_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "sgd":     return optimizers.SGD(learning_rate=lr, momentum=0.0)
    if name == "rmsprop": return optimizers.RMSprop(learning_rate=lr)
    if name == "adagrad": return optimizers.Adagrad(learning_rate=lr)
    if name == "adamw":   return optimizers.AdamW(learning_rate=lr)
    return optimizers.Adam(learning_rate=lr)

def create_model(
    input_shape,
    num_classes,
    hidden_layers: Sequence[int] = (64,),
    learning_rate: float = 0.01,
    activation: str = "relu",
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    task_type: str = "classification",
    **kwargs
):
    l2 = regularizers.l2(weight_decay) if weight_decay > 0 else None
    rank = len(input_shape)

    if task_type == "clustering":
        k        = kwargs.get("k", kwargs.get("clustering_k", 3))
        init     = kwargs.get("clustering_init", "k-means++")
        n_init   = int(kwargs.get("clustering_n_init", 10))
        max_iter = int(kwargs.get("clustering_max_iter", 300))
        tol      = float(kwargs.get("clustering_tol", 1e-4))
        seed     = kwargs.get("random_state", kwargs.get("seed", None))
        return KMeansAdapter(
            input_shape=input_shape, k=k, init=init, n_init=n_init,
            max_iter=max_iter, tol=tol, random_state=seed
        )

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
