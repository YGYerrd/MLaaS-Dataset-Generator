from __future__ import annotations
from typing import Sequence
from keras import layers, models, optimizers, regularizers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from .adapters import KMeansAdapter, make_random_forest

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
    model_type: str | None = None,
    **kwargs
):
    l2 = regularizers.l2(weight_decay) if weight_decay > 0 else None
    rank = len(input_shape)
    model_choice = (model_type or ("cnn" if rank == 3 else "mlp")).lower()

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
        if model_choice == "mobilenetv2":
            base = MobileNetV2(include_top=False, weights="imagenet", pooling=None)
            base.trainable = bool(kwargs.get("mobilenet_trainable", False))

            inputs = layers.Input(shape=input_shape)
            x = inputs
            if input_shape[-1] == 1:
                x = layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(x)
            elif input_shape[-1] != 3:
                raise ValueError(f"MobileNetV2 expects 1 or 3 channel input; got shape {input_shape}")
            x = layers.Resizing(96, 96)(x)
            x = layers.Lambda(lambda t: tf.cast(t, tf.float32))(x)
            x = layers.Lambda(preprocess_input)(x)
            x = base(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)
            outputs = layers.Dense(out_units, activation=out_activation, kernel_regularizer=l2)(x)
            model = models.Model(inputs=inputs, outputs=outputs, name="mlaas_mobilenetv2")
        else:
            model = models.Sequential(name="mlaas_cnn")
            model.add(layers.Input(shape=input_shape))
            model.add(layers.Conv2D(32, 3, padding="same", activation=activation, kernel_regularizer=l2))
            model.add(layers.MaxPooling2D())
            model.add(layers.Conv2D(64, 3, padding="same", activation=activation, kernel_regularizer=l2))
            model.add(layers.MaxPooling2D())
            model.add(layers.Flatten())
            for units in hidden_layers:
                model.add(layers.Dense(units, activation=activation, kernel_regularizer=l2))
                if dropout > 0:
                    model.add(layers.Dropout(dropout))
            model.add(layers.Dense(out_units, activation=out_activation, kernel_regularizer=l2))

    elif rank == 1:
        if model_choice == "randomforest":
            rf_kwargs = {
                "n_estimators": int(kwargs.get("rf_trees", kwargs.get("n_estimators", 100))),
                "max_depth": kwargs.get("rf_max_depth", kwargs.get("max_depth", None)),
                "random_state": kwargs.get("seed", kwargs.get("random_state", None)),
            }
            return make_random_forest(task_type="regression" if is_regression else "classification", **rf_kwargs)

        model_name = "mlaas_logreg" if model_choice == "logreg" else "mlaas_mlp"
        model = models.Sequential(name=model_name)

        model.add(layers.Input(shape=input_shape))
        layers_to_use = [] if model_choice == "logreg" else list(hidden_layers)
        for units in layers_to_use:
            model.add(layers.Dense(units, activation=activation, kernel_regularizer=l2))
            if dropout and dropout > 0:
                model.add(layers.Dropout(dropout))
        model.add(layers.Dense(out_units, activation=out_activation, kernel_regularizer=l2))

    else:
        raise ValueError(f"Unsupported input_shape {input_shape}; rank {rank} not handled.")

    opt = _make_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model
