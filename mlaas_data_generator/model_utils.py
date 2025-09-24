"""Model helpers for the MLaaS data generator."""

from __future__ import annotations

import numpy as np
from keras import layers, models, optimizers, regularizers

# --- Clustering adapter (K-Means) -------------------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

class KMeansAdapter:
    """
    Minimal "model-like" wrapper so the FL loop can treat K-Means as a model.
    - fit(X[, y])            : runs K-Means; stores labels_, cluster_centers_
    - predict(X)             : returns cluster indices
    - evaluate(X[, y_true])  : returns (loss, primary, extra) -> (nan, silhouette, inertia)
    - get_weights()/set_weights(): expose/ingest centers to mimic NN interface
    - count_params()         : number of elements in centers
    Notes:
      * If X is image-like (rank 3 or 4), we flatten to vectors.
      * Silhouette needs at least 2 clusters and 2 samples; else returns np.nan.
      * If y_true is provided (e.g., iris/wine/digits), we can compute ARI/NMI externally.
    """
    def __init__(self, input_shape, k=3, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=None):
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.k = int(k)
        self.kw = dict(init=init, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)
        self.km: KMeans | None = None
        self._centers = None
        self._labels_ = None

    def _flatten(self, X):
        X = np.asarray(X, dtype="float32")
        if X.ndim > 2:  # images or tensors -> flatten per sample
            return X.reshape((X.shape[0], -1))
        return X

    # parity with keras-like API bits used by your loop
    def fit(self, X, y=None, epochs=None, batch_size=None, verbose=0):
        Xf = self._flatten(X)
        self.km = KMeans(n_clusters=self.k, **self.kw).fit(Xf)
        self._centers = self.km.cluster_centers_
        self._labels_ = self.km.labels_
        return self

    def predict(self, X, verbose=0):
        if self.km is None:
            raise RuntimeError("KMeansAdapter not fitted.")
        return self.km.predict(self._flatten(X))

    def evaluate(self, X, y_true=None, verbose=0):
        if self.km is None:
            raise RuntimeError("KMeansAdapter not fitted.")
        Xf = self._flatten(X)
        labels = self.km.predict(Xf)
        inertia = float(self.km.inertia_) if hasattr(self.km, "inertia_") else np.nan
        # Silhouette is undefined for single cluster or too few samples
        try:
            sil = float(silhouette_score(Xf, labels)) if len(np.unique(labels)) > 1 and Xf.shape[0] > 2 else np.nan
        except Exception:
            sil = np.nan
        # we return (loss, primary_metric, extra_metric) to match your loop
        return (np.nan, sil, inertia)

    # mimic NN weight API so the federated code doesn't crash
    def get_weights(self):
        if self._centers is None:
            return []
        # return a list to resemble keras layer weights
        return [self._centers.astype("float32")]

    def set_weights(self, weights_list):
        if not weights_list:
            return
        centers = np.asarray(weights_list[0], dtype="float32")
        # If already fitted, update internal centers; else create a stub KMeans
        self._centers = centers
        if self.km is None:
            # Create a dummy KMeans object with provided centers
            self.km = KMeans(n_clusters=centers.shape[0], **self.kw)
            # sklearn doesn't support "setting" centers before fit, but we can simulate
            # by assigning attributes used by predict if needed; to keep it robust:
            # we'll do a tiny one-iteration "fit" later when fit/predict is called.
        return

    def count_params(self):
        return 0 if self._centers is None else int(self._centers.size)


def _make_optimizer(name, lr):
    name = name.lower()
    if name == "sgd":     return optimizers.SGD(learning_rate=lr, momentum=0.0)
    if name == "rmsprop": return optimizers.RMSprop(learning_rate=lr)
    if name == "adagrad": return optimizers.Adagrad(learning_rate=lr)
    if name == "adamw":   return optimizers.AdamW(learning_rate=lr)
    return optimizers.Adam(learning_rate=lr)

def create_model(input_shape, num_classes, hidden_layers=64, learning_rate=0.01, activation="relu", dropout=0.0, weight_decay=0.0, optimizer="adam", task_type="classification", **kwargs):
    l2 = regularizers.l2(weight_decay) if weight_decay > 0 else None
    rank = len(input_shape)

    is_regression = (task_type == "regression")
    out_units = 1 if is_regression else int(num_classes)
    out_activation = "linear" if is_regression else "softmax"
    loss = "mse" if is_regression else "sparse_categorical_crossentropy"
    metrics = ["mse"] if is_regression else ["accuracy"]
    
    if task_type == "clustering":
        k         = kwargs.get("k", kwargs.get("clustering_k", 3))
        init      = kwargs.get("clustering_init", "k-means++")
        n_init    = int(kwargs.get("clustering_n_init", 10))
        max_iter  = int(kwargs.get("clustering_max_iter", 300))
        tol       = float(kwargs.get("clustering_tol", 1e-4))
        seed      = kwargs.get("random_state", kwargs.get("seed", None))
        return KMeansAdapter(input_shape=input_shape, k=k, init=init, n_init=n_init,
                                max_iter=max_iter, tol=tol, random_state=seed)

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