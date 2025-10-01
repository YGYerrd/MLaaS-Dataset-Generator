from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, mean_squared_error

class KMeansAdapter:
    """
    Minimal "model-like" wrapper so the FL loop can treat K-Means as a model.
    - fit(X[, y])
    - predict(X)
    - evaluate(X[, y_true]) -> (loss, silhouette, inertia)
    - get_weights()/set_weights()
    - count_params()
    Notes:
      * If X is image-like (rank 3 or 4), we flatten to vectors.
      * Silhouette needs at least 2 clusters and 2 samples; else returns np.nan.
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
        if X.ndim > 2:  # images/tensors -> flatten per sample
            return X.reshape((X.shape[0], -1))
        return X

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
        try:
            sil = float(silhouette_score(Xf, labels)) if len(np.unique(labels)) > 1 and Xf.shape[0] > 2 else np.nan
        except Exception:
            sil = np.nan
        return (np.nan, sil, inertia)

    def get_weights(self):
        return []

    def set_weights(self, weights_list):
        return

    def count_params(self):
        return 0 if self._centers is None else int(self._centers.size)


class EstimatorAdapter:
    """Minimal wrapper to present sklearn estimators with a Keras-like surface."""

    def __init__(self, estimator, task_type: str):
        self.estimator = estimator
        self._task_type = task_type

    def _reshape(self, X):
        X = np.asarray(X)
        if X.ndim > 2:
            return X.reshape((X.shape[0], -1))
        return X

    def fit(self, X, y, epochs=None, batch_size=None, verbose=0):
        self.estimator.fit(self._reshape(X), y)
        return self

    def predict(self, X, verbose=0):
        Xr = self._reshape(X)
        if self._task_type == "classification":
            if hasattr(self.estimator, "predict_proba"):
                return self.estimator.predict_proba(Xr)
            preds = self.estimator.predict(Xr)
            classes = getattr(self.estimator, "classes_", None)
            if classes is not None:
                class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
                eye = np.eye(len(classes), dtype="float32")
                return eye[[class_to_idx[p] for p in preds]]
            return preds
        return self.estimator.predict(Xr)

    def evaluate(self, X, y_true, verbose=0):
        Xr = self._reshape(X)
        if self._task_type == "classification":
            y_pred = self.estimator.predict(Xr)
            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            return np.nan, acc, f1

        y_pred = self.estimator.predict(Xr)
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        return mse, rmse, None

    def get_weights(self):
        return []

    def set_weights(self, weights_list):
        return

    def count_params(self):
        return 0


def make_random_forest(task_type: str, **kwargs) -> EstimatorAdapter:
    if task_type == "regression":
        estimator = RandomForestRegressor(**kwargs)
    else:
        estimator = RandomForestClassifier(**kwargs)
    return EstimatorAdapter(estimator, task_type)