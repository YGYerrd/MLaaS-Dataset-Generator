from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
        if self._centers is None:
            return []
        return [self._centers.astype("float32")]

    def set_weights(self, weights_list):
        if not weights_list:
            return
        centers = np.asarray(weights_list[0], dtype="float32")
        self._centers = centers
        if self.km is None:
            self.km = KMeans(n_clusters=centers.shape[0], **self.kw)
        return

    def count_params(self):
        return 0 if self._centers is None else int(self._centers.size)
