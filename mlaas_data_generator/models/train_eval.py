from __future__ import annotations
import numpy as np

def train_local_model(model, x, y, epochs=1, batch_size=32):
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    if hasattr(model, "get_weights"):
        try:
            weights = model.get_weights()
            if weights is not None and len(weights) > 0:
                return {f"layer_{i}": w for i, w in enumerate(weights)}
        except Exception:
            pass
    return None


def aggregate_weights(client_weights):
    """Aggregate a list of weight dictionaries by simple FedAvg (mean)."""
    layers = list(client_weights[0].keys())
    for w in client_weights[1:]:
        assert list(w.keys()) == layers, "Mismatched layer keys across clients."
    aggregated = {}
    for i in range(layers):
        aggregated[f"layer_{i}"] = np.mean([w[f"layer_{i}"] for w in client_weights], axis=0)
    return aggregated

def evaluate_model(model, x_test, y_test, task_type="classification"):
    if hasattr(model, "evaluate"):
        results = model.evaluate(x_test, y_test, verbose=0)
        if isinstance(results, (list, tuple)):
            loss = float(results[0])
            metric = float(results[1]) if len(results) > 1 else float(results[0])
        else:
            loss = float(results); metric = float(results)
        if task_type == "regression":
            return loss, metric, None
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = y_pred.argmax(axis=1)  # Keras softmax path
        f1 = _macro_f1(y_test, y_pred_classes)
        return loss, metric, f1

    # Adapter path
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
    y_pred = model.predict(x_test)
    if task_type == "regression":
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        return mse, rmse, None
    # classification
    y_hat = y_pred if np.ndim(y_pred) == 1 else np.argmax(y_pred, axis=1)
    acc = float(accuracy_score(y_test, y_hat))
    f1m = float(f1_score(y_test, y_hat, average="macro", zero_division=0))
    return 1.0 - acc, acc, f1m


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
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))
