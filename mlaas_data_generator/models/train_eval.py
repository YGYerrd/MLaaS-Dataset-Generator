from __future__ import annotations
import numpy as np

def train_local_model(model, x, y, epochs: int = 1, batch_size: int = 32):
    """Train model on data and return weights dict {layer_i: array}."""
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return {f"layer_{i}": w for i, w in enumerate(model.get_weights())}

def aggregate_weights(client_weights):
    """Aggregate a list of weight dictionaries by simple FedAvg (mean)."""
    num_layers = len(client_weights[0])
    aggregated = {}
    for i in range(num_layers):
        aggregated[f"layer_{i}"] = np.mean([w[f"layer_{i}"] for w in client_weights], axis=0)
    return aggregated

def evaluate_model(model, x_test, y_test, task_type="classification"):
    """Return (loss, primary_metric, extra_metric). For classification: extra = macro-F1."""
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
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))
