from __future__ import annotations
import numpy as np

def apply_feature_scaler(x_train, x_test, scaler):
    if not scaler:
        return x_train.astype("float32"), x_test.astype("float32"), None
    if scaler == "standard":
        from sklearn.preprocessing import StandardScaler as S
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler as S
    else:
        raise ValueError(f"Unknown scaler: {scaler}")
    s = S()
    x_train = s.fit_transform(x_train).astype("float32")
    x_test  = s.transform(x_test).astype("float32")
    return x_train, x_test, scaler


def apply_target_scaler(y_train, y_test, method: str | None):
    if not method or method == "none":
        return y_train.astype("float32"), y_test.astype("float32"), None
    if method == "standard":
        mean = float(np.mean(y_train))
        std  = float(np.std(y_train)) if np.std(y_train) > 0 else 1.0
        y_train = ((y_train - mean) / std).astype("float32")
        y_test  = ((y_test  - mean) / std).astype("float32")
        return y_train, y_test, {"type": "standard", "mean": mean, "std": std}
    raise ValueError(f"Unknown target scaler: {method}")