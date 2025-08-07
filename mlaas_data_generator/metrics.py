"""Metric utilities for MLaaS data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def flatten_weights(weights_dict):
    """Flatten a nested weight dictionary into a 1D array."""
    flat = []
    for layer in weights_dict.values():
        layer_arr = np.array(layer)
        flat.extend(layer_arr.ravel())
    return np.array(flat)


def calculate_mum(local_weights, global_weights):
    """Return the model update magnitude (MUM)."""
    lw = flatten_weights(local_weights)
    gw = flatten_weights(global_weights)
    return np.linalg.norm(lw - gw)


def compute_dum(client_distribution, global_reference, threshold: float = 400.0):
    if len(global_reference) == 0:
        return 0
    ed = np.sqrt(np.sum((np.array(list(client_distribution.values())) - global_reference) ** 2))
    return 1 if ed < threshold else 0


def compute_sum(response_time, avg_response_time, alpha: float = 1.0, threshold: float = 1.5):
    value = (response_time / avg_response_time) ** alpha
    return 1 if value < threshold else 0


def compute_hqs(quality_factor, avg_quality_factor, threshold: float = 0.05):
    return 1 if abs(quality_factor - avg_quality_factor) <= threshold else 0


def compute_srs(reliability_score, threshold: float = 0.7):
    return 1 if reliability_score >= threshold else 0


def compute_binary_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binary utility metrics for every client in ``df``."""
    client_distributions = np.array([list(dist.values()) for dist in df["Data_Distribution"]])
    global_reference = np.mean(client_distributions, axis=0) if len(client_distributions) > 0 else np.array([])
    avg_response_time = df["Computation_Time"].mean()
    avg_quality_factor = df["Quality_Factor"].mean()

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "DUM": compute_dum(row["Data_Distribution"], global_reference),
                "SUM": compute_sum(row["Computation_Time"], avg_response_time),
                "HQS": compute_hqs(row["Quality_Factor"], avg_quality_factor),
                "SRS": compute_srs(row.get("Reliability_Score", 1.0)),
            }
        )
    return pd.DataFrame(rows)