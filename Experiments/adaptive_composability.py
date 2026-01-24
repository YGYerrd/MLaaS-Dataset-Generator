"""
adaptive_composability.py
Implements the Adaptive MLaaS Composability Model extracted from Experiment 1.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class AdaptiveComposabilityModel:
    """
    A simplified adaptive composability model that can operate on any dataset.

    Each client provides a set of service scores or predictions.
    The model composes them using rule-based or QoS-based logic.
    """

    def __init__(self, mode="rule", qos_weights=None, threshold=0.5):
        """
        mode: 'rule' or 'qos'
        qos_weights: dict of per-metric weights for QoS-based composition
        threshold: decision boundary for binary classification
        """
        self.mode = mode
        self.qos_weights = qos_weights or {}
        self.threshold = threshold

    # ---------------------------------------------------------------------
    # Composition logic
    # ---------------------------------------------------------------------
    def compose(self, df):
        """
        df : DataFrame with columns ['DUM','SUM','HQS','SRS','MUM', ...].
        Returns a vector of composed decisions (0/1 or probabilities).
        """
        metrics = ['DUM', 'SUM', 'HQS', 'SRS', 'MUM']
        df = df.copy()

        if self.mode == "rule":
            # Example: rule-based mean with binary threshold
            df["composite"] = df[metrics].mean(axis=1)
            df["decision"] = (df["composite"] >= self.threshold).astype(int)

        elif self.mode == "qos":
            # Weighted sum using provided QoS weights
            weights = np.array([self.qos_weights.get(m, 1.0) for m in metrics])
            df["composite"] = np.dot(df[metrics].values, weights) / weights.sum()
            df["decision"] = (df["composite"] >= self.threshold).astype(int)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return df["decision"].values, df

    # ---------------------------------------------------------------------
    # Evaluation logic
    # ---------------------------------------------------------------------
    def evaluate(self, y_true, y_pred):
        """Compute standard classification metrics."""
        metrics = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        }
        return metrics

    # ---------------------------------------------------------------------
    def fit_predict(self, df, y_true):
        """
        Convenience method for one-shot evaluation.
        """
        y_pred, composed = self.compose(df)
        results = self.evaluate(y_true, y_pred)
        return results, composed
