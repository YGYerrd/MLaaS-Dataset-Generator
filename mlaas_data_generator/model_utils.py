"""Compat façade for model helpers.

Other modules may import:
    from .model_utils import create_model, train_local_model, evaluate_model, aggregate_weights, KMeansAdapter
This shim re-exports from the split modules.
"""
from __future__ import annotations

from .models.builders import create_model
from .models.train_eval import train_local_model, evaluate_model, aggregate_weights
from .models.adapters import KMeansAdapter

__all__ = [
    "create_model",
    "train_local_model",
    "evaluate_model",
    "aggregate_weights",
    "KMeansAdapter",
]
