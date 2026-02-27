from .hf_strategy import HFStrategy
from .keras_strategy import ClassificationStrategy, RegressionStrategy
from .clustering import ClusteringStrategy
from .base import TaskStrategy


def make_task_strategy(task_type: str, meta: dict, knobs: dict, config: dict, x_test, y_test, metric_key: str, save_weights: bool) -> TaskStrategy:
    if task_type == "classification":
        mt = (config.get("model_type") or "").lower()
        
        if mt in ("hf", "hf_text", "transformers", "hf_finetune", "hf_train", "transformers_finetune"):
            return HFStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)
        
        return ClassificationStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)

    if task_type == "regression":
        return RegressionStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)

    if task_type == "clustering":
        return ClusteringStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)

    raise ValueError(f"Unknown task type: {task_type}")
