# mlaas_data_generator/bench/test_runner.py
from __future__ import annotations
import os
import json
from datetime import datetime

from ..federated.orchestrator import FederatedDataGenerator


def _case(name, dataset_args, config_overrides):
    return {
        "name": name,
        "dataset_args": dataset_args,
        "config": config_overrides,
    }


def run_benchmarks(db_path="outputs/federated_bench.db"):
    run_group_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Keep these SMALL so you can iterate fast
    base_config = {
        "db_path": db_path,
        "dataset": "hf",
        "task_type": "classification",
        "model_type": "hf_finetune",          # important: triggers your finetune branch
        "num_clients": 3,
        "num_rounds": 3,
        "client_dropout_rate": 0.0,
        "batch_size": 16,
        "local_epochs": 1,                    # per-round local epochs
        "learning_rate": 5e-5,                # HF fine-tune lr (make sure your adapter uses this)
        "sample_size": 300,                   # small for testing
        "sample_frac": 0.5,
        "distribution_type": "iid",
        "distribution_bins": 10,
        "save_weights": False,
        "seed": 42,
    }

    # ---- Benchmark cases ----
    # Note: dataset_args keys must match your HF loader
    # Also note: you should use a *base model* for finetuning, not the already-finetuned sst2 head,
    # but we can include both to see behaviour differences.
    cases = [
        _case(
            name="wnut17_distilbert_token",
            dataset_args={
                "dataset_name": "wnut_17",
                "train_split": "train",
                "test_split": "validation",
                "tokens_column": "tokens",
                "label_column": "ner_tags",
                "max_samples": 500,
                "max_length": 128,
                "hf_model_id": "distilbert-base-uncased",
                "hf_task": "token_classification",
            },
            config_overrides={
                "learning_rate": 5e-5,
            },
        ),
        _case(
            name="sst2_distilbert_base",
            dataset_args={
                "dataset_name": "glue",
                "dataset_config": "sst2",
                "train_split": "train",
                "test_split": "validation",
                "text_column": "sentence",
                "label_column": "label",
                "max_samples": 600,
                "max_length": 128,
                "hf_model_id": "distilbert-base-uncased",
            },
            config_overrides={
                "learning_rate": 5e-5,
            },
        ),
        _case(
            name="sst2_distilbert_preft",
            dataset_args={
                "dataset_name": "glue",
                "dataset_config": "sst2",
                "train_split": "train",
                "test_split": "validation",
                "text_column": "sentence",
                "label_column": "label",
                "max_samples": 600,
                "max_length": 128,
                "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
            },
            config_overrides={
                "learning_rate": 2e-5,
            },
        ),
        
        _case(
            name="ag_news_distilbert_base",
            dataset_args={
                "dataset_name": "ag_news",
                "dataset_config": None,
                "train_split": "train",
                "test_split": "test",
                "text_column": "text",
                "label_column": "label",
                "max_samples": 800,
                "max_length": 128,
                "hf_model_id": "distilbert-base-uncased",
            },
            config_overrides={
                "learning_rate": 5e-5,
            },
        ),
        
    ]

    results = []
    print(f"\n=== BENCH RUN GROUP: {run_group_id} ===")
    print(f"DB: {db_path}\n")

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    for i, c in enumerate(cases, start=1):
        cfg = dict(base_config)
        cfg.update(c.get("config") or {})

        # Add metadata so you can filter later (stored via run_params if you write them)
        # If you don't yet store these, you can still keep them in a sidecar json.
        cfg["run_group_id"] = run_group_id
        cfg["case_name"] = c["name"]

        print(f"\n--- Case {i}/{len(cases)}: {c['name']} ---")
        print(f"  model: {c['dataset_args'].get('hf_model_id')}")
        print(f"  lr: {cfg.get('learning_rate')}, rounds: {cfg.get('num_rounds')}, clients: {cfg.get('num_clients')}")

        gen = FederatedDataGenerator(
            config=cfg,
            dataset="hf",
            task_type="classification",
            model_type="hf_finetune",
            dataset_args=c["dataset_args"],
        )

        out = gen.run()  # your orchestrator returns run_id/db info
        out["run_group_id"] = run_group_id
        out["case_name"] = c["name"]
        results.append(out)

        print(f"  -> run_id: {out.get('run_id')}")

    # Optional sidecar (handy even if you don’t store run_group_id in DB yet)
    sidecar = {
        "run_group_id": run_group_id,
        "db_path": db_path,
        "cases": results,
    }
    sidecar_path = os.path.join(os.path.dirname(db_path) or ".", f"bench_{run_group_id}.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"Sidecar: {sidecar_path}")
    return results


if __name__ == "__main__":
    run_benchmarks()