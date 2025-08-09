"""Configuration for the MLaaS data generator."""
from pathlib import Path

BASE_OUTPUT_DIR = Path("outputs")
RUNS_DIR = BASE_OUTPUT_DIR / "runs" 
MERGED_DIR = BASE_OUTPUT_DIR / "merged" 

import os
OVERRIDE = os.getenv("MLAAS_OUTDIR")
if OVERRIDE:
    BASE_OUTPUT_DIR = Path(OVERRIDE)
    RUNS_DIR = BASE_OUTPUT_DIR / "runs"
    MERGED_DIR = BASE_OUTPUT_DIR / "merged"

CONFIG = {
    "num_clients": 20,
    "num_rounds": 5,
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.01,
    "reduced_neurons": 64,
}