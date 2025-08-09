"""Path resolver logic for saving outputs"""

from __future__ import annotations
from pathlib import Path
from typing import Literal
from .config import RUNS_DIR, MERGED_DIR

def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

def default_filename(stem: str, ext: str = ".csv") -> str:
    return f"{stem}{ext}"

def resolve_output_path(
    filename: str = "clients",
    kind: Literal["run", "merged"] = "run",) -> Path:
    ensure_dirs()
    base = RUNS_DIR if kind == "run" else MERGED_DIR
    fname = Path(filename).name
    if not fname.lower().endswith(".csv"):
        fname += ".csv"
    return base / filename

