"""Utility helpers for capturing system and hardware metrics during a run."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import psutil

MB = 1024 * 1024


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        if f != f:  # NaN guard
            return None
        return f
    except Exception:
        return None


def query_gpu_snapshot() -> List[Dict[str, Any]]:
    """Return per-GPU utilization information if nvidia-smi is available."""

    binary = shutil.which("nvidia-smi")
    if not binary:
        return []

    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    gpus: List[Dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        name, mem_total, mem_used, util_gpu, util_mem = parts
        gpus.append(
            {
                "name": name,
                "memory_total_mb": _safe_float(mem_total),
                "memory_used_mb": _safe_float(mem_used),
                "utilization": _safe_float(util_gpu),
                "memory_utilization": _safe_float(util_mem),
            }
        )
    return gpus


def capture_hardware_snapshot() -> Dict[str, Any]:
    """Capture a static snapshot of the environment for the run metadata."""

    try:
        freq = psutil.cpu_freq()
        cpu_freq = {
            "current_mhz": _safe_float(freq.current) if freq else None,
            "min_mhz": _safe_float(freq.min) if freq else None,
            "max_mhz": _safe_float(freq.max) if freq else None,
        }
    except Exception:
        cpu_freq = None

    try:
        vm = psutil.virtual_memory()
        memory = {
            "total_mb": _safe_float(vm.total / MB),
            "available_mb": _safe_float(vm.available / MB),
        }
    except Exception:
        memory = None

    snapshot: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "frequency": cpu_freq,
        },
        "memory": memory,
        "gpu": query_gpu_snapshot(),
        "environment": {
            "pid": os.getpid(),
            "working_directory": os.getcwd(),
        },
    }
    return snapshot


@dataclass
class ResourceUsage:
    cpu_time_s: float | None
    cpu_utilization: float | None
    memory_used_mb: float | None
    memory_utilization: float | None
    gpu_utilization: float | None
    gpu_memory_utilization: float | None
    gpu_memory_used_mb: float | None


class ResourceTracker:
    """Measure process resource usage across an execution block."""

    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self._cpu_times_start: psutil._common.pcputimes | None = None
        self._cpu_count = psutil.cpu_count(logical=True) or 1

    def start(self) -> None:
        try:
            self._process.cpu_percent(None)
        except Exception:
            pass
        try:
            self._cpu_times_start = self._process.cpu_times()
        except Exception:
            self._cpu_times_start = None

    def stop(self, duration_s: float | None = None) -> ResourceUsage:
        cpu_time_s: float | None = None
        try:
            end_times = self._process.cpu_times()
            if self._cpu_times_start is not None:
                cpu_time_s = (
                    (end_times.user - self._cpu_times_start.user)
                    + (end_times.system - self._cpu_times_start.system)
                )
        except Exception:
            cpu_time_s = None

        cpu_util: float | None = None
        if cpu_time_s is not None and duration_s and duration_s > 0:
            try:
                cpu_util = max(0.0, min(100.0, (cpu_time_s / duration_s) / self._cpu_count * 100.0))
            except Exception:
                cpu_util = None

        try:
            memory_used_mb = self._process.memory_info().rss / MB
        except Exception:
            memory_used_mb = None

        try:
            memory_util = psutil.virtual_memory().percent
        except Exception:
            memory_util = None

        gpu_snapshot = query_gpu_snapshot()
        gpu_util = _aggregate_gpu_metric(gpu_snapshot, "utilization")
        gpu_mem_util = _aggregate_gpu_metric(gpu_snapshot, "memory_utilization")
        gpu_mem_used = _aggregate_gpu_sum(gpu_snapshot, "memory_used_mb")

        return ResourceUsage(
            cpu_time_s=_safe_float(cpu_time_s),
            cpu_utilization=_safe_float(cpu_util),
            memory_used_mb=_safe_float(memory_used_mb),
            memory_utilization=_safe_float(memory_util),
            gpu_utilization=_safe_float(gpu_util),
            gpu_memory_utilization=_safe_float(gpu_mem_util),
            gpu_memory_used_mb=_safe_float(gpu_mem_used),
        )


def _aggregate_gpu_metric(gpus: Iterable[Dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(g.get(key)) for g in gpus]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return max(values)


def _aggregate_gpu_sum(gpus: Iterable[Dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(g.get(key)) for g in gpus]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values)


def summarize_round_usage(
    outcomes: Iterable[Dict[str, Any]],
    scheduled_clients: int,
    skipped_clients: int,
) -> Dict[str, Any]:
    """Aggregate client resource metrics for storage alongside round data."""

    # Materialise so we can iterate multiple times.
    outcomes_list = list(outcomes)

    def _pair(values: List[float | None]) -> Tuple[float | None, float | None]:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return None, None
        return float(mean(filtered)), float(max(filtered))

    durations = [o.get("duration") for o in outcomes_list]
    cpu_utils = [o.get("cpu_utilization") for o in outcomes_list]
    mem_utils = [o.get("memory_utilization") for o in outcomes_list]
    mem_used = [o.get("memory_used_mb") for o in outcomes_list]
    gpu_utils = [o.get("gpu_utilization") for o in outcomes_list]
    gpu_mem_utils = [o.get("gpu_memory_utilization") for o in outcomes_list]
    gpu_mem_used = [o.get("gpu_memory_used_mb") for o in outcomes_list]
    cpu_times = [o.get("cpu_time_s") for o in outcomes_list]

    avg_duration, max_duration = _pair(durations)
    avg_cpu_util, max_cpu_util = _pair(cpu_utils)
    avg_mem_util, max_mem_util = _pair(mem_utils)
    avg_mem_used, max_mem_used = _pair(mem_used)
    avg_gpu_util, max_gpu_util = _pair(gpu_utils)
    avg_gpu_mem_util, max_gpu_mem_util = _pair(gpu_mem_utils)
    avg_gpu_mem_used, max_gpu_mem_used = _pair(gpu_mem_used)
    avg_cpu_time, max_cpu_time = _pair(cpu_times)

    participated = sum(1 for o in outcomes_list if o.get("participated"))
    attempted = len(outcomes_list)
    dropped = skipped_clients + sum(1 for o in outcomes_list if not o.get("participated"))

    return {
        "scheduled_clients": int(scheduled_clients),
        "attempted_clients": int(attempted),
        "participating_clients": int(participated),
        "dropped_clients": int(dropped),
        "avg_client_duration": _safe_float(avg_duration),
        "max_client_duration": _safe_float(max_duration),
        "avg_cpu_util": _safe_float(avg_cpu_util),
        "max_cpu_util": _safe_float(max_cpu_util),
        "avg_memory_util": _safe_float(avg_mem_util),
        "max_memory_util": _safe_float(max_mem_util),
        "avg_memory_used_mb": _safe_float(avg_mem_used),
        "max_memory_used_mb": _safe_float(max_mem_used),
        "avg_gpu_util": _safe_float(avg_gpu_util),
        "max_gpu_util": _safe_float(max_gpu_util),
        "avg_gpu_memory_util": _safe_float(avg_gpu_mem_util),
        "max_gpu_memory_util": _safe_float(max_gpu_mem_util),
        "avg_gpu_memory_used_mb": _safe_float(avg_gpu_mem_used),
        "max_gpu_memory_used_mb": _safe_float(max_gpu_mem_used),
        "avg_cpu_time_s": _safe_float(avg_cpu_time),
        "max_cpu_time_s": _safe_float(max_cpu_time),
    }
