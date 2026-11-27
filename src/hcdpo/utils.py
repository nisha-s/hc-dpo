"""Seeding, device and precision selection, config loading, small IO helpers."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import platform
import random
from pathlib import Path
from typing import Any, ContextManager, Iterable

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Pick the best available device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_bf16(device: torch.device) -> bool:
    """True only on CUDA GPUs with native bf16 (A100, L4). Everything else runs fp32."""
    if device.type != "cuda":
        return False
    # By default torch also reports True for GPUs that only emulate bf16 (e.g. T4), which is slow.
    return torch.cuda.is_bf16_supported(including_emulation=False)


def autocast(device: torch.device) -> ContextManager[Any]:
    """bf16 autocast where supported, otherwise a no-op context."""
    if use_bf16(device):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def parse_override_value(raw: str) -> Any:
    """Parse a `--set key=value` value with YAML rules, plus floats like `5e-5`."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # YAML 1.1 reads "5e-5" as a string because it has no decimal point.
        try:
            return float(value)
        except ValueError:
            return value
    return value


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set `config["a"]["b"] = value` for key "a.b". The key must already exist, which catches typos."""
    *parents, last = dotted_key.split(".")
    node = config
    for key in parents:
        node = node[key]
    if last not in node:
        raise KeyError(f"Unknown config key '{dotted_key}'")
    node[last] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply `key=value` overrides (dotted keys for nested values)."""
    with open(path) as f:
        config = yaml.safe_load(f)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override '{item}' is not of the form key=value")
        set_by_dotted_key(config, key.strip(), parse_override_value(raw.strip()))
    return config


def print_environment(device: torch.device) -> dict[str, str]:
    """Print and return library versions, device and precision, for the start of a run."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "none",
        "precision": "bf16 autocast" if use_bf16(device) else "fp32",
    }
    for package in ("transformers", "peft", "datasets"):
        info[package] = importlib.metadata.version(package)
    print("Environment: " + ", ".join(f"{k}={v}" for k, v in info.items()))
    return info


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dicts to a JSONL file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    """Write an object to a JSON file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)
