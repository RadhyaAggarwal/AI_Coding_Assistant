"""Loads config.yaml. The only place in the app allowed to know the file format.

Every other module receives a plain dict from load_config() instead of
reading config.yaml itself.
"""
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
