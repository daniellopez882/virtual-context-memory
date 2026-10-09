"""Locations of downloaded data and models.

The data directory is read from ``VCMEM_DATA_DIR`` so that no machine-specific path
ever lands in the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_DATA_DIR = "VCMEM_DATA_DIR"


def data_dir() -> Path:
    raw = os.environ.get(ENV_DATA_DIR)
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "virtual-context-memory"


def locomo_path() -> Path:
    return data_dir() / "locomo" / "locomo10.json"


def minilm_dir() -> Path:
    return data_dir() / "models" / "all-MiniLM-L6-v2"
