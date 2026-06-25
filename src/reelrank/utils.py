"""Small shared helpers: seeding, device resolution, and .env loading."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def load_env(path: str | Path = ".env") -> None:
    """Populate os.environ from a .env file (real env vars take precedence).

    A tiny loader so scripts pick up secrets like TMDB_API_KEY without an extra
    dependency. In production the host sets these as real environment variables.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(preference: str) -> str:
    """Resolve 'auto'|'cpu'|'cuda' to a concrete device string."""
    if preference == "cpu":
        return "cpu"
    import torch

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"
