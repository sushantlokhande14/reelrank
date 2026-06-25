"""Small shared helpers: seeding and device resolution."""

from __future__ import annotations

import random

import numpy as np


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
