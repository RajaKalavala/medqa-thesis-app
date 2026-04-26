"""Global determinism helpers."""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # FAISS interactions are non-deterministic
    except ImportError:  # pragma: no cover
        pass
