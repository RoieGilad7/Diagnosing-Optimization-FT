from __future__ import annotations

import os
import platform
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Best-effort; MPS/CUDA are not guaranteed bitwise-deterministic for all ops.
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def software_versions() -> dict[str, Any]:
    import transformers
    import lightning

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "lightning": lightning.__version__,
        "transformers": transformers.__version__,
    }
