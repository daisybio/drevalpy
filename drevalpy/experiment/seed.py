"""Random seed setup for experiment runs."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Seed python ``random``, numpy, torch (CPU + CUDA), and ``PYTHONHASHSEED``.

    Call once at the top of a run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
