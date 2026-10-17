"""Random seed setup for experiment runs."""

from __future__ import annotations

import hashlib
import json
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drevalpy.types.results._identity import RunIdentity


def seed_everything(seed: int = 42) -> None:
    """Seed python ``random``, numpy, and torch (CPU + CUDA).

    :param seed: Random seed applied to all supported RNG backends.

    ``PYTHONHASHSEED`` is deliberately not touched: it is read once at interpreter
    start-up, so setting it afterwards would not change string hashing.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_seed(root_seed: int, identity: RunIdentity) -> int:
    """Derive the stable RNG seed of one fit from a root seed and its execution identity.

    The identity's provenance blob is excluded, so the seed depends only on the
    model, dataset, split, fold, randomization and robustness trial.

    :param root_seed: User-facing seed of the experiment.
    :param identity: Identity of the execution about to be fitted.
    :returns: Deterministic 32-bit unsigned integer, usable with every RNG backend.
    """
    key = json.dumps(
        [
            root_seed,
            identity.model_name,
            identity.dataset_name,
            identity.split_mode,
            identity.fold_index,
            identity.fold_id,
            list(identity.randomization) if identity.randomization else None,
            identity.robustness_trial,
        ],
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
