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


def _digest32(parts: list) -> int:
    """Hash JSON-serialisable *parts* to a 32-bit unsigned integer, stable across processes and platforms.

    :param parts: Values that identify what is seeded.
    :returns: The first four bytes of the SHA-256 of their compact JSON, big-endian.
    """
    key = json.dumps(parts, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def fit_seed(root_seed: int, identity: RunIdentity) -> int:
    """Derive the stable RNG seed of one fit from a root seed and its run identity.

    The identity's provenance blob is excluded, so the seed depends only on the
    model, dataset, split, fold, randomization and robustness repeat.

    :param root_seed: User-facing seed of the experiment.
    :param identity: Identity of the run about to be fitted.
    :returns: Deterministic 32-bit unsigned integer, usable with every RNG backend.
    """
    return _digest32(
        [
            root_seed,
            identity.model_name,
            identity.dataset_name,
            identity.split_mode,
            identity.fold_index,
            identity.fold_id,
            list(identity.randomization) if identity.randomization else None,
            identity.robustness_repeat,
        ]
    )


def keyed_seed(root_seed: int, key: str) -> int:
    """Derive the RNG seed of one keyed part of a run from a root seed and the part's key.

    A single-drug predictor fits one model per drug, keyed by the drug id, and a randomized
    featurizer draws the values of each entity, keyed by the entity id, and the permutation of each
    group of views, keyed by their names. Seeding each part from its key, rather than letting the parts
    share one RNG stream, makes a part independent of which other parts there are and of the order or
    process they are drawn in.

    :param root_seed: Seed of the whole, such as the seed drawn once per fit from the already seeded
        global RNG.
    :param key: Identifier of the part, such as a drug id.
    :returns: Deterministic 32-bit unsigned integer, usable with every RNG backend.
    """
    return _digest32([root_seed, key])
