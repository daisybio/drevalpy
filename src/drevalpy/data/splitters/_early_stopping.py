"""The early-stopping set: whole leakage groups cut from a fold's validation set.

A model that stops early watches a part of the validation set while it trains, and the
rest of the validation set scores tuning trials and joins the final training set. The
stopping part is cut by the groups of the split mode
(:func:`~drevalpy.registry.splitter._groups.pair_groups`), so under LDO the stopping drugs,
and under LTO the stopping tissues, are in neither the rest of validation nor the final
training set. A mode whose validation is drawn by other groups (its
:attr:`~drevalpy.registry.splitter._groups.LeakageGroups.cut`; for LStO (source sample, drug))
has its stopping part cut by those groups too.

The split is made at execution, for models that stop early only, rather than by the
splitters: a fold whose validation set has a single group is still a valid fold for
every other model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.errors import InvalidInputError
from drevalpy.registry.splitter._groups import MODES, in_groups, pair_groups
from drevalpy.registry.splitter._validation import validate_parts
from drevalpy.types import PairMask

if TYPE_CHECKING:
    from drevalpy.registry.splitter import Validation
    from drevalpy.types import Dataset, Fold


def early_stopping_split(
    dataset: Dataset, fold: Fold, mode: Validation, *, fraction: float = 0.25
) -> tuple[PairMask, PairMask]:
    """Cut a seeded *fraction* of the validation groups out of *fold* as its early-stopping set.

    ``max(1, int(n_groups * fraction))`` groups are drawn with a generator seeded from the
    validation mask's ``pair_seed``, so each robustness repeat draws other groups, or from
    the fold's ``random_state`` (0 for a fold without one). The four parts (train, stopping,
    the rest of validation, test) are then validated along *mode*, which raises
    ``SplitValidationError`` on a leak.

    :param dataset: Dataset the fold indexes.
    :param fold: Fold whose validation set is cut.
    :param mode: Leakage constraint of the fold's split mode.
    :param fraction: Share of the validation groups held out for early stopping.
    :returns: ``(stopping, rest)``: the early-stopping set and the remaining validation set,
        both with the validation mask's ``pair_seed``.
    :raises InvalidInputError: If the validation set has fewer than two groups of the mode.
    """
    cut = MODES[mode].cut
    groups = pair_groups(dataset, cut)
    validation = fold.val
    fold_index = fold.metadata.get("fold_index", 0)
    present = np.unique(groups[validation.mask])
    if len(present) < 2:
        ratio = fold.metadata.get("validation_ratio", "unknown")
        raise InvalidInputError(
            f"Early stopping under {mode} needs at least two {MODES[cut].unit} in the validation set, to hold "
            f"whole ones out, but the validation set of fold {fold_index} has {len(present)} of the dataset's "
            f"{len(np.unique(groups))}. Raise validation_ratio (now {ratio}): "
            "drevalpy.run(validation_ratio=...), drevalpy run --validation-ratio, or "
            "drevalpy.split(..., validation_ratio=...)."
        )
    seed = validation.pair_seed if validation.pair_seed is not None else fold.metadata.get("random_state", 0)
    chosen = np.random.default_rng(seed).permutation(present)[: max(1, int(len(present) * fraction))]
    in_stopping = validation.mask & in_groups(groups, chosen)
    stopping = PairMask(in_stopping, validation.pair_seed)
    rest = PairMask(validation.mask & ~in_stopping, validation.pair_seed)
    validate_parts(
        {"train": fold.train, "stopping": stopping, "validation": rest, "test": fold.test},
        mode,
        dataset,
        context=f"early stopping of fold {fold_index}",
    )
    return stopping, rest
