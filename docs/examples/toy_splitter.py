"""Splitter example: a leave-sample-out variant with a custom fold count.

A splitter is a plain function, not a class. ``validation=`` picks the leakage
constraint the registry enforces after every call, so a splitter that leaks a
sample into both train and test raises ``SplitValidationError`` rather than
silently producing an optimistic score.

``drevalpy.split`` and ``drevalpy.run`` pass the curve-quality rule to the
splitter as the required ``quality`` keyword. :func:`~drevalpy.plugin.observed_mask`
turns it into the pairs the splitter may use. This splitter tightens the rule
with an extra ``min_r2`` threshold of its own.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.model_selection import KFold

from drevalpy.plugin import CurveQuality, MuDataLike, SplitMask, SplitMasks, observed_mask, register_splitter


@register_splitter(
    "TOY_LSO",
    "Leave-Sample-Out with a fixed 80/20 train/validation carve-out",
    validation="LSO",
)
def toy_leave_sample_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality,
) -> list[SplitMasks]:
    """Split so that every sample is tested exactly once.

    :param mudataset: Dataset whose response matrix is split.
    :param n_splits: Number of folds.
    :param validation_ratio: Ignored; this splitter always holds out a fifth of the
        training samples, which is what makes it worth registering
        separately from the built-in ``LSO``.
    :param random_state: Seed for the fold assignment.
    :param quality: Curve-quality rule of the run; the extra ``min_r2`` threshold is added to it.
    :returns: One :class:`~drevalpy.plugin.SplitMasks` per fold, each carrying
        train/test/validation masks shaped like the response matrix.
    """
    _ = validation_ratio
    # ``min_r2`` is an extra this splitter opts into. Asking for it reads the
    # ``R2`` layer, even when the run used ``CurveQuality.off()``.
    observed = observed_mask(mudataset, replace(quality, min_r2=0.5))
    folds: list[SplitMasks] = []
    rng = np.random.default_rng(random_state)

    for train_rows, test_rows in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
        np.arange(observed.shape[0])
    ):
        shuffled = rng.permutation(train_rows)
        n_validation = max(1, len(shuffled) // 5)
        folds.append(
            SplitMasks(
                train=SplitMask(_rows_mask(observed, shuffled[n_validation:])),
                test=SplitMask(_rows_mask(observed, test_rows)),
                val=SplitMask(_rows_mask(observed, shuffled[:n_validation])),
            )
        )
    return folds


def _rows_mask(observed: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Keep the measured pairs of *rows* and blank everything else."""
    mask = np.zeros_like(observed)
    mask[rows, :] = observed[rows, :]
    return mask
