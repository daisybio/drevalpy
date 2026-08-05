"""Structural typing for shared literature predictor helpers."""

from __future__ import annotations

from collections.abc import Sequence, Sized
from typing import Protocol

import numpy as np


class MultiOmicsFeatureAttributes(Protocol):
    """Omic feature name lists stored on MOLIR and SuperFELTR algorithms."""

    gene_expression_features: Sequence[str] | np.ndarray | None
    mutations_features: Sequence[str] | np.ndarray | None
    copy_number_variation_features: Sequence[str] | np.ndarray | None


def feature_count(features: Sized | None) -> int | None:
    """Return ``len(features)`` when *features* is sized.

    :param features: Sized feature collection or ``None``.

    :returns: Feature count, or ``None`` when *features* is ``None``.
    """
    if features is None:
        return None
    return len(features)
