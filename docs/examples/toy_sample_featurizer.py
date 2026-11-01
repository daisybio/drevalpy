"""Sample featurizer example: standardize one sample view.

Shows the two methods every featurizer implements, ``fit`` and ``transform``, and the
class-body declarations: ``sides`` (where the featurizer may be used) and ``input_views``
(the views it reads). Model training calls both methods only with usable entity IDs; it
drops entities whose row is all NaN beforehand and pads them back in afterwards.
"""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np

from drevalpy.plugin import (
    FeatureSource,
    Featurizer,
    FeaturizerResult,
    Side,
    dense_result,
    register_featurizer,
    view_matrix,
)

VIEW = "gene_expression"


@register_featurizer(
    "toySample",
    description="Gene expression standardized with statistics learned on the training samples.",
)
class ToySampleFeaturizer(Featurizer):
    """Standardize the ``gene_expression`` view column by column."""

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})
    input_views: ClassVar[tuple[str, ...]] = (VIEW,)

    def __init__(self) -> None:
        """Create an unfitted featurizer."""
        self._names: tuple[str, ...] | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Learn one mean and one standard deviation per column from the training samples.

        The feature names are recorded too, so ``transform`` reads the columns in the
        same order even from a source that stores them differently.
        """
        _ = train_pair_ids, es_pair_ids
        names = source.feature_names(VIEW)
        self._names = None if names is None else tuple(names)
        matrix = view_matrix(source, VIEW, train_ids, self._names)
        self._mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self._scale = np.where(scale > 0.0, scale, 1.0)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return the standardized rows of *ids* as one numeric block.

        The block is named after the view, which is the name the default
        ``output_blocks`` declares and a block predictor asks for.
        """
        if self._mean is None or self._scale is None:
            msg = "ToySampleFeaturizer must be fitted before transforming"
            raise RuntimeError(msg)
        matrix = view_matrix(source, VIEW, ids, self._names)
        return dense_result(ids, VIEW, (matrix - self._mean) / self._scale, self._names)
