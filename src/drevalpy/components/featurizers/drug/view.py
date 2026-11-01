"""Single-view drug featurizer."""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np

from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side


@register("view", side=Side.DRUG, description="Pass through one dense drug view from a FeatureSource.")
class ViewDrugFeaturizer(Featurizer):
    """Featurize one drug view without additional transformation; the block is named after the view."""

    input_views: ClassVar[tuple[str, ...]] = ("morgan_fingerprint",)

    def __init__(self, *, view: str | None = None) -> None:
        """Bind the view this instance reads.

        :param view: View name; ``None`` reads the declared input view.
        """
        self._view = view or self.input_views[0]
        self._view_names: tuple[str, ...] | None = None

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Record the view's feature names.

        :param source: Drug feature source.
        :param train_ids: Usable training drug IDs; unused.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer.
        """
        self._view_names = source.feature_names(self._view)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return the view's rows, columns in fitted order.

        :param source: Drug feature source.
        :param ids: Drug IDs, in output order.
        :returns: Result with one block named after the view.
        """
        matrix = view_matrix(source, self._view, ids, self._view_names)
        return dense_result(ids, self._view, matrix, self._view_names)
