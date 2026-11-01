"""Sample pass-through featurizer: one dense omics view, unchanged."""

from __future__ import annotations

from typing import Self

import numpy as np

from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer, all_nan_rows_mask
from drevalpy.components.featurizers.sample.gene_lists import view_genes
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side


@register("view", side=Side.SAMPLE, description="Pass through one dense omics view without preprocessing.")
class ViewSampleFeaturizer(Featurizer):
    """Featurize one omics view as a dense matrix without transformation."""

    def __init__(self, *, view: str, gene_list: str | None = None) -> None:
        """Bind the omics view to pass through.

        :param view: Name of the omics view to emit unchanged.
        :param gene_list: Name of a shipped gene list to keep, in list order; the view must hold every gene of
            it. ``None`` keeps the whole view.
        :raises ValueError: If *view* is empty or blank.
        """
        if not view or not view.strip():
            msg = "raw featurizer requires an explicit view"
            raise ValueError(msg)
        self._view = view
        self._gene_list = gene_list
        self._names: tuple[str, ...] | None = None

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Return which samples have a row of the bound view that is not all NaN.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs to check.
        :returns: Boolean array with one entry per ID.
        """
        return all_nan_rows_mask(source, self._view, ids)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Bind the pass-through to the genes of the gene list, or to the whole view without one.

        :param source: Feature source of the sample side.
        :param train_ids: Unused; nothing is learned from the rows.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        _ = train_ids, train_pair_ids, es_pair_ids
        self._names = view_genes(source, self._view, self._gene_list)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return the view's rows for *ids*, with columns in fitted order.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One block named after the view.
        """
        return dense_result(ids, self._view, view_matrix(source, self._view, ids, self._names), self._names)
