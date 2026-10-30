"""PharmaFormer gene-expression preprocessing featurizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.components.featurizers.sample.gene_lists import view_genes
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side

if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

_VIEW = "gene_expression"


@register("pharmaFormerGeneExpression", description="Reduced landmark genes scaled for PharmaFormer.")
class PharmaFormerGeneExpressionFeaturizer(Featurizer):
    """Apply the PharmaFormer StandardScaler then MinMaxScaler sequence, fitted once per training sample."""

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})
    input_views: ClassVar[tuple[str, ...]] = (_VIEW,)

    def __init__(self, *, gene_list: str | None = None) -> None:
        """Store the gene list.

        :param gene_list: Name of a shipped gene list to keep, in list order; the view must hold every
            gene of it. ``None`` keeps the whole view.
        """
        self._gene_list = gene_list
        self._names: tuple[str, ...] | None = None
        self._scalers: tuple[StandardScaler, MinMaxScaler] | None = None

    def stored_params(self) -> Mapping[str, Any] | None:
        """Match a stored variant only when it was built for the same gene list.

        :returns: The gene-list setting, empty for the whole view.
        """
        return {} if self._gene_list is None else {"gene_list": self._gene_list}

    def from_stored(self, source: FeatureSource, ids: np.ndarray, matrix: np.ndarray) -> FeaturizerResult:
        """Wrap a stored scaled matrix as the ``gene_expression`` block.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, one per row of *matrix*.
        :param matrix: Stored scaled expression.
        :returns: One ``gene_expression`` block.
        """
        return dense_result(ids, _VIEW, matrix, view_genes(source, _VIEW, self._gene_list))

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Fit both scalers on the sorted unique training samples, so a sample with many pairs counts once.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        _ = train_pair_ids, es_pair_ids
        self._names = view_genes(source, _VIEW, self._gene_list)
        matrix = view_matrix(source, _VIEW, np.unique(train_ids), self._names)
        scaler = StandardScaler()
        minmax = MinMaxScaler().fit(scaler.fit_transform(matrix))
        self._scalers = (scaler, minmax)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Apply the fitted scalers to the rows of *ids*.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One ``gene_expression`` block.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._scalers is None:
            msg = "PharmaFormerGeneExpressionFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        scaler, minmax = self._scalers
        if len(ids) == 0:
            matrix = np.empty((0, int(scaler.n_features_in_)))
        else:
            matrix = minmax.transform(scaler.transform(view_matrix(source, _VIEW, ids, self._names)))
        return dense_result(ids, _VIEW, matrix, self._names)
