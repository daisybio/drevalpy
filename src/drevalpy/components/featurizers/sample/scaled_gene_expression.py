"""Scaled gene-expression featurizer for samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer, all_nan_rows_mask
from drevalpy.components.featurizers.sample.gene_lists import view_genes
from drevalpy.errors import InvalidInputError
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler

TRANSFORMS = ("arcsinh", "rank")
_BLOCK = "gene_expression"


def rank_transform(matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Place every value at its position among the sorted reference values of its gene.

    :param matrix: ``(n_samples, n_genes)`` values to transform.
    :param reference: ``(n_reference, n_genes)`` values sorted along axis 0, from the training samples.
    :returns: Positions scaled to ``[0, 1]``: ``searchsorted(side="left") / (n_reference - 1)``, clipped.
    """
    percentile = np.empty(matrix.shape, dtype=np.float64)
    for gene in range(matrix.shape[1]):
        percentile[:, gene] = np.searchsorted(reference[:, gene], matrix[:, gene], side="left")
    return np.clip(percentile / max(1, reference.shape[0] - 1), 0.0, 1.0)


@register(
    "scaledGeneExpression",
    side=Side.SAMPLE,
    description="Gene expression with an arcsinh or rank transform and scaling.",
)
class ScaledGeneExpressionFeaturizer(Featurizer):
    """Match sklearn baseline gene-expression preprocessing."""

    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

    def __init__(
        self, *, view: str = "gene_expression", gene_list: str | None = None, transform: str = "arcsinh"
    ) -> None:
        """Bind the gene-expression view to scale.

        :param view: Name of the view to transform and standardize.
        :param gene_list: Name of a shipped gene list to keep, in list order; the view must hold every
            gene of it. ``None`` keeps the whole view.
        :param transform: ``arcsinh`` applies the pointwise inverse hyperbolic sine. ``rank`` replaces each
            value by its position among the sorted values of the training samples (see
            :func:`rank_transform`), so a sample is always scored against the fixed training reference.
        :raises InvalidInputError: If *transform* is not ``arcsinh`` or ``rank``.
        """
        if transform not in TRANSFORMS:
            msg = f"Unknown transform {transform!r}; expected one of {', '.join(TRANSFORMS)}."
            raise InvalidInputError(msg)
        self._view = view
        self._gene_list = gene_list
        self._transform = transform
        self._names: tuple[str, ...] | None = None
        self._rank_reference: np.ndarray | None = None
        self._scaler: StandardScaler | None = None

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Publish under the canonical gene-expression block name, whatever view is read.

        :param config: Unused.
        :returns: The ``gene_expression`` block.
        """
        _ = config
        return (BlockSpec(_BLOCK, FeatureFormat.NUMERIC_MATRIX),)

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
        """Fit the rank reference (for ``rank``) and the scaler on the sorted unique training samples.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        from sklearn.preprocessing import StandardScaler

        _ = train_pair_ids, es_pair_ids
        self._names = view_genes(source, self._view, self._gene_list)
        raw = view_matrix(source, self._view, np.unique(train_ids), self._names)
        if self._transform == "rank":
            self._rank_reference = np.sort(raw.astype(np.float64), axis=0)
        self._scaler = StandardScaler().fit(self._transform_values(raw))
        return self

    def _transform_values(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the configured transform before scaling.

        :param matrix: Raw view matrix.
        :returns: Transformed matrix; float64 for ``rank``.
        """
        if self._rank_reference is None:
            return np.arcsinh(matrix)
        return rank_transform(matrix.astype(np.float64), self._rank_reference)

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Transform and scale the view's rows for *ids*.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One ``gene_expression`` block.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._scaler is None:
            msg = "ScaledGeneExpressionFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        if len(ids) == 0:
            matrix = np.empty((0, int(self._scaler.n_features_in_)))
        else:
            matrix = self._scaler.transform(self._transform_values(view_matrix(source, self._view, ids, self._names)))
        return dense_result(ids, _BLOCK, matrix, self._names)
