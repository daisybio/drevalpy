"""Scaled gene-expression featurizer for samples."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.components.featurizers.sample.gene_lists import select_genes
from drevalpy.errors import InvalidInputError
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.feature_source import FeatureSource

TRANSFORMS = ("arcsinh", "rank")


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
    description="Gene expression with an arcsinh or rank transform and scaling.",
    output_format=FeatureFormat.NUMERIC_MATRIX,
)
class ScaledGeneExpressionFeaturizer(DenseViewSampleFeaturizer):
    """Match sklearn baseline gene-expression preprocessing."""

    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    requires_fit: ClassVar[bool] = True
    fit_on_unique_ids: ClassVar[bool] = True

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
        from sklearn.preprocessing import StandardScaler

        super().__init__(view=view)
        if transform not in TRANSFORMS:
            msg = f"Unknown transform {transform!r}; expected one of {', '.join(TRANSFORMS)}."
            raise InvalidInputError(msg)
        self._gene_list = gene_list
        self._transform = transform
        self._rank_reference: np.ndarray | None = None
        self._scaler = StandardScaler()

    def _names_to_fit(self, source: FeatureSource) -> tuple[str, ...] | None:
        """Bind the transform to the genes of the gene list, or to the whole view without one.

        :param source: Feature source providing view matrices.
        :returns: Gene names in gene-list order, or every feature name of the view.
        """
        if self._gene_list is None:
            return super()._names_to_fit(source)
        return select_genes(source, self._view, self._gene_list)

    def _fetch_hyperparameters(self) -> dict[str, Any] | None:
        """Match a stored variant only when it was built for the same gene list.

        :returns: The gene-list and non-default transform settings, or ``None`` when both are default.
        """
        settings: dict[str, Any] = {}
        if self._gene_list is not None:
            settings["gene_list"] = self._gene_list
        if self._transform != "arcsinh":
            settings["transform"] = self._transform
        return settings or None

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit the rank reference (for ``rank``) and the scaler on the transformed training rows.

        :param source: Feature source providing view matrices.
        :param entity_ids: Deduplicated sample identifiers to fit on.
        :returns: Output feature dimension.
        """
        raw = self._raw_matrix(source, entity_ids)
        if self._transform == "rank":
            self._rank_reference = np.sort(raw.astype(np.float64), axis=0)
        matrix = self._transform_values(raw)
        self._scaler.fit(matrix)
        return int(matrix.shape[1])

    def _transform_values(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the configured transform before scaling.

        :param matrix: Raw view matrix.
        :returns: Transformed matrix; float64 for ``rank``.
        :raises RuntimeError: If ``rank`` is configured but no reference was fitted.
        """
        if self._transform == "arcsinh":
            return np.arcsinh(matrix)
        if self._rank_reference is None:
            msg = "ScaledGeneExpressionFeaturizer has no fitted rank reference"
            raise RuntimeError(msg)
        return rank_transform(matrix.astype(np.float64), self._rank_reference)

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Transform and scale *matrix*.

        :param source: Feature source the matrix came from.
        :param matrix: Raw view matrix for the requested entity IDs.
        :returns: Scaled feature matrix.
        """
        _ = source
        return self._scaler.transform(self._transform_values(matrix))

    def _block_name(self) -> str:
        """Publish under the canonical gene-expression block name.

        :returns: Block name.
        """
        return "gene_expression"

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state."""
        if not self._is_fitted:
            return {}
        return {
            "gene_expression_scaler": self._scaler,
            "gene_list": self._gene_list,
            "transform": self._transform,
            "rank_reference": self._rank_reference,
            **self._dense_state(),
            "fitted": True,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping previously returned by ``get_state``.
        """
        from sklearn.preprocessing import StandardScaler

        scaler = state.get("gene_expression_scaler")
        if isinstance(scaler, StandardScaler):
            self._scaler = scaler
        gene_list = state.get("gene_list")
        if gene_list is None or isinstance(gene_list, str):
            self._gene_list = gene_list
        transform = state.get("transform")
        if isinstance(transform, str) and transform in TRANSFORMS:
            self._transform = transform
        reference = state.get("rank_reference")
        if isinstance(reference, np.ndarray) or reference is None:
            self._rank_reference = reference
        self._restore_dense_state(state)
