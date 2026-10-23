"""MOLIR multi-omics preprocessing featurizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._feature_names import align_to_fitted
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.components.featurizers.sample.gene_lists import checked_view_gene_lists, select_genes
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource

_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


@register(
    "molirOmics",
    description="MOLIR multi-omics input preparation.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class MOLIROmicsFeaturizer(DenseViewSampleFeaturizer):
    """Arcsinh-scale and select variable gene-expression features for MOLIR."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("copy_number_variation_gistic", FeatureFormat.NUMERIC_MATRIX),
    )
    input_views: ClassVar[tuple[str, ...]] = _VIEWS
    fit_on_unique_ids: ClassVar[bool] = True

    def __init__(
        self, *, n_gene_expression_features: int = 1000, gene_lists: Mapping[str, str | None] | None = None
    ) -> None:
        """Store the variance-selection feature count and initialize scalers.

        :param n_gene_expression_features: Number of gene-expression features to keep.
        :param gene_lists: Shipped gene-list name per view, keyed by view name; each named view must hold
            every gene of its list. A view left out or mapped to ``None`` keeps all its genes. A key that is not
        one of the three input views is rejected with ``InvalidInputError``.
        """
        from sklearn.preprocessing import StandardScaler

        super().__init__()
        self._gene_lists = checked_view_gene_lists(gene_lists, _VIEWS)
        self._n_features = int(n_gene_expression_features)
        self._scaler = StandardScaler()
        self._mask: np.ndarray = np.array([], dtype=bool)
        self._selected_feature_names: tuple[str, ...] = ()
        self._feature_names: dict[str, tuple[str, ...] | None] = {}

    def _view_gene_names(self, source: FeatureSource, view: str) -> tuple[str, ...] | None:
        """Return the feature names *view* is read in: its gene list, or every feature of the view.

        :param source: Feature source providing view matrices.
        :param view: One of the input views.
        :returns: Ordered feature names, or ``None`` when the source has none.
        """
        name = self._gene_lists.get(view)
        return source.get_feature_names(view) if name is None else select_genes(source, view, name)

    def _names_to_fit(self, source: FeatureSource) -> tuple[str, ...] | None:
        """Bind the gene-expression transform to the genes of its list, if it has one.

        :param source: Feature source providing view matrices.
        :returns: Ordered gene-expression feature names, or ``None`` when the source has none.
        """
        return self._view_gene_names(source, self._view)

    def _on_precomputed_fit(self, source: FeatureSource) -> None:
        """Accept every stored column and record the per-view feature names.

        :param source: Feature source carrying the stored variant.
        """
        self._mask = np.ones(self._output_dim, dtype=bool)
        self._feature_names = {view: source.get_feature_names(view) for view in _VIEWS}

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit the scaler and pick the highest-variance gene-expression features.

        :param source: Feature source providing view matrices.
        :param entity_ids: Deduplicated sample identifiers to fit on.
        :returns: Number of selected gene-expression features.
        """
        matrix = np.arcsinh(self._raw_matrix(source, entity_ids))
        self._scaler.fit(matrix)
        # Standardization makes every nonconstant variance approximately one;
        # rank expression variance before that step, using training rows only.
        variances = np.var(matrix, axis=0)
        self._mask = np.zeros(len(variances), dtype=bool)
        self._mask[np.argsort(variances)[::-1][: min(self._n_features, len(variances))]] = True

        ge_names = self._view_names
        self._selected_feature_names = () if ge_names is None else tuple(np.array(ge_names)[self._mask])
        self._feature_names = {
            "gene_expression": self._selected_feature_names,
            **{view: self._view_gene_names(source, view) for view in _VIEWS[1:]},
        }
        return int(self._mask.sum())

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Arcsinh-scale *matrix* and keep only the selected features.

        :param source: Feature source the matrix came from.
        :param matrix: Raw gene-expression matrix.
        :returns: Selected gene-expression features.
        """
        _ = source
        return self._scaler.transform(np.arcsinh(matrix))[:, self._mask]

    def _block_feature_names(self, source: FeatureSource) -> tuple[str, ...] | None:
        """Return the selected gene names recorded at fit time.

        :param source: Feature source (unused; names are recorded during fit).
        :returns: Selected gene-expression feature names, or ``None``.
        """
        _ = source
        return self._feature_names.get("gene_expression")

    def _transform_features(self, source: FeatureSource, entity_ids: np.ndarray) -> FeaturizerResult:
        """Return per-omics numeric blocks for MOLIR.

        The gene-expression block goes through the shared dense path; the two other
        omics views are passed through untouched.

        :param source: Feature source providing view matrices.
        :param entity_ids: Sample identifiers to transform.
        :returns: Mapping of omics view name to numeric blocks.
        """
        result = super()._transform_features(source, entity_ids)
        blocks = {
            **result.blocks,
            **{
                view: numeric_feature_block(
                    align_to_fitted(
                        source, view, source.get_view_matrix(view, entity_ids), self._feature_names.get(view)
                    ).astype(np.float32),
                    feature_names=self._feature_names.get(view),
                )
                for view in _VIEWS[1:]
            },
        }
        return FeaturizerResult(entity_ids, blocks, result.projection)

    @property
    def output_dim(self) -> int:
        """Return the number of selected gene-expression features.

        :returns: Selected gene-expression feature count.
        """
        return int(self._mask.sum()) if self._mask.size > 0 else 0

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable variance-selection feature count.

        :returns: Hyperparameter search space mapping.
        """
        return {"n_gene_expression_features": {"type": "int", "low": 1, "high": 1000, "default": 1000}}

    def get_state(self) -> dict[str, object]:
        """Serialize scaler, mask, and feature-name metadata.

        :returns: Fitted state mapping.
        """
        return {
            **self._dense_state(),
            "scaler": self._scaler,
            "mask": self._mask,
            "selected_feature_names": self._selected_feature_names,
            "feature_names": self._feature_names,
            "n_gene_expression_features": self._n_features,
            "gene_lists": self._gene_lists,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore scaler, mask, and feature names from ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        from sklearn.preprocessing import StandardScaler

        scaler = state.get("scaler")
        if isinstance(scaler, StandardScaler):
            self._scaler = scaler
        mask = state.get("mask")
        if isinstance(mask, np.ndarray):
            self._mask = mask
        selected = state.get("selected_feature_names")
        if isinstance(selected, tuple):
            self._selected_feature_names = selected
        names = state.get("feature_names")
        if isinstance(names, dict):
            self._feature_names = {str(key): value for key, value in names.items()}
        gene_lists = state.get("gene_lists")
        if isinstance(gene_lists, dict):
            self._gene_lists = dict(gene_lists)
        self._restore_dense_state(state)
