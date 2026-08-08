"""SuperFELTR multi-omics feature-selection featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.preprocessing import VarianceFeatureSelector
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.loading.multiomics import get_multiomics_feature_dataset

_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


@register_cell_line_featurizer(
    "superfeltrOmics",
    description="SuperFELTR variance-selected multi-omics inputs.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SuperFELTROmicsFeaturizer(CellLineFeaturizer):
    """Select high-variance features independently in each SuperFELTR view."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("copy_number_variation_gistic", FeatureFormat.NUMERIC_MATRIX),
    )
    input_views: ClassVar[tuple[str, ...]] = _VIEWS

    def __init__(self, *, n_features_per_view: int = 1000) -> None:
        """Store per-view variance-selection feature count and initialize selectors.

        :param n_features_per_view: Number of features to keep per omics view.
        """
        self._n_features = int(n_features_per_view)
        self._selectors = {view: VarianceFeatureSelector(view, self._n_features) for view in _VIEWS}
        self._feature_names: dict[str, tuple[str, ...]] = {}

    @classmethod
    def load_features(cls, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load full omics and apply SuperFELTR's arcsinh expression transform.

        :param dataset_name: Dataset folder name.
        :param kwargs: Unused loader keyword arguments.
        :returns: Multi-omics feature dataset with arcsinh gene expression.
        """
        _ = cls, kwargs
        features = get_multiomics_feature_dataset(dataset_name, gene_lists=None, omics=list(_VIEWS))
        features.apply(np.arcsinh, view="gene_expression")
        return features

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> SuperFELTROmicsFeaturizer:
        """Fit variance selectors independently in each omics view.

        :param features: Cell-line multi-omics feature dataset.
        :param entity_ids: Optional explicit fit ids; otherwise derived from *context*.
        :param context: Optional fit context supplying unique training ids.
        :returns: Fitted featurizer instance.
        """
        ids = np.unique(
            entity_ids if entity_ids is not None else context.unique_train_ids if context else features.identifiers
        )
        for view, selector in self._selectors.items():
            selector.fit_on_ids(features, ids)
            self._feature_names[view] = tuple(selector.selected_meta_info)
        return self

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        """Return variance-selected gene-expression features only.

        :param features: Cell-line multi-omics feature dataset.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Float matrix of selected gene-expression features.
        """
        selector = self._selectors["gene_expression"]
        return stack_view_matrix(features, "gene_expression", entity_ids)[:, selector.mask].astype(np.float32)

    def transform_blocks(self, features: FeatureDataset, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return per-omics numeric blocks with variance-selected columns.

        :param features: Cell-line multi-omics feature dataset.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Mapping of omics view name to numeric blocks.
        """
        return {
            view: numeric_feature_block(
                stack_view_matrix(features, view, entity_ids)[:, selector.mask].astype(np.float32),
                feature_names=self._feature_names.get(view),
            )
            for view, selector in self._selectors.items()
        }

    @property
    def output_dim(self) -> int:
        """Return total selected features across all views.

        :returns: Sum of selected features in every view.
        """
        return int(sum(selector.mask.sum() for selector in self._selectors.values()))

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable per-view variance-selection feature count.

        :returns: Ray Tune-style hyperparameter space mapping.
        """
        return {"n_features_per_view": {"type": "int", "low": 1, "high": 1000, "default": 1000}}

    def get_state(self) -> dict[str, object]:
        """Serialize selectors and feature-name metadata.

        :returns: Fitted state mapping.
        """
        return {
            "selectors": self._selectors,
            "feature_names": self._feature_names,
            "n_features_per_view": self._n_features,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore selectors and feature names from ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        selectors = state.get("selectors")
        if isinstance(selectors, dict) and all(
            isinstance(value, VarianceFeatureSelector) for value in selectors.values()
        ):
            self._selectors = {str(key): value for key, value in selectors.items()}
        names = state.get("feature_names")
        if isinstance(names, dict):
            self._feature_names = {str(key): tuple(value) for key, value in names.items()}
