"""MOLIR multi-omics preprocessing featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.preprocessing import StandardScaler

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._matrix import feature_names_for_view, stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.preprocessing import VarianceFeatureSelector
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.loading.multiomics import get_multiomics_feature_dataset

_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


@register_cell_line_featurizer(
    "molirOmics",
    description="MOLIR multi-omics input preparation.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class MOLIROmicsFeaturizer(CellLineFeaturizer):
    """Arcsinh-scale and select variable gene-expression features for MOLIR."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("copy_number_variation_gistic", FeatureFormat.NUMERIC_MATRIX),
    )
    input_views: ClassVar[tuple[str, ...]] = _VIEWS

    def __init__(self, *, n_gene_expression_features: int = 1000) -> None:
        """Store the variance-selection feature count and initialize scalers.

        :param n_gene_expression_features: Number of gene-expression features to keep.
        """
        self._n_features = int(n_gene_expression_features)
        self._scaler = StandardScaler()
        self._selector = VarianceFeatureSelector("gene_expression", self._n_features)
        self._feature_names: dict[str, tuple[str, ...] | None] = {}

    @classmethod
    def load_features(cls, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load the intersection-gene multi-omics tables required by MOLIR.

        :param dataset_name: Dataset folder name.
        :param kwargs: Unused loader keyword arguments.
        :returns: Multi-omics feature dataset with intersection gene lists.
        """
        _ = cls, kwargs
        return get_multiomics_feature_dataset(
            dataset_name,
            gene_lists={
                "gene_expression": "gene_expression_intersection",
                "mutations": "mutations_intersection",
                "copy_number_variation_gistic": "copy_number_variation_gistic_intersection",
            },
            omics=list(_VIEWS),
        )

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> MOLIROmicsFeaturizer:
        """Arcsinh-scale gene expression and fit variance selection on training ids.

        :param features: Cell-line multi-omics feature dataset.
        :param entity_ids: Optional explicit fit ids; otherwise derived from *context*.
        :param context: Optional fit context supplying unique training ids.
        :returns: Fitted featurizer instance.
        """
        ids = np.unique(
            entity_ids if entity_ids is not None else context.unique_train_ids if context else features.identifiers
        )
        matrix = np.arcsinh(stack_view_matrix(features, "gene_expression", ids))
        self._scaler.fit(matrix)
        scaled = features.copy()
        for identifier in scaled.identifiers:
            value = np.asarray(scaled.features[str(identifier)]["gene_expression"], dtype=float)
            scaled.features[str(identifier)]["gene_expression"] = self._scaler.transform(np.arcsinh(value)[None, :])[0]
        self._selector.fit_on_ids(scaled, ids)
        self._feature_names = {
            "gene_expression": tuple(self._selector.selected_meta_info),
            **{view: feature_names_for_view(features, view) for view in _VIEWS[1:]},
        }
        return self

    def _gene_expression(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        matrix = np.arcsinh(stack_view_matrix(features, "gene_expression", entity_ids))
        return self._scaler.transform(matrix)[:, self._selector.mask].astype(np.float32)

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        """Return scaled, variance-selected gene-expression features.

        :param features: Cell-line multi-omics feature dataset.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Float matrix of selected gene-expression features.
        """
        return self._gene_expression(features, entity_ids)

    def transform_blocks(self, features: FeatureDataset, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return per-omics numeric blocks for MOLIR.

        :param features: Cell-line multi-omics feature dataset.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Mapping of omics view name to numeric blocks.
        """
        return {
            "gene_expression": numeric_feature_block(
                self._gene_expression(features, entity_ids), feature_names=self._feature_names.get("gene_expression")
            ),
            **{
                view: numeric_feature_block(
                    stack_view_matrix(features, view, entity_ids).astype(np.float32),
                    feature_names=self._feature_names.get(view),
                )
                for view in _VIEWS[1:]
            },
        }

    @property
    def output_dim(self) -> int:
        """Return the number of selected gene-expression features.

        :returns: Selected gene-expression feature count.
        """
        return int(self._selector.mask.sum())

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable variance-selection feature count.

        :returns: Ray Tune-style hyperparameter space mapping.
        """
        return {"n_gene_expression_features": {"type": "int", "low": 1, "high": 1000, "default": 1000}}

    def get_state(self) -> dict[str, object]:
        """Serialize scaler, selector, and feature-name metadata.

        :returns: Fitted state mapping.
        """
        return {
            "scaler": self._scaler,
            "selector": self._selector,
            "feature_names": self._feature_names,
            "n_gene_expression_features": self._n_features,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore scaler, selector, and feature names from ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        scaler = state.get("scaler")
        if isinstance(scaler, StandardScaler):
            self._scaler = scaler
        selector = state.get("selector")
        if isinstance(selector, VarianceFeatureSelector):
            self._selector = selector
        names = state.get("feature_names")
        if isinstance(names, dict):
            self._feature_names = {str(key): value for key, value in names.items()}
