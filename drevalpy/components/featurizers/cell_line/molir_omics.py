"""MOLIR multi-omics preprocessing featurizer."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.data_loading.multiomics import get_multiomics_feature_dataset
from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._matrix import feature_names_for_view, stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.preprocessing import VarianceFeatureSelector
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.datasets.dataset import FeatureDataset

_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


@register_cell_line_featurizer(
    "molirOmics",
    description="MOLIR multi-omics input preparation.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class MOLIROmicsFeaturizer(CellLineFeaturizer):
    """Arcsinh-scale and select variable gene-expression features for MOLIR."""

    def __init__(self, *, n_gene_expression_features: int = 1000) -> None:
        self._n_features = int(n_gene_expression_features)
        self._scaler = StandardScaler()
        self._selector = VarianceFeatureSelector("gene_expression", self._n_features)
        self._feature_names: dict[str, tuple[str, ...] | None] = {}

    @classmethod
    def load_features(cls, data_path: str, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load the intersection-gene multi-omics tables required by MOLIR."""
        _ = cls, kwargs
        return get_multiomics_feature_dataset(
            data_path,
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
        return self._gene_expression(features, entity_ids)

    def transform_blocks(self, features: FeatureDataset, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
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
        return int(self._selector.mask.sum())

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {"n_gene_expression_features": {"type": "int", "low": 1, "high": 1000, "default": 1000}}

    def get_state(self) -> dict[str, object]:
        return {
            "scaler": self._scaler,
            "selector": self._selector,
            "feature_names": self._feature_names,
            "n_gene_expression_features": self._n_features,
        }

    def set_state(self, state: dict[str, object]) -> None:
        scaler = state.get("scaler")
        if isinstance(scaler, StandardScaler):
            self._scaler = scaler
        selector = state.get("selector")
        if isinstance(selector, VarianceFeatureSelector):
            self._selector = selector
        names = state.get("feature_names")
        if isinstance(names, dict):
            self._feature_names = {str(key): value for key, value in names.items()}
