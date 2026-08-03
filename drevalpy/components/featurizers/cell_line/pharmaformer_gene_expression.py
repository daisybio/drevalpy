"""PharmaFormer gene-expression preprocessing featurizer."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.data_loading.multiomics import load_and_select_gene_features
from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._matrix import feature_names_for_view, stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.datasets.dataset import FeatureDataset


@register_cell_line_featurizer(
    "pharmaFormerGeneExpression",
    description="Reduced landmark genes scaled for PharmaFormer.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PharmaFormerGeneExpressionFeaturizer(CellLineFeaturizer):
    """Apply the PharmaFormer StandardScaler then MinMaxScaler sequence."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._minmax = MinMaxScaler()
        self._feature_names: tuple[str, ...] | None = None
        self._output_dim = 0
        self._is_fitted = False

    @classmethod
    def load_features(cls, data_path: str, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load reduced landmark gene-expression values without arcsinh."""
        _ = cls, kwargs
        return load_and_select_gene_features("gene_expression", "landmark_genes_reduced", data_path, dataset_name)

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> PharmaFormerGeneExpressionFeaturizer:
        _ = entity_ids
        if context is None:
            raise ValueError("pharmaFormerGeneExpression requires FeaturizerFitContext")
        matrix = stack_view_matrix(features, "gene_expression", context.pair_expanded_train_ids)
        self._minmax.fit(self._scaler.fit_transform(matrix))
        self._feature_names = feature_names_for_view(features, "gene_expression")
        self._output_dim = int(matrix.shape[1])
        self._is_fitted = True
        return self

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("PharmaFormerGeneExpressionFeaturizer must be fit before transform")
        matrix = stack_view_matrix(features, "gene_expression", entity_ids)
        return self._minmax.transform(self._scaler.transform(matrix)).astype(np.float32)

    def transform_blocks(self, features: FeatureDataset, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        return {
            "gene_expression": numeric_feature_block(
                self.transform(features, entity_ids),
                feature_names=self._feature_names,
            )
        }

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def get_state(self) -> dict[str, object]:
        if not self._is_fitted:
            return {}
        return {
            "scaler": self._scaler,
            "minmax": self._minmax,
            "feature_names": self._feature_names,
            "output_dim": self._output_dim,
            "fitted": True,
        }

    def set_state(self, state: dict[str, object]) -> None:
        scaler, minmax = state.get("scaler"), state.get("minmax")
        if isinstance(scaler, StandardScaler):
            self._scaler = scaler
        if isinstance(minmax, MinMaxScaler):
            self._minmax = minmax
        names = state.get("feature_names")
        if isinstance(names, tuple):
            self._feature_names = tuple(str(name) for name in names)
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
        self._is_fitted = bool(state.get("fitted"))
