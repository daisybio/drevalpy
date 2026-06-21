"""Single-view cell-line featurizer with optional gene-expression scaling."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.models.utils import (
    ProteomicsMedianCenterAndImputeTransformer,
    prepare_proteomics,
    scale_gene_expression,
)


@register_cell_line_featurizer(
    "view",
    description="Pass through one dense cell-line view from a FeatureDataset.",
    category="native",
)
class ViewCellLineFeaturizer(CellLineFeaturizer):
    """Featurize one cell-line view without additional transformation."""

    def __init__(self, *, view: str = "gene_expression") -> None:
        self._view = view
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> ViewCellLineFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        matrix = stack_view_matrix(features, self._view, ids)
        self._output_dim = int(matrix.shape[1])
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        return stack_view_matrix(features, self._view, entity_ids).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim


@register_cell_line_featurizer(
    "scaledGeneExpression",
    description="Landmark gene expression with arcsinh transform and scaling.",
    category="native",
)
class ScaledGeneExpressionFeaturizer(CellLineFeaturizer):
    """Match sklearn baseline gene-expression preprocessing."""

    def __init__(self, *, view: str = "gene_expression") -> None:
        self._view = view
        self._scaler = StandardScaler()
        self._output_dim = 0
        self._fitted_features = None

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> ScaledGeneExpressionFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        scaled = scale_gene_expression(
            cell_line_input=features.copy(),
            cell_line_ids=np.unique(ids),
            training=True,
            gene_expression_scaler=self._scaler,
        )
        self._fitted_features = scaled
        matrix = stack_view_matrix(scaled, self._view, np.array(list(scaled.features.keys())))
        self._output_dim = int(matrix.shape[1])
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        if self._fitted_features is None:
            msg = "ScaledGeneExpressionFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        scaled = scale_gene_expression(
            cell_line_input=features.copy(),
            cell_line_ids=np.unique(entity_ids),
            training=False,
            gene_expression_scaler=self._scaler,
        )
        return stack_view_matrix(scaled, self._view, entity_ids).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def get_state(self) -> dict[str, object]:
        return {
            "gene_expression_scaler": self._scaler,
            "fitted": self._fitted_features is not None,
        }

    def set_state(self, state: dict[str, object]) -> None:
        scaler = state.get("gene_expression_scaler")
        if scaler is not None:
            self._scaler = scaler
        if state.get("fitted"):
            self._fitted_features = object()


@register_cell_line_featurizer(
    "pca",
    description="PCA compression of one dense cell-line view fit on training cell lines.",
    category="native",
)
class PCACellLineFeaturizer(CellLineFeaturizer):
    """Reduce one cell-line view with PCA."""

    def __init__(self, *, view: str = "gene_expression", n_components: int = 128) -> None:
        from sklearn.decomposition import PCA

        self._view = view
        self._n_components = int(n_components)
        self._pca = PCA(n_components=self._n_components)
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> PCACellLineFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        matrix = stack_view_matrix(features, self._view, ids)
        n_components = min(self._n_components, matrix.shape[0], matrix.shape[1])
        self._pca.n_components = n_components
        self._pca.fit(matrix)
        self._output_dim = n_components
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        matrix = stack_view_matrix(features, self._view, entity_ids)
        return self._pca.transform(matrix).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "n_components": {"type": "int", "low": 8, "high": 512, "default": 128},
        }


@register_cell_line_featurizer(
    "proteomics",
    description="Proteomics view with median centering and imputation.",
    category="native",
)
class ProteomicsCellLineFeaturizer(CellLineFeaturizer):
    """Match sklearn baseline proteomics preprocessing."""

    def __init__(
        self,
        *,
        view: str = "proteomics",
        proteomics_feature_threshold: float = 0.7,
        proteomics_n_features: int = 1000,
        proteomics_normalization_width: float = 0.3,
        proteomics_normalization_downshift: float = 1.8,
    ) -> None:
        self._view = view
        self._transformer = ProteomicsMedianCenterAndImputeTransformer(
            feature_threshold=proteomics_feature_threshold,
            n_features=proteomics_n_features,
            normalization_width=proteomics_normalization_width,
            normalization_downshift=proteomics_normalization_downshift,
        )
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> ProteomicsCellLineFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        processed = prepare_proteomics(
            cell_line_input=features.copy(),
            cell_line_ids=np.unique(ids),
            training=True,
            transformer=self._transformer,
        )
        matrix = stack_view_matrix(processed, self._view, np.array(list(processed.features.keys())))
        self._output_dim = int(matrix.shape[1])
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        processed = prepare_proteomics(
            cell_line_input=features.copy(),
            cell_line_ids=np.unique(entity_ids),
            training=False,
            transformer=self._transformer,
        )
        return stack_view_matrix(processed, self._view, entity_ids).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def get_state(self) -> dict[str, object]:
        return {"proteomics_transformer": self._transformer}

    def set_state(self, state: dict[str, object]) -> None:
        transformer = state.get("proteomics_transformer")
        if transformer is not None:
            self._transformer = transformer
