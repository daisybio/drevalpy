"""Multi-view cell-line concatenation featurizer."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.data.preprocessing import (
    ProteomicsMedianCenterAndImputeTransformer,
    prepare_expression_and_methylation,
    prepare_proteomics,
)


@register_cell_line_featurizer(
    "multiConcat",
    description="Concatenate multiple cell-line views with baseline preprocessing.",
    category="native",
)
class MultiConcatCellLineFeaturizer(CellLineFeaturizer):
    """Concatenate gene expression, methylation, mutations, CNV, and proteomics views."""

    def __init__(
        self,
        *,
        views: list[str] | None = None,
        methylation_n_components: int = 100,
        proteomics_feature_threshold: float = 0.7,
        proteomics_n_features: int = 1000,
        proteomics_normalization_width: float = 0.3,
        proteomics_normalization_downshift: float = 1.8,
    ) -> None:
        self._views = views or [
            "gene_expression",
            "methylation",
            "mutations",
            "copy_number_variation_gistic",
        ]
        self._methylation_n_components = int(methylation_n_components)
        self._gene_expression_scaler = StandardScaler()
        self._methylation_scaler = StandardScaler()
        self._methylation_pca = PCA(n_components=self._methylation_n_components)
        self._proteomics_transformer = ProteomicsMedianCenterAndImputeTransformer(
            feature_threshold=proteomics_feature_threshold,
            n_features=proteomics_n_features,
            normalization_width=proteomics_normalization_width,
            normalization_downshift=proteomics_normalization_downshift,
        )
        self._output_dim = 0
        self._view_dims: dict[str, int] = {}
        self._train_ids: np.ndarray | None = None

    def _preprocess(self, features, entity_ids: np.ndarray, *, training: bool):
        processed = features.copy()
        if "gene_expression" in self._views or "methylation" in self._views:
            processed = prepare_expression_and_methylation(
                cell_line_input=processed,
                cell_line_ids=np.unique(entity_ids),
                training=training,
                gene_expression_scaler=self._gene_expression_scaler
                if "gene_expression" in self._views
                else None,
                methylation_scaler=self._methylation_scaler if "methylation" in self._views else None,
                methylation_pca=self._methylation_pca if "methylation" in self._views else None,
            )
        if "proteomics" in self._views:
            processed = prepare_proteomics(
                cell_line_input=processed,
                cell_line_ids=np.unique(entity_ids),
                training=training,
                transformer=self._proteomics_transformer,
            )
        return processed

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> MultiConcatCellLineFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        self._train_ids = np.unique(ids)
        processed = self._preprocess(features, ids, training=True)
        entity_matrix_ids = np.array(list(processed.features.keys()))
        rows = [stack_view_matrix(processed, view, entity_matrix_ids) for view in self._views]
        self._view_dims = {view: int(rows[index].shape[1]) for index, view in enumerate(self._views)}
        matrix = np.concatenate(rows, axis=1)
        self._output_dim = int(matrix.shape[1])
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        if self._train_ids is None:
            msg = "MultiConcatCellLineFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        processed = self._preprocess(features, entity_ids, training=False)
        parts = [stack_view_matrix(processed, view, entity_ids) for view in self._views]
        return np.concatenate(parts, axis=1).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def view_dims(self) -> dict[str, int]:
        return dict(self._view_dims)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "methylation_n_components": {"type": "int", "low": 20, "high": 200, "default": 100},
        }

    def get_state(self) -> dict[str, object]:
        return {
            "views": list(self._views),
            "view_dims": dict(self._view_dims),
            "gene_expression_scaler": self._gene_expression_scaler,
            "methylation_scaler": self._methylation_scaler,
            "methylation_pca": self._methylation_pca,
            "proteomics_transformer": self._proteomics_transformer,
            "output_dim": self._output_dim,
            "train_ids": self._train_ids,
        }

    def set_state(self, state: dict[str, object]) -> None:
        views = state.get("views")
        if views is not None:
            self._views = list(views)
        view_dims = state.get("view_dims")
        if view_dims is not None:
            self._view_dims = dict(view_dims)
        gene_expression_scaler = state.get("gene_expression_scaler")
        if gene_expression_scaler is not None:
            self._gene_expression_scaler = gene_expression_scaler
        methylation_scaler = state.get("methylation_scaler")
        if methylation_scaler is not None:
            self._methylation_scaler = methylation_scaler
        methylation_pca = state.get("methylation_pca")
        if methylation_pca is not None:
            self._methylation_pca = methylation_pca
        proteomics_transformer = state.get("proteomics_transformer")
        if proteomics_transformer is not None:
            self._proteomics_transformer = proteomics_transformer
        output_dim = state.get("output_dim")
        if output_dim is not None:
            self._output_dim = int(output_dim)
        train_ids = state.get("train_ids")
        if train_ids is not None:
            self._train_ids = np.asarray(train_ids)
