"""DIPK gene-expression autoencoder featurizer."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
    GeneExpressionEncoder,
    encode_gene_expression,
    train_gene_expession_autoencoder,
)
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.data.features import load_and_select_gene_features
from drevalpy.datasets.dataset import FeatureDataset


@register_cell_line_featurizer(
    "dipkGeneExpression",
    description="DIPK gene-expression autoencoder embeddings.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class DIPKGeneExpressionFeaturizer(CellLineFeaturizer):
    """Encode intersection genes into the 512-dimensional DIPK representation."""

    def __init__(self, *, epochs_autoencoder: int = 100) -> None:
        self._epochs = int(epochs_autoencoder)
        self._encoder: GeneExpressionEncoder | None = None
        self._input_dim = 0
        self._latent_dim = 512

    @classmethod
    def load_features(cls, data_path: str, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load the DIPK intersection gene-expression view."""
        _ = cls, kwargs
        return load_and_select_gene_features("gene_expression", "gene_expression_intersection", data_path, dataset_name)

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> DIPKGeneExpressionFeaturizer:
        _ = entity_ids
        if context is None:
            raise ValueError("dipkGeneExpression requires FeaturizerFitContext")
        train_ids = context.pair_expanded_train_ids
        validation_ids = context.pair_expanded_early_stopping_ids
        if len(train_ids) == 0 or len(validation_ids) == 0:
            raise ValueError("dipkGeneExpression requires non-empty train and early-stopping IDs")
        train = stack_view_matrix(features, "gene_expression", train_ids)
        validation = stack_view_matrix(features, "gene_expression", validation_ids)
        self._input_dim = int(train.shape[1])
        self._encoder = train_gene_expession_autoencoder(train, validation, self._epochs)
        self._latent_dim = int(self._encoder.latent_dim)
        return self

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("DIPKGeneExpressionFeaturizer must be fit before transform")
        return encode_gene_expression(stack_view_matrix(features, "gene_expression", entity_ids), self._encoder).astype(
            np.float32
        )

    def transform_blocks(self, features: FeatureDataset, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        return {"gene_expression": numeric_feature_block(self.transform(features, entity_ids))}

    @property
    def output_dim(self) -> int:
        return self._latent_dim if self._encoder is not None else 0

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {"epochs_autoencoder": {"type": "int", "low": 1, "high": 500, "default": 100}}

    def get_state(self) -> dict[str, object]:
        if self._encoder is None:
            return {}
        return {
            "encoder_state": save_state_dict(self._encoder.state_dict()),
            "input_dim": self._input_dim,
            "latent_dim": self._latent_dim,
            "epochs": self._epochs,
            "epochs_autoencoder": self._epochs,
        }

    def set_state(self, state: dict[str, object]) -> None:
        blob = state.get("encoder_state")
        input_dim = state.get("input_dim")
        latent_dim = state.get("latent_dim", 512)
        if not isinstance(blob, bytes) or not isinstance(input_dim, int) or not isinstance(latent_dim, int):
            return
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        epochs = state.get("epochs", state.get("epochs_autoencoder"))
        if isinstance(epochs, int):
            self._epochs = epochs
        self._encoder = GeneExpressionEncoder(input_dim, latent_dim=latent_dim)
        self._encoder.load_state_dict(load_state_dict(blob))
        self._encoder.eval()
