"""DIPK gene-expression autoencoder featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.feature_source import FeatureSource
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
    GeneExpressionEncoder,
    encode_gene_expression,
    train_gene_expession_autoencoder,
)
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "dipkGeneExpression",
    description="DIPK gene-expression autoencoder embeddings.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class DIPKGeneExpressionFeaturizer(CellLineFeaturizer):
    """Encode intersection genes into the 512-dimensional DIPK representation."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)
    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

    def __init__(self, *, epochs_autoencoder: int = 100) -> None:
        """Store the autoencoder training epoch budget.

        :param epochs_autoencoder: Number of autoencoder training epochs.
        """
        self._epochs = int(epochs_autoencoder)
        self._encoder: GeneExpressionEncoder | None = None
        self._input_dim = 0
        self._latent_dim = 512

    def fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> DIPKGeneExpressionFeaturizer:
        """Train the DIPK autoencoder on pair-expanded train and validation IDs.

        :param source: Feature source providing view matrices.
        :param entity_ids: Unused; IDs come from *context*.
        :param context: Fit context with train and early-stopping cell-line IDs.
        :returns: Fitted featurizer instance.
        :raises ValueError: If *context* or required ID sets are missing or empty.
        """
        _ = entity_ids
        if context is None:
            raise ValueError("dipkGeneExpression requires FeaturizerFitContext")
        train_ids = context.pair_expanded_train_ids
        validation_ids = context.pair_expanded_early_stopping_ids
        if len(train_ids) == 0 or len(validation_ids) == 0:
            raise ValueError("dipkGeneExpression requires non-empty train and early-stopping IDs")
        train = source.get_view_matrix("gene_expression", train_ids)
        validation = source.get_view_matrix("gene_expression", validation_ids)
        self._input_dim = int(train.shape[1])
        self._encoder = train_gene_expession_autoencoder(train, validation, self._epochs)
        self._latent_dim = int(self._encoder.latent_dim)
        return self

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Encode gene expression into DIPK latent vectors.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Float matrix of latent embeddings.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._encoder is None:
            raise RuntimeError("DIPKGeneExpressionFeaturizer must be fit before transform")
        return encode_gene_expression(source.get_view_matrix("gene_expression", entity_ids), self._encoder).astype(
            np.float32
        )

    def transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return a single ``gene_expression`` numeric block.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Mapping with one numeric ``gene_expression`` block.
        """
        return {"gene_expression": numeric_feature_block(self.transform(source, entity_ids))}

    @property
    def output_dim(self) -> int:
        """Return latent embedding width after fitting.

        :returns: Latent dimensionality, or ``0`` before fitting.
        """
        return self._latent_dim if self._encoder is not None else 0

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable autoencoder epoch count.

        :returns: Ray Tune-style hyperparameter space mapping.
        """
        return {"epochs_autoencoder": {"type": "int", "low": 1, "high": 500, "default": 100}}

    def get_state(self) -> dict[str, object]:
        """Serialize fitted encoder weights and shape metadata.

        :returns: State mapping, or an empty dict before fitting.
        """
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
        """Restore a serialized DIPK encoder from ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
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
