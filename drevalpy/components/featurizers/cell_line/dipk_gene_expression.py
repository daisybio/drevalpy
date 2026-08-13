"""DIPK gene-expression autoencoder featurizer.

The autoencoder in
``drevalpy.components.predictors.literature.dipk.gene_expression_encoder``
defines ``torch.nn.Module`` subclasses, so importing it costs ~0.32s of ``torch``.
``drevalpy.registry`` imports this module to register the
``dipkGeneExpression`` featurizer on ``import drevalpy``, so the three symbols are
pulled in inside the methods that use them and re-exported lazily below for the
historical import path. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource
from drevalpy.utils.torch_io import load_state_dict, save_state_dict

if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import GeneExpressionEncoder

_RE_EXPORTED = frozenset({"GeneExpressionEncoder", "encode_gene_expression", "train_gene_expession_autoencoder"})


def __getattr__(name: str) -> Any:
    """Resolve the lazily re-exported autoencoder symbols on first access.

    :param name: Attribute being looked up on this module.
    :returns: The requested attribute, imported on demand.
    :raises AttributeError: If *name* is not one of the re-exported symbols.
    """
    if name in _RE_EXPORTED:
        from drevalpy.components.predictors.literature.dipk import gene_expression_encoder

        return getattr(gene_expression_encoder, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


@register(
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

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> DIPKGeneExpressionFeaturizer:
        """Train the DIPK autoencoder on pair-expanded train and validation IDs.

        :param source: Feature source providing view matrices.
        :param entity_ids: Unused; IDs come from *pair_expanded_ids*.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Early-stopping entity IDs with duplicates.
        :returns: Fitted featurizer instance.
        :raises ValueError: If required ID sets are missing or empty.
        """
        from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
            train_gene_expession_autoencoder,
        )

        _ = entity_ids
        mdata = getattr(source, "mdata", None)
        if mdata is not None and pair_expanded_ids is not None:
            precomputed = self.fetch(mdata, pair_expanded_ids)
            if precomputed is not None:
                self._latent_dim = int(precomputed.shape[1])
                return self
        if pair_expanded_ids is None:
            raise ValueError("dipkGeneExpression requires pair_expanded_ids")
        train_ids = pair_expanded_ids
        validation_ids = pair_expanded_es_ids
        if validation_ids is None or len(train_ids) == 0 or len(validation_ids) == 0:
            raise ValueError("dipkGeneExpression requires non-empty train and early-stopping IDs")
        train = source.get_view_matrix("gene_expression", train_ids)
        validation = source.get_view_matrix("gene_expression", validation_ids)
        self._input_dim = int(train.shape[1])
        self._encoder = train_gene_expession_autoencoder(train, validation, self._epochs)
        self._latent_dim = int(self._encoder.latent_dim)
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Encode gene expression into DIPK latent vectors.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Float matrix of latent embeddings.
        :raises RuntimeError: If called before ``fit``.
        """
        from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import encode_gene_expression

        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, entity_ids) if mdata is not None else None
        if precomputed is not None:
            return precomputed.astype(np.float32)
        if self._encoder is None:
            raise RuntimeError("DIPKGeneExpressionFeaturizer must be fit before transform")
        return encode_gene_expression(source.get_view_matrix("gene_expression", entity_ids), self._encoder).astype(
            np.float32
        )

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return a single ``gene_expression`` numeric block.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Mapping with one numeric ``gene_expression`` block.
        """
        return {"gene_expression": numeric_feature_block(self._transform(source, entity_ids))}

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
        from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import GeneExpressionEncoder

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
