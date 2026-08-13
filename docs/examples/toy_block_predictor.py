"""Block predictor example: ridge on one named cell-line block.

``BlockPredictor`` is the interface for predictors that must keep the sides (or
individual featurizer outputs) apart instead of flattening everything into one
matrix. ``required_cell_line_blocks`` names the blocks the stack has to supply;
composition rejects a recipe whose featurizers emit none of them.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.linear_model import Ridge

from drevalpy.plugin import (
    BlockPredictor,
    FeatureFormat,
    ModelInputBatch,
    register_predictor,
)

BLOCK = "gene_expression"


@register_predictor(
    "toyBlockRidge",
    description="Ridge regression on a named gene-expression block plus the drug features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class ToyBlockRidgePredictor(BlockPredictor):
    """Read one named block instead of the flattened design matrix."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = (BLOCK,)

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Create an untrained predictor.

        Args:
            hyperparameters: Overrides merged onto the declared defaults.
        """
        super().__init__(hyperparameters)
        self._estimator: Ridge | None = None

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train on the named block, widened with the drug features when present."""
        self._estimator = Ridge(alpha=1.0)
        self._estimator.fit(self._design_matrix(batch), batch.response)

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Score the same design matrix ``_fit`` was trained on."""
        if self._estimator is None:
            msg = "ToyBlockRidgePredictor must be fitted before predicting"
            raise RuntimeError(msg)
        return np.asarray(self._estimator.predict(self._design_matrix(batch)), dtype=np.float64)

    @staticmethod
    def _design_matrix(batch: ModelInputBatch) -> np.ndarray:
        """Expand the entity-level block to one row per pair.

        Blocks are indexed by entity, not by pair, so they have to be expanded
        through ``cell_line_pair_idx`` before a pair-level model can use them.
        """
        matrix = batch.cell_line_blocks[BLOCK].values[batch.cell_line_pair_idx]
        if batch.drug_features is not None and batch.drug_pair_idx is not None:
            matrix = np.hstack([matrix, batch.drug_features[batch.drug_pair_idx]])
        return matrix

    def get_state(self) -> dict[str, object]:
        """Return the trained estimator for checkpoint persistence."""
        return {} if self._estimator is None else {"estimator": self._estimator}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore the state produced by ``get_state``."""
        estimator = state.get("estimator")
        if estimator is not None:
            self._estimator = estimator  # type: ignore[assignment]

    def is_fitted(self) -> bool:
        """Report whether ``fit`` has run."""
        return self._estimator is not None
