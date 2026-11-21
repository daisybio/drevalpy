"""Block predictor example: ridge on one named sample block.

A predictor that must keep the sides (or individual featurizer outputs) apart instead
of flattening everything into one matrix subclasses ``Predictor`` directly and names
the blocks it reads in ``sample_inputs`` / ``drug_inputs``. Config validation rejects
a recipe whose sample featurizer does not emit them.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.linear_model import Ridge

from drevalpy.plugin import (
    BlockSpec,
    FeatureFormat,
    Inputs,
    ModelInputBatch,
    Predictor,
    register_predictor,
)

BLOCK = "gene_expression"


@register_predictor(
    "toyBlockRidge",
    description="Ridge regression on a named gene-expression block plus the drug features.",
)
class ToyBlockRidgePredictor(Predictor):
    """Read one named block instead of the flattened design matrix."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec(BLOCK, FeatureFormat.NUMERIC_MATRIX))
    drug_inputs: ClassVar[Inputs] = Inputs.matrix()

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Create an untrained predictor.

        :param hyperparameters: Values merged onto the defaults of the space.
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
        through ``sample_pair_idx`` before a pair-level model can use them.
        """
        matrix = batch.sample.blocks[BLOCK].values[batch.sample_pair_idx]
        if batch.drug is not None and batch.drug_pair_idx is not None:
            matrix = np.hstack([matrix, batch.drug.matrix[batch.drug_pair_idx]])
        return matrix
