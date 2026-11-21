"""Per-drug three-omic predictor shell shared by MOLIR and SuperFELTR.

:class:`SingleDrugOmicsPredictor` holds what both predictors share on top of
:class:`~drevalpy.components.predictors.abstract.single_drug.SingleDrugPredictor`: the block
contract, early-stopping support and the feature names the models were trained on. The helpers
it relies on (feature names, realignment, early-stopping split) are in ``_omic_helpers.py``.

Only ``numpy`` is imported at module scope - both predictors are registered on
``import drevalpy``, so the training stack must stay out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import BlockSpec
from drevalpy.components.predictors.abstract.single_drug import SingleDrugPredictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.literature._omic_helpers import (
    OMIC_BLOCK_NAMES,
    OmicFeatureNames,
    aligned_pair_matrices,
    omic_feature_names,
)

if TYPE_CHECKING:
    from drevalpy.types.data.predictor_input import PredictorInput


class SingleDrugOmicsPredictor(SingleDrugPredictor):
    """Per-drug multi-omics predictor shell shared by MOLIR and SuperFELTR.

    A subclass trains one model per drug (``_fit_drug``) whose ``predict`` takes the three
    pair-level omic matrices; the shell realigns their columns onto the feature names seen in
    training before scoring.
    """

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(*(BlockSpec(name) for name in OMIC_BLOCK_NAMES))
    supports_early_stopping: ClassVar[bool] = True

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize predictor with optional hyperparameter overrides.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._feature_names = OmicFeatureNames(None, None, None)

    def _fit(self, data: PredictorInput) -> None:
        """Record the omic feature names of *data*, then train one model per drug.

        :param data: Training batch with all required sample omics blocks.
        """
        self._feature_names = omic_feature_names(data)
        super()._fit(data)

    def _predict_drug(self, drug_id: str, model: Any, data: PredictorInput) -> np.ndarray:
        """Score the pairs of one drug on omic columns realigned to the training order.

        :param drug_id: Drug whose pairs *data* holds.
        :param model: The drug's fitted model.
        :param data: The drug's pairs to score.
        :returns: Predicted responses, one per pair.
        """
        return np.atleast_1d(model.predict(*aligned_pair_matrices(data, self._feature_names)))
