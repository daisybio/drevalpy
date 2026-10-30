"""Per-drug three-omic predictor shell shared by MOLIR and SuperFELTR.

:class:`SingleDrugOmicsPredictor` holds what both predictors share: class variables,
``__init__``, and per-drug fit and predict routing. The helpers it relies on (feature names,
realignment, checkpoint directories) are in ``_omic_helpers.py``.

Only ``numpy`` is imported at module scope - both predictors are registered on
``import drevalpy``, so the training stack must stay out.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.literature._omic_helpers import (
    OMIC_BLOCK_NAMES,
    OmicFeatureNames,
    iter_drug_subsets,
    omic_feature_names,
)
from drevalpy.components.predictors.single_drug_routing import routing_keys, run_per_drug
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.enums.model_scope import ModelScope

if TYPE_CHECKING:
    from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


class SingleDrugOmicsPredictor(Predictor):
    """Per-drug multi-omics predictor shell shared by MOLIR and SuperFELTR.

    A subclass trains one model per drug (:meth:`_fit_single_drug`) and scores one drug's pairs
    (:meth:`_predict_single_drug`). Everything around that is shared: the block contract and
    per-drug routing by ``batch.drug_ids``, so the drug side reads no features.
    """

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(
        *(BlockSpec(name, FeatureFormat.NUMERIC_MATRIX) for name in OMIC_BLOCK_NAMES)
    )
    drug_inputs: ClassVar[Inputs] = Inputs.none()
    scope: ClassVar[ModelScope] = ModelScope.SINGLE_DRUG
    supports_early_stopping: ClassVar[bool] = True

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize predictor with optional hyperparameter overrides.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._models: dict[str, Any] = {}
        self._feature_names: dict[str, OmicFeatureNames] = {}

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train one model per drug, in ``batch.training_context.n_jobs`` worker processes.

        :param batch: Training batch with all required sample omics blocks.
        """
        self._models = {}
        self._feature_names = {}
        fitted = run_per_drug(self._fit_single_drug, iter_drug_subsets(batch), batch.training_context.n_jobs)
        feature_names = omic_feature_names(batch)
        self._feature_names = {drug_id: feature_names for drug_id in fitted}
        self._models = {drug_id: model for drug_id, model in fitted.items() if model is not None}

    @abstractmethod
    def _fit_single_drug(self, drug_id: str, batch: ModelInputBatch) -> Any | None:
        """Fit the model of one drug.

        This may run in a worker process, so it returns the model instead of recording it on ``self``.

        :param drug_id: Identifier of the drug to train on.
        :param batch: Subset batch for this drug.
        :returns: The fitted model, or ``None`` for a drug that gets no model.
        """

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses, routing each pair to its drug's model.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair, ``NaN`` for drugs without a model.
        """
        keys = routing_keys(batch)
        predictions = np.full(batch.n_pairs, np.nan, dtype=np.float64)
        for drug_id in np.unique(keys):
            model = self._models.get(str(drug_id))
            if model is None:
                continue
            mask = keys == drug_id
            preds = self._predict_single_drug(str(drug_id), model, batch.subset_pairs(mask))
            predictions[mask] = np.asarray(preds, dtype=np.float64).ravel()
        return predictions

    @abstractmethod
    def _predict_single_drug(self, drug_id: str, model: Any, batch: ModelInputBatch) -> np.ndarray:
        """Predict the pairs of one drug.

        :param drug_id: Drug identifier, used to look up the trained feature names.
        :param model: The drug's fitted model.
        :param batch: Subset batch holding only this drug's pairs.
        :returns: Predicted responses, one per pair.
        """
