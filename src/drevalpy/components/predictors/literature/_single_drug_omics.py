"""Per-drug three-omic predictor shell shared by MOLIR and SuperFELTR.

:class:`SingleDrugOmicsPredictor` holds what both predictors share: class variables,
``__init__``, per-drug fit and predict routing, and the ``algorithms`` state layout. The
helpers it relies on (feature names, realignment, checkpoint directories, state decoding)
are in ``_omic_helpers.py``.

Only ``numpy`` is imported at module scope - both predictors are registered on
``import drevalpy``, so the training stack must stay out. The leading underscore
keeps the module out of ``registry/_builtins.py::_discover_modules``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._omic_helpers import (
    OMIC_BLOCK_NAMES,
    OmicFeatureNames,
    feature_names_from_payload,
    iter_drug_subsets,
    restore_algorithms,
)
from drevalpy.components.predictors.single_drug_routing import require_known_training_keys, routing_keys
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode

if TYPE_CHECKING:
    from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


class SingleDrugOmicsPredictor(BlockPredictor):
    """Per-drug multi-omics predictor shell shared by MOLIR and SuperFELTR.

    A subclass trains one model per drug (:meth:`_fit_single_drug`), scores one drug's pairs
    (:meth:`_predict_single_drug`) and converts one model to and from its payload
    (:meth:`_serialize_model`, :meth:`_restore_model`). Everything around that is shared: the
    block contract, per-drug routing, and the ``algorithms`` layout of the state.
    """

    scope: ClassVar[ModelScope] = ModelScope.SINGLE_DRUG
    required_sample_blocks: ClassVar[tuple[str, ...]] = OMIC_BLOCK_NAMES
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("identity",)
    required_sample_block_specs: ClassVar[tuple[BlockSpec, ...]] = tuple(
        BlockSpec(name, FeatureFormat.NUMERIC_MATRIX) for name in OMIC_BLOCK_NAMES
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("identity", FeatureFormat.NUMERIC_MATRIX),)
    validate_drug_graphs: ClassVar[bool] = False
    supports_early_stopping: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize predictor with optional hyperparameter overrides.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._models: dict[str, Any] = {}
        self._feature_names: dict[str, OmicFeatureNames] = {}

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train one model per drug.

        :param batch: Training batch with all required sample omics blocks.
        """
        require_known_training_keys(routing_keys(batch))
        self._models = {}
        self._feature_names = {}
        for drug_id, sub in iter_drug_subsets(batch):
            self._fit_single_drug(drug_id, sub)

    @abstractmethod
    def _fit_single_drug(self, drug_id: str, batch: ModelInputBatch) -> None:
        """Fit the model of one drug and record it in ``self._models``.

        :param drug_id: Identifier of the drug to train on.
        :param batch: Subset batch for this drug.
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
            if drug_id == "" or model is None:
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

    def is_fitted(self) -> bool:
        """Report whether trained models exist.

        :returns: ``True`` when at least one per-drug model has been fitted.
        """
        return bool(self._models)

    def get_state(self) -> dict[str, object]:
        """Serialize fitted state for all per-drug models.

        :returns: Mapping with one serialized blob per drug and the hyperparameters; empty before fitting.
        """
        if not self._models:
            return {}
        return {
            "algorithms": {drug_id: self._serialize_model(drug_id, model) for drug_id, model in self._models.items()},
            "predictor_hyperparameters": dict(self._hyperparameters),
        }

    @abstractmethod
    def _serialize_model(self, drug_id: str, model: Any) -> bytes:
        """Serialize one drug's model together with its recorded feature names.

        :param drug_id: Drug identifier.
        :param model: The drug's fitted model.
        :returns: The payload bytes stored under ``algorithms``.
        """

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state from serialized algorithm blobs.

        :param state: State mapping from ``get_state``.
        :raises PredictorStateError: If the state is malformed or a payload lacks a required entry.
        """
        owner = type(self).__name__
        hyperparameters, payloads = restore_algorithms(state, owner)
        models: dict[str, Any] = {}
        feature_names: dict[str, OmicFeatureNames] = {}
        for drug_id, payload in payloads.items():
            feature_names[drug_id] = feature_names_from_payload(payload)
            try:
                models[drug_id] = self._restore_model(payload)
            except KeyError as exc:
                msg = f"{owner} payload for {drug_id!r} is missing {exc}"
                raise PredictorStateError(msg) from exc
        self._hyperparameters = hyperparameters
        self._models = models
        self._feature_names = feature_names

    @abstractmethod
    def _restore_model(self, payload: dict[str, Any]) -> Any:
        """Rebuild one drug's model from its deserialized payload.

        :param payload: Deserialized per-drug payload.
        :returns: The restored model.
        """
