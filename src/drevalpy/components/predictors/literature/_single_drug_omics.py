"""Per-drug three-omic plumbing shared by MOLIR and SuperFELTR.

Both are single-drug models over the same three sample views, and both carried
their own copy of: the per-drug checkpoint directory, the record of which feature
names a model was trained on, the column realignment applied at predict time, the
early-stopping index lookup, the feature-name (de)serialization, and the predictor
shell around them. :class:`SingleDrugOmicsPredictor` is that shell: class variables,
``__init__``, per-drug fit and predict routing, and the ``algorithms`` state layout.

Only ``numpy`` is imported at module scope - both predictors are registered on
``import drevalpy``, so the training stack must stay out. The leading underscore
keeps the module out of ``registry/_builtins.py::_discover_modules``.
"""

from __future__ import annotations

import hashlib
from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from upath import UPath

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit
from drevalpy.components.predictors.literature.molir._omics import _realign_omic_matrix
from drevalpy.components.predictors.single_drug_routing import require_known_training_keys, routing_keys
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode
from drevalpy.utils.torch_io import load_trusted_mapping

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from drevalpy.types.data.batch.model_input_batch import ModelInputBatch

#: The three sample views both models consume, in encoder order.
OMIC_BLOCK_NAMES = ("gene_expression", "mutations", "copy_number_variation_gistic")


@dataclass(frozen=True)
class OmicFeatureNames:
    """The feature names a per-drug model was trained on, one tuple per omic view."""

    gene_expression: tuple[str, ...] | None
    mutations: tuple[str, ...] | None
    copy_number_variation: tuple[str, ...] | None

    def as_tuple(self) -> tuple[tuple[str, ...] | None, ...]:
        """Return the three tuples in :data:`OMIC_BLOCK_NAMES` order.

        :returns: Feature names per view, aligned with ``OMIC_BLOCK_NAMES``.
        """
        return (self.gene_expression, self.mutations, self.copy_number_variation)


@dataclass(frozen=True)
class OmicMatrices:
    """One batch's three entity-level omic matrices, as float32."""

    gene_expression: np.ndarray
    mutations: np.ndarray
    copy_number_variation: np.ndarray

    def widths(self) -> tuple[int, int, int]:
        """Return the feature width of each view.

        :returns: Tuple of ``(expression, mutation, cnv)`` widths.
        """
        return (
            self.gene_expression.shape[1],
            self.mutations.shape[1],
            self.copy_number_variation.shape[1],
        )

    def split(self, pair_idx: np.ndarray, response: np.ndarray) -> OmicsSplit:
        """Pair these entity matrices with a pair-level response and index array.

        :param pair_idx: Row of each matrix to use for each pair.
        :param response: Pair-level response values.
        :returns: A loader-ready split.
        """
        return OmicsSplit(
            gene_expression=self.gene_expression,
            mutations=self.mutations,
            copy_number=self.copy_number_variation,
            response=response,
            pair_idx=pair_idx,
        )


def omic_matrices(batch: ModelInputBatch) -> OmicMatrices:
    """Read the three sample views out of *batch*.

    :param batch: Batch carrying all three blocks.
    :returns: The entity-level matrices as float32.
    """
    values = [np.asarray(batch.sample.blocks[name].values, dtype=np.float32) for name in OMIC_BLOCK_NAMES]
    return OmicMatrices(*values)


def omic_feature_names(batch: ModelInputBatch) -> OmicFeatureNames:
    """Record the feature names of the three sample views of *batch*.

    :param batch: Batch carrying all three blocks.
    :returns: The recorded feature names.
    """
    return OmicFeatureNames(*(batch.sample.blocks[name].feature_names for name in OMIC_BLOCK_NAMES))


def checkpoint_dir_for_drug(base_dir: UPath | str, drug_id: str) -> UPath:
    """Return a unique checkpoint directory for a given drug.

    Hashing keeps the segment filesystem-safe: drug ids come from the dataset and
    routinely contain ``/`` and other separators.

    :param base_dir: Base directory for checkpoints.
    :param drug_id: Drug identifier to hash.
    :returns: Path to the drug-specific checkpoint directory.
    """
    digest = hashlib.sha256(drug_id.encode()).hexdigest()[:16]
    return UPath(base_dir) / f"drug_{digest}"


def iter_drug_subsets(batch: ModelInputBatch) -> Iterator[tuple[str, ModelInputBatch]]:
    """Split *batch* per drug, giving each sub-batch its own checkpoint directory.

    :param batch: Full training batch.
    :yields: ``(drug_id, sub_batch)`` for every drug present.
    """
    from drevalpy.components.contracts.training_context import TrainingContext
    from drevalpy.components.predictors.single_drug_routing import iter_drug_masks

    base_dir = batch.training_context.checkpoint_dir
    for drug_id, mask in iter_drug_masks(batch):
        context = TrainingContext(checkpoint_dir=checkpoint_dir_for_drug(base_dir, drug_id))
        yield drug_id, replace(batch.subset_pairs(mask), training_context=context)


def early_stopping_indices(batch: ModelInputBatch) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Map the batch's early-stopping response onto sample entity rows.

    :param batch: Training batch.
    :returns: Tuple of ``(val_pair_idx, val_response)``, both ``None`` when there is
        no usable early-stopping split.
    """
    es_resp = batch.early_stopping_response
    if es_resp is None or len(es_resp) < 2:
        return None, None

    entity_map = {str(eid): row for row, eid in enumerate(batch.sample.entity_ids)}
    val_idx = np.array([entity_map[str(cl_id)] for cl_id in es_resp.sample_ids], dtype=np.intp)
    return val_idx, np.asarray(es_resp.response, dtype=np.float32)


def validation_split(matrices: OmicMatrices, batch: ModelInputBatch) -> OmicsSplit | None:
    """Build the early-stopping split over the same entity matrices as training.

    :param matrices: The batch's entity-level omic matrices.
    :param batch: Training batch carrying the early-stopping response.
    :returns: The validation split, or ``None`` when there is no usable one.
    """
    val_pair_idx, val_response = early_stopping_indices(batch)
    if val_pair_idx is None or val_response is None:
        return None
    return matrices.split(val_pair_idx, val_response)


def aligned_pair_matrices(
    batch: ModelInputBatch,
    feature_names: OmicFeatureNames,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand the three views to pair level and realign them onto the trained order.

    :param batch: Batch to score.
    :param feature_names: Feature names the model was trained on.
    :returns: Tuple of ``(expression, mutation, cnv)`` pair-level matrices.
    """
    pair_idx = batch.sample_pair_idx
    aligned = []
    for name, trained in zip(OMIC_BLOCK_NAMES, feature_names.as_tuple(), strict=True):
        block = batch.sample.blocks[name]
        values = np.asarray(block.values[pair_idx], dtype=np.float32)
        aligned.append(_align(values, trained, block.feature_names))
    return aligned[0], aligned[1], aligned[2]


def feature_names_payload(feature_names: OmicFeatureNames | None) -> dict[str, list[str] | None]:
    """Render feature names into the serializable form both predictors persist.

    :param feature_names: Recorded feature names, or ``None`` when unknown.
    :returns: Mapping of payload key to feature-name list.
    """
    names = feature_names.as_tuple() if feature_names is not None else (None, None, None)
    keys = ("gene_expression_features", "mutations_features", "copy_number_variation_features")
    return {key: list(value) if value else None for key, value in zip(keys, names, strict=True)}


def feature_names_from_payload(payload: dict[str, Any]) -> OmicFeatureNames:
    """Read feature names back out of a persisted payload.

    :param payload: Deserialized per-drug payload.
    :returns: The recorded feature names.
    """
    keys = ("gene_expression_features", "mutations_features", "copy_number_variation_features")
    values = [tuple(payload[key]) if payload.get(key) else None for key in keys]
    return OmicFeatureNames(*values)


def _align(
    values: np.ndarray,
    model_features: tuple[str, ...] | None,
    current_features: tuple[str, ...] | None,
) -> np.ndarray:
    """Align omic matrix columns to match the training feature order.

    :param values: Input matrix to realign.
    :param model_features: Feature names expected by the model.
    :param current_features: Feature names in the current batch.
    :returns: Realigned matrix.
    """
    if model_features is None or current_features is None:
        return values
    if len(model_features) == values.shape[1] and model_features == current_features:
        return values
    return _realign_omic_matrix(values, model_features, current_features)


def restore_algorithms(
    state: Mapping[str, Any],
    owner: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Decode the per-drug payloads a single-drug omics predictor stored in ``get_state``.

    :param state: State mapping handed to ``set_state``.
    :param owner: Predictor name used in error messages.
    :returns: The predictor hyperparameters and one deserialized payload per drug id.
    :raises PredictorStateError: If ``algorithms`` is not a mapping of byte blobs, a blob cannot be
        deserialized, or ``predictor_hyperparameters`` is not a mapping.
    """
    algorithms = state.get("algorithms")
    if not isinstance(algorithms, dict):
        msg = f"{owner} state requires an 'algorithms' mapping"
        raise PredictorStateError(msg)
    hyperparameters = state.get("predictor_hyperparameters")
    if not isinstance(hyperparameters, dict):
        msg = f"{owner} state requires predictor_hyperparameters"
        raise PredictorStateError(msg)
    payloads: dict[str, dict[str, Any]] = {}
    for drug_id, blob in algorithms.items():
        if not isinstance(blob, (bytes, bytearray)):
            msg = f"{owner} algorithm payload for {drug_id!r} must be bytes"
            raise PredictorStateError(msg)
        try:
            payloads[str(drug_id)] = load_trusted_mapping(bytes(blob))
        except Exception as exc:
            msg = f"{owner} algorithm payload for {drug_id!r} could not be deserialized"
            raise PredictorStateError(msg) from exc
    return dict(hyperparameters), payloads


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
