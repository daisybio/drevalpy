"""Private featurizer/predictor execution stack for DRPModel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from drevalpy.components.core.batch.feature_block import FeatureBlock
from drevalpy.components.core.batch.model_input_batch import ModelInputBatch
from drevalpy.components.core.batch.model_input_build import build_model_input_batch
from drevalpy.components.core.contracts.training_context import TrainingContext
from drevalpy.components.core.features.feature_source import CellLineFeatureSource, DrugFeatureSource, FeatureSource
from drevalpy.components.core.fitting.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.core.fitting.featurizer_label import qualified_featurizer_selector
from drevalpy.components.featurizers._matrix import unique_entity_ids
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.data.structures.response_batch import ResponseBatch
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictionMode
from drevalpy.models.config.resolved import ResolvedModelConfig

if TYPE_CHECKING:
    from drevalpy.data.structures import SplitMask
    from drevalpy.data.structures.dataset import Dataset


def _build_fit_context(
    response: ResponseBatch,
    *,
    early_stopping: ResponseBatch | None,
    side: Literal["cell_line", "drug"],
) -> FeaturizerFitContext:
    if side == "cell_line":
        train_ids = response.cell_line_ids
        es_ids = early_stopping.cell_line_ids if early_stopping is not None else np.array([], dtype=str)
    elif side == "drug":
        train_ids = response.drug_ids
        es_ids = early_stopping.drug_ids if early_stopping is not None else np.array([], dtype=str)
    else:
        msg = f"Unknown featurizer side {side!r}"
        raise ValueError(msg)
    return FeaturizerFitContext(
        unique_train_ids=unique_entity_ids(train_ids),
        pair_expanded_train_ids=np.asarray(train_ids, dtype=str),
        unique_early_stopping_ids=unique_entity_ids(es_ids) if es_ids.size else np.array([], dtype=str),
        pair_expanded_early_stopping_ids=np.asarray(es_ids, dtype=str) if es_ids.size else np.array([], dtype=str),
        side=side,
    )


def _entity_id_only_featurizer(featurizer: Featurizer | None) -> bool:
    return getattr(featurizer, "entity_id_only", False)


def _filter_response_batch(
    batch: ResponseBatch,
    valid_cl_ids: set[str] | None,
    valid_dr_ids: set[str] | None,
) -> ResponseBatch:
    """Keep only pairs whose entities are in the valid sets."""
    mask = np.ones(len(batch), dtype=bool)
    if valid_cl_ids is not None:
        mask &= np.array([cl in valid_cl_ids for cl in batch.cell_line_ids])
    if valid_dr_ids is not None:
        mask &= np.array([dr in valid_dr_ids for dr in batch.drug_ids])
    return ResponseBatch(
        response=batch.response[mask],
        cell_line_ids=batch.cell_line_ids[mask],
        drug_ids=batch.drug_ids[mask],
    )


def _filter_nan_entity_features(
    response: ResponseBatch,
    entity_ids: np.ndarray | None,
    matrix: np.ndarray | None,
    *,
    side: str,
) -> tuple[ResponseBatch, np.ndarray | None, np.ndarray | None]:
    """Remove entities with NaN features and filter response pairs accordingly."""
    if entity_ids is None or matrix is None or matrix.size == 0:
        return response, entity_ids, matrix
    valid = ~np.any(np.isnan(matrix), axis=1)
    if valid.all():
        return response, entity_ids, matrix
    valid_ids = set(entity_ids[valid])
    entity_ids = entity_ids[valid]
    matrix = matrix[valid]
    if side == "cell_line":
        pair_mask = np.array([cl in valid_ids for cl in response.cell_line_ids])
    else:
        pair_mask = np.array([dr in valid_ids for dr in response.drug_ids])
    response = ResponseBatch(
        response=response.response[pair_mask],
        cell_line_ids=response.cell_line_ids[pair_mask],
        drug_ids=response.drug_ids[pair_mask],
    )
    return response, entity_ids, matrix


def _instantiate_featurizer(
    config: FeaturizerConfig,
    resolved: ResolvedModelConfig,
) -> Featurizer:
    registry = str(config.registry)
    if config.name == "concatFeaturizers":
        children = [_instantiate_featurizer(child, resolved) for child in (config.featurizers or ())]
        return config.create_instance({"featurizers": children})
    selector = qualified_featurizer_selector(config.name, config.view)
    return config.create_instance(resolved.featurizer_values(registry, selector))


def build_component_stack(config: ModelConfig | ResolvedModelConfig) -> _ComponentStack:
    """Instantiate featurizers and predictor for a validated config.

    :param config: Template or resolved model configuration.
    :returns: Component stack ready for training.
    """
    from drevalpy.components.core.tuning.search_space import resolve_model_config

    resolved = config if isinstance(config, ResolvedModelConfig) else resolve_model_config(config)
    template = resolved.template
    cell_line = (
        _instantiate_featurizer(template.cell_line_featurizer, resolved)
        if template.cell_line_featurizer is not None
        else None
    )
    drug = _instantiate_featurizer(template.drug_featurizer, resolved) if template.drug_featurizer is not None else None
    predictor_hp: dict[str, Any] = {
        **resolved.predictor_values(),
        "prediction_mode": template.prediction_mode,
    }
    predictor = template.predictor.create_instance(predictor_hp)
    return _ComponentStack(
        cell_line,
        drug,
        predictor,
        prediction_mode=template.prediction_mode,
        resolved=resolved,
    )


class _ComponentStack:
    """Fit featurizers on training entities, then train a predictor on featurized pairs."""

    def __init__(
        self,
        cell_line_featurizer: Featurizer | None,
        drug_featurizer: Featurizer | None,
        predictor: Predictor,
        *,
        prediction_mode: PredictionMode = PredictionMode.REGRESSION,
        resolved: ResolvedModelConfig,
    ) -> None:
        self._cell_line_featurizer = cell_line_featurizer
        self._drug_featurizer = drug_featurizer
        self._predictor = predictor
        self._prediction_mode = prediction_mode
        self._resolved = ResolvedModelConfig.model_validate(resolved.model_dump(mode="python"))
        self._cell_line_matrix: np.ndarray | None = None
        self._drug_matrix: np.ndarray | None = None
        self._cell_line_entity_ids: np.ndarray | None = None
        self._drug_entity_ids: np.ndarray | None = None

    @property
    def config(self) -> ModelConfig:
        """Return the immutable template associated with this stack.

        :returns: Template ``ModelConfig``.
        """
        return self._resolved.template

    @property
    def resolved(self) -> ResolvedModelConfig:
        """Return the resolved instance configuration.

        :returns: ``ResolvedModelConfig``.
        """
        return self._resolved

    def _build_batch(
        self,
        response: ResponseBatch,
        *,
        cell_line_input: FeatureSource,
        drug_input: FeatureSource | None,
        cell_line_entity_ids: np.ndarray,
        drug_entity_ids: np.ndarray | None,
        cell_line_matrix: np.ndarray,
        drug_matrix: np.ndarray | None,
        output_earlystopping: ResponseBatch | None = None,
        training_context: TrainingContext | None = None,
    ) -> ModelInputBatch:
        cell_line_blocks: dict[str, FeatureBlock] = {}
        if self._cell_line_featurizer is not None:
            cell_line_blocks = self._cell_line_featurizer.transform_blocks(
                cell_line_input,
                cell_line_entity_ids,
            )

        drug_blocks: dict[str, FeatureBlock] = {}
        if self._drug_featurizer is not None and drug_entity_ids is not None:
            if drug_input is None and not _entity_id_only_featurizer(self._drug_featurizer):
                msg = "drug_input is required when a drug featurizer is configured"
                raise ValueError(msg)
            if drug_input is not None:
                drug_blocks = self._drug_featurizer.transform_blocks(drug_input, drug_entity_ids)

        return build_model_input_batch(
            response,
            cell_line_entity_ids=cell_line_entity_ids,
            drug_entity_ids=drug_entity_ids if self._drug_featurizer is not None else None,
            cell_line_features=cell_line_matrix,
            drug_features=drug_matrix if self._drug_featurizer is not None else None,
            cell_line_blocks=cell_line_blocks,
            drug_blocks=drug_blocks,
            early_stopping_response=output_earlystopping,
            training_context=training_context,
        )

    def _require_drug_input(self, drug_input: FeatureSource | None) -> FeatureSource | None:
        if drug_input is None and not _entity_id_only_featurizer(self._drug_featurizer):
            msg = "drug_input is required when a drug featurizer is configured"
            raise ValueError(msg)
        return drug_input

    def _fit_transform_featurizer(
        self,
        featurizer: Featurizer,
        source: FeatureSource | None,
        *,
        train_entity_ids: np.ndarray,
        entity_id_only_ids: np.ndarray,
        fit_context: FeaturizerFitContext | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        featurizer.fit(source, entity_ids=train_entity_ids, context=fit_context)
        if _entity_id_only_featurizer(featurizer):
            entity_ids = np.asarray(entity_id_only_ids, dtype=str)
        else:
            entity_ids = np.asarray(train_entity_ids, dtype=str)
        matrix = featurizer.transform(source, entity_ids)
        return entity_ids, matrix

    def _train_cell_line_side(
        self,
        output: ResponseBatch,
        cell_line_input: FeatureSource,
        *,
        output_earlystopping: ResponseBatch | None = None,
    ) -> None:
        if self._cell_line_featurizer is None:
            self._cell_line_entity_ids = np.array([], dtype=str)
            self._cell_line_matrix = np.empty((0, 0), dtype=np.float32)
            return
        train_cell_lines = unique_entity_ids(output.cell_line_ids)
        fit_context = _build_fit_context(
            output,
            early_stopping=output_earlystopping,
            side="cell_line",
        )
        entity_ids, matrix = self._fit_transform_featurizer(
            self._cell_line_featurizer,
            cell_line_input,
            train_entity_ids=train_cell_lines,
            entity_id_only_ids=train_cell_lines,
            fit_context=fit_context,
        )
        self._cell_line_entity_ids = entity_ids
        self._cell_line_matrix = matrix

    def _train_drug_side(
        self,
        output: ResponseBatch,
        drug_input: FeatureSource | None,
        *,
        output_earlystopping: ResponseBatch | None = None,
    ) -> None:
        if self._drug_featurizer is None:
            self._drug_entity_ids = np.array([], dtype=str)
            self._drug_matrix = np.empty((0, 0), dtype=np.float32)
            return
        train_drugs = unique_entity_ids(output.drug_ids)
        drug_source = self._require_drug_input(drug_input)
        fit_context = _build_fit_context(
            output,
            early_stopping=output_earlystopping,
            side="drug",
        )
        entity_ids, matrix = self._fit_transform_featurizer(
            self._drug_featurizer,
            drug_source,
            train_entity_ids=train_drugs,
            entity_id_only_ids=train_drugs,
            fit_context=fit_context,
        )
        self._drug_entity_ids = entity_ids
        self._drug_matrix = matrix

    def _filter_nan_features(
        self,
        output: ResponseBatch,
        output_earlystopping: ResponseBatch | None,
    ) -> tuple[ResponseBatch, ResponseBatch | None]:
        """Safety net: warn if NaN features remain after upfront filtering.

        In normal operation, the run() function pre-filters pairs to entities with
        available features, so this should be a no-op. If NaN still appears,
        it indicates a data integrity issue.
        """
        valid_cl_ids = self._purge_nan_rows("cell_line")
        valid_dr_ids = self._purge_nan_rows("drug")

        if valid_cl_ids is None and valid_dr_ids is None:
            return output, output_earlystopping

        from drevalpy.log import get_logger

        get_logger(__name__).warning(
            "NaN features detected after featurization — this indicates a data integrity issue. "
            "Filtering affected pairs."
        )

        output = _filter_response_batch(output, valid_cl_ids, valid_dr_ids)
        if output_earlystopping is not None:
            output_earlystopping = _filter_response_batch(output_earlystopping, valid_cl_ids, valid_dr_ids)
            if len(output_earlystopping) == 0:
                output_earlystopping = None

        return output, output_earlystopping

    def _purge_nan_rows(self, side: str) -> set[str] | None:
        """Remove NaN rows from entity matrix, return valid IDs or None if clean."""
        if side == "cell_line":
            matrix, ids = self._cell_line_matrix, self._cell_line_entity_ids
        else:
            matrix, ids = self._drug_matrix, self._drug_entity_ids

        if matrix is None or matrix.size == 0:
            return None
        valid_mask = ~np.any(np.isnan(matrix), axis=1)
        if valid_mask.all():
            return None

        if side == "cell_line":
            self._cell_line_entity_ids = ids[valid_mask]
            self._cell_line_matrix = matrix[valid_mask]
        else:
            self._drug_entity_ids = ids[valid_mask]
            self._drug_matrix = matrix[valid_mask]
        return set(ids[valid_mask])

    def _fit_featurizers_and_predictor(
        self,
        output: ResponseBatch,
        cell_line_input: FeatureSource,
        drug_input: FeatureSource | None = None,
        *,
        output_earlystopping: ResponseBatch | None = None,
        training_context: TrainingContext | None = None,
    ) -> _ComponentStack:
        """Fit featurizers on entity features and train the predictor on the batch.

        :param output: Training response pairs.
        :param cell_line_input: Cell-line feature source.
        :param drug_input: Drug feature source, or ``None``.
        :param output_earlystopping: Optional early-stopping response pairs.
        :param training_context: Optional runtime metadata.
        :returns: Self after training.
        """
        if len(output) == 0:
            return self

        self._train_cell_line_side(output, cell_line_input, output_earlystopping=output_earlystopping)
        self._train_drug_side(output, drug_input, output_earlystopping=output_earlystopping)

        output, output_earlystopping = self._filter_nan_features(output, output_earlystopping)
        if len(output) == 0:
            return self

        cell_line_entity_ids = (
            self._cell_line_entity_ids if self._cell_line_entity_ids is not None else np.array([], dtype=str)
        )
        cell_line_matrix = (
            self._cell_line_matrix if self._cell_line_matrix is not None else np.empty((0, 0), dtype=np.float32)
        )
        batch = self._build_batch(
            output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_entity_ids=cell_line_entity_ids,
            drug_entity_ids=self._drug_entity_ids,
            cell_line_matrix=cell_line_matrix,
            drug_matrix=self._drug_matrix,
            output_earlystopping=output_earlystopping,
            training_context=training_context,
        )
        self._predictor.fit(batch)

        return self

    def is_fitted(self) -> bool:
        """Return whether the predictor has fitted state.

        :returns: ``True`` when the predictor has been fitted.
        """
        return self._predictor.is_fitted()

    def component_state(self) -> dict[str, object]:
        """Return serializable state owned by the component stack.

        :returns: Mapping with predictor and featurizer state dicts.
        """
        return {
            "predictor": self._predictor.get_state(),
            "cell_line_featurizer": (
                self._cell_line_featurizer.get_state() if self._cell_line_featurizer is not None else {}
            ),
            "drug_featurizer": self._drug_featurizer.get_state() if self._drug_featurizer is not None else {},
        }

    def restore_component_state(self, state: dict[str, object]) -> None:
        """Restore state produced by ``component_state``.

        :param state: Serialized component state mapping.
        :raises ValueError: If predictor or featurizer state is not a mapping.
        """
        predictor_state = state.get("predictor", {})
        if not isinstance(predictor_state, dict):
            raise ValueError("predictor state is not a mapping")
        self._predictor.set_state(predictor_state)
        for key, featurizer in (
            ("cell_line_featurizer", self._cell_line_featurizer),
            ("drug_featurizer", self._drug_featurizer),
        ):
            value = state.get(key, {})
            if featurizer is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"{key} state is not a mapping")
                featurizer.set_state(value)

    def predict_from_features(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureSource,
        drug_input: FeatureSource | None = None,
    ) -> np.ndarray:
        """Predict using pre-built FeatureSource objects.

        :param cell_line_ids: Cell-line identifiers for each pair.
        :param drug_ids: Drug identifiers for each pair.
        :param cell_line_input: Cell-line feature source.
        :param drug_input: Drug feature source, or ``None``.
        :returns: Predicted response values.
        :raises RuntimeError: If the predictor has not been fitted.
        """
        if not self.is_fitted():
            msg = "Model has not been trained; call train() or load() before predict()"
            raise RuntimeError(msg)
        if len(cell_line_ids) == 0:
            return np.array([])

        response = ResponseBatch(
            response=np.zeros(len(cell_line_ids)),
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
        )

        cell_line_entity_ids, cell_line_matrix = self._transform_cell_line_features(cell_line_ids, cell_line_input)
        drug_entity_ids, drug_matrix = self._transform_drug_features(drug_ids, drug_input)

        response, cell_line_entity_ids, cell_line_matrix = _filter_nan_entity_features(
            response, cell_line_entity_ids, cell_line_matrix, side="cell_line"
        )
        response, drug_entity_ids, drug_matrix = _filter_nan_entity_features(
            response, drug_entity_ids, drug_matrix, side="drug"
        )

        if len(response) == 0:
            return np.array([])

        batch = self._build_batch(
            response,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_entity_ids=cell_line_entity_ids,
            drug_entity_ids=drug_entity_ids,
            cell_line_matrix=cell_line_matrix,
            drug_matrix=drug_matrix,
        )
        return self._predictor.predict(batch)

    def _transform_cell_line_features(
        self,
        cell_line_ids: np.ndarray,
        cell_line_input: FeatureSource,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform cell-line features for prediction."""
        if self._cell_line_featurizer is None:
            return np.array([], dtype=str), np.empty((0, 0), dtype=np.float32)
        entity_ids = unique_entity_ids(cell_line_ids)
        matrix = self._cell_line_featurizer.transform(cell_line_input, entity_ids)
        return entity_ids, matrix

    def _transform_drug_features(
        self,
        drug_ids: np.ndarray,
        drug_input: FeatureSource | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Transform drug features for prediction."""
        if self._drug_featurizer is None:
            return None, None
        entity_ids = unique_entity_ids(drug_ids)
        if drug_input is None and not _entity_id_only_featurizer(self._drug_featurizer):
            msg = "drug_input is required when a drug featurizer is configured"
            raise ValueError(msg)
        if drug_input is not None:
            matrix = self._drug_featurizer.transform(drug_input, entity_ids)
        else:
            matrix = np.empty((0, 0), dtype=np.float32)
        return entity_ids, matrix

    # ------------------------------------------------------------------
    # Dataset-backed API
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response_pairs(
        mudataset: Dataset,
        scope: SplitMask,
    ) -> ResponseBatch:
        """Build a ResponseBatch from the Dataset for given pair indices.

        :param mudataset: Source of response values.
        :param scope: SplitMask with 2D pair array.
        :returns: Flat ResponseBatch of (cell_line, drug, response) triples.
        """
        pairs = scope.pairs
        if len(pairs) == 0:
            return ResponseBatch(
                response=np.array([], dtype=np.float64),
                cell_line_ids=np.array([], dtype=str),
                drug_ids=np.array([], dtype=str),
            )

        cl_ids = mudataset.cell_line_ids
        drug_ids = mudataset.drug_ids
        response_matrix = mudataset.response_matrix

        cl_idx = pairs[:, 0]
        dr_idx = pairs[:, 1]
        responses = response_matrix[cl_idx, dr_idx]

        valid = ~np.isnan(responses)
        return ResponseBatch(
            response=responses[valid].astype(np.float64),
            cell_line_ids=cl_ids[cl_idx[valid]],
            drug_ids=drug_ids[dr_idx[valid]],
        )

    def _build_features_from_mudataset(
        self,
        mudataset: Dataset,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
    ) -> tuple[FeatureSource, FeatureSource | None]:
        """Construct FeatureSource adapters from Dataset for the relevant entities.

        :param mudataset: Source of feature data.
        :param cell_line_ids: Unique cell-line IDs needed.
        :param drug_ids: Unique drug IDs needed.
        :returns: Tuple of (cell_line_source, drug_source).
        """
        cl_source = CellLineFeatureSource(mudataset, cell_line_ids)
        drug_source = DrugFeatureSource(mudataset, drug_ids) if self._drug_featurizer is not None else None
        return cl_source, drug_source

    def train(
        self,
        mudataset: Dataset,
        scope: SplitMask,
        *,
        training_context: TrainingContext | None = None,
    ) -> _ComponentStack:
        """Train the component stack using a Dataset and SplitMask.

        Extracts response pairs and features from the Dataset, then fits
        featurizers and the predictor.

        :param mudataset: Source of response values and features.
        :param scope: Entity scope defining cell-line/drug indices to train on.
        :param training_context: Optional runtime metadata.
        :returns: Self after training.
        """
        output = self._extract_response_pairs(mudataset, scope)
        if len(output) == 0:
            return self

        output_earlystopping: ResponseBatch | None = None

        all_cl_ids = unique_entity_ids(
            np.concatenate(
                [
                    output.cell_line_ids,
                    output_earlystopping.cell_line_ids if output_earlystopping else np.array([], dtype=str),
                ]
            )
        )
        all_drug_ids = unique_entity_ids(
            np.concatenate(
                [
                    output.drug_ids,
                    output_earlystopping.drug_ids if output_earlystopping else np.array([], dtype=str),
                ]
            )
        )

        cell_line_input, drug_input = self._build_features_from_mudataset(mudataset, all_cl_ids, all_drug_ids)

        return self._fit_featurizers_and_predictor(
            output,
            cell_line_input,
            drug_input,
            output_earlystopping=output_earlystopping,
            training_context=training_context,
        )

    def train_with_early_stopping(
        self,
        mudataset: Dataset,
        scope: SplitMask,
        early_stopping_scope: SplitMask,
        *,
        training_context: TrainingContext | None = None,
    ) -> _ComponentStack:
        """Train with an explicit early-stopping scope.

        :param mudataset: Source of response values and features.
        :param scope: Entity scope defining cell-line/drug indices to train on.
        :param early_stopping_scope: Entity scope for early-stopping samples.
        :param training_context: Optional runtime metadata.
        :returns: Self after training.
        """
        output = self._extract_response_pairs(mudataset, scope)
        if len(output) == 0:
            return self

        output_earlystopping = self._extract_response_pairs(mudataset, early_stopping_scope)
        if len(output_earlystopping) == 0:
            output_earlystopping = None

        all_cl_ids = unique_entity_ids(
            np.concatenate(
                [
                    output.cell_line_ids,
                    output_earlystopping.cell_line_ids if output_earlystopping else np.array([], dtype=str),
                ]
            )
        )
        all_drug_ids = unique_entity_ids(
            np.concatenate(
                [
                    output.drug_ids,
                    output_earlystopping.drug_ids if output_earlystopping else np.array([], dtype=str),
                ]
            )
        )

        cell_line_input, drug_input = self._build_features_from_mudataset(mudataset, all_cl_ids, all_drug_ids)

        return self._fit_featurizers_and_predictor(
            output,
            cell_line_input,
            drug_input,
            output_earlystopping=output_earlystopping,
            training_context=training_context,
        )

    def predict(
        self,
        mudataset: Dataset,
        scope: SplitMask,
    ) -> np.ndarray:
        """Predict responses for the entities defined by an SplitMask.

        Returns one prediction per pair in scope. Pairs with missing features
        get NaN predictions (maintaining alignment with scope.pairs).

        :param mudataset: Source of feature data and entity IDs.
        :param scope: Entity scope with cell-line/drug indices for prediction.
        :returns: Predicted response values aligned to scope.pairs.
        :raises RuntimeError: If the predictor has not been fitted.
        """
        if not self.is_fitted():
            msg = "Model has not been trained; call train() or load() before predict()"
            raise RuntimeError(msg)

        test_response = self._extract_response_pairs(mudataset, scope)
        if len(test_response) == 0:
            return np.full(len(scope.pairs), np.nan)

        all_cl_ids = unique_entity_ids(test_response.cell_line_ids)
        all_drug_ids = unique_entity_ids(test_response.drug_ids)

        cell_line_input, drug_input = self._build_features_from_mudataset(mudataset, all_cl_ids, all_drug_ids)

        raw_predictions = self.predict_from_features(
            test_response.cell_line_ids,
            test_response.drug_ids,
            cell_line_input,
            drug_input,
        )

        # Align predictions back to scope.pairs (NaN for filtered pairs)
        if len(raw_predictions) == len(scope.pairs):
            return raw_predictions

        cl_ids = mudataset.cell_line_ids
        drug_ids = mudataset.drug_ids
        result = np.full(len(scope.pairs), np.nan)
        predicted_pairs = set(zip(test_response.cell_line_ids.tolist(), test_response.drug_ids.tolist(), strict=True))

        pred_idx = 0
        for i, (cl_i, dr_i) in enumerate(scope.pairs):
            pair_key = (cl_ids[cl_i], drug_ids[dr_i])
            if pair_key in predicted_pairs and pred_idx < len(raw_predictions):
                result[i] = raw_predictions[pred_idx]
                pred_idx += 1

        return result
