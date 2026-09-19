"""Private featurizer/predictor execution stack for DRPModel."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.featurizers._featurizer_label import qualified_featurizer_selector
from drevalpy.components.featurizers._matrix import unique_entity_ids
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictionMode
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.models.tuning.search_space import resolve_model_config
from drevalpy.types import SplitMask
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.model_input_build import build_model_input_batch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import CellLineFeatureSource, DrugFeatureSource, FeatureSource

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


def _entity_id_only_featurizer(featurizer: Featurizer | None) -> bool:
    return getattr(featurizer, "entity_id_only", False)


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


class StackState(Enum):
    """Training outcome owned by the component stack."""

    UNTRAINED = "untrained"
    EMPTY = "empty"
    FITTED = "fitted"


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
        self.state = StackState.UNTRAINED

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

    def _reset(self) -> None:
        self.state = StackState.UNTRAINED
        fresh = build_component_stack(self.resolved)
        self._cell_line_featurizer = fresh._cell_line_featurizer
        self._drug_featurizer = fresh._drug_featurizer
        self._predictor = fresh._predictor

    @staticmethod
    def _features(
        featurizer: Featurizer | None,
        source: FeatureSource | None,
        ids: np.ndarray,
        *,
        train_ids: np.ndarray | None = None,
        es_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, FeaturizerResult]:
        if featurizer is None:
            return np.array([], dtype=str), FeaturizerResult(np.empty((0, 0), dtype=np.float32), {})
        if source is None and not _entity_id_only_featurizer(featurizer):
            raise ValueError("Feature source is required when a featurizer is configured")
        if train_ids is not None:
            featurizer.fit(
                source,
                entity_ids=unique_entity_ids(train_ids),
                pair_expanded_ids=train_ids,
                pair_expanded_es_ids=es_ids,
            )
        ids = unique_entity_ids(ids)
        return ids, featurizer.transform_features(source, ids)

    def _batch(
        self,
        response: ResponseBatch,
        cell_line_input: FeatureSource,
        drug_input: FeatureSource | None,
        *,
        fitting: bool = False,
        output_earlystopping: ResponseBatch | None = None,
        training_context: TrainingContext | None = None,
    ) -> ModelInputBatch:
        results = []
        for attr, featurizer, source in (
            ("cell_line_ids", self._cell_line_featurizer, cell_line_input),
            ("drug_ids", self._drug_featurizer, drug_input),
        ):
            ids = getattr(response, attr)
            es_ids = getattr(output_earlystopping, attr) if output_earlystopping is not None else None
            all_ids = np.concatenate([ids, es_ids]) if es_ids is not None else ids
            results.append(
                self._features(featurizer, source, all_ids, train_ids=ids if fitting else None, es_ids=es_ids)
            )
        (cl_ids, cl), (drug_ids, drug) = results
        return build_model_input_batch(
            response,
            cell_line_entity_ids=cl_ids,
            drug_entity_ids=drug_ids if self._drug_featurizer is not None else None,
            cell_line_features=cl.matrix,
            drug_features=drug.matrix if self._drug_featurizer is not None else None,
            cell_line_blocks=cl.blocks,
            drug_blocks=drug.blocks,
            early_stopping_response=output_earlystopping,
            training_context=training_context,
        )

    def _usable_responses(
        self,
        response: ResponseBatch,
        cell_line_input: FeatureSource,
        drug_input: FeatureSource | None,
    ) -> ResponseBatch:
        keep = ~np.isnan(response.response)
        for ids, featurizer, source in (
            (response.cell_line_ids, self._cell_line_featurizer, cell_line_input),
            (response.drug_ids, self._drug_featurizer, drug_input),
        ):
            if featurizer is not None:
                keep &= featurizer._detect_valid(source, ids)
        return ResponseBatch(response.response[keep], response.cell_line_ids[keep], response.drug_ids[keep])

    def _fit_featurizers_and_predictor(
        self,
        output: ResponseBatch,
        cell_line_input: FeatureSource,
        drug_input: FeatureSource | None = None,
        *,
        output_earlystopping: ResponseBatch | None = None,
        training_context: TrainingContext | None = None,
    ) -> _ComponentStack:
        """Fit fresh components using only usable training responses.

        :param output: Training responses.
        :param cell_line_input: Cell-line feature source.
        :param drug_input: Optional drug feature source.
        :param output_earlystopping: Disjoint stopping responses.
        :param training_context: Checkpoint and logging context.
        :returns: This stack in EMPTY or FITTED state; failures leave it UNTRAINED.
        """
        self._reset()
        output = self._usable_responses(output, cell_line_input, drug_input)
        if len(output) == 0:
            self.state = StackState.EMPTY
            return self
        if output_earlystopping is not None:
            output_earlystopping = self._usable_responses(output_earlystopping, cell_line_input, drug_input)
        batch = self._batch(
            output,
            cell_line_input,
            drug_input,
            fitting=True,
            output_earlystopping=output_earlystopping,
            training_context=training_context,
        )
        if not self._predictor._valid_pair_mask(batch).any():
            self.state = StackState.EMPTY
            return self
        self._predictor.fit(batch)
        self.state = StackState.FITTED
        return self

    def is_fitted(self) -> bool:
        """Return whether training or checkpoint restoration completed successfully.

        :returns: True only for a fitted stack.
        """
        return self.state == StackState.FITTED

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
        self.state = StackState.UNTRAINED
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

        if self._predictor.is_fitted():
            self.state = StackState.FITTED

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
        if self.state == StackState.EMPTY:
            return np.full(len(cell_line_ids), np.nan)
        if not self.is_fitted():
            raise RuntimeError("Model has not been trained; call train() or load() before predict()")
        if len(cell_line_ids) == 0:
            return np.array([])
        response = ResponseBatch(np.zeros(len(cell_line_ids)), cell_line_ids, drug_ids)
        batch = self._batch(response, cell_line_input, drug_input)
        predictions = np.asarray(self._predictor.predict(batch))
        if predictions.shape != (len(response),):
            raise ValueError("Predictor must return one value per requested response position")
        return predictions

    # ------------------------------------------------------------------
    # Dataset-backed API
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response_pairs(
        mudataset: Dataset,
        scope: SplitMask,
        response_transformation: TransformerMixin | None = None,
    ) -> ResponseBatch:
        """Build a ResponseBatch from the Dataset for given pair indices.

        :param mudataset: Source of response values.
        :param scope: SplitMask with 2D pair array.
        :param response_transformation: Optional *already fitted* transformer applied to
            the extracted responses. Pass it only for training-time extractions; the
            prediction and evaluation paths must read the raw matrix.
        :returns: Flat ResponseBatch of (cell_line, drug, response) triples.
        """
        return ResponseBatch.from_dataset(mudataset, scope, observed_only=True, transformation=response_transformation)

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
        response_transformation: TransformerMixin | None = None,
    ) -> _ComponentStack:
        """Train the component stack using a Dataset and SplitMask.

        Extracts response pairs and features from the Dataset, then fits
        featurizers and the predictor.

        :param mudataset: Source of response values and features.
        :param scope: Entity scope defining cell-line/drug indices to train on.
        :param training_context: Optional runtime metadata.
        :param response_transformation: Optional fitted transformer applied to the
            training targets.
        :returns: Self after training.
        """
        return self.train_with_early_stopping(
            mudataset,
            scope,
            None,
            training_context=training_context,
            response_transformation=response_transformation,
        )

    def train_with_early_stopping(
        self,
        mudataset: Dataset,
        scope: SplitMask,
        early_stopping_scope: SplitMask | None,
        *,
        training_context: TrainingContext | None = None,
        response_transformation: TransformerMixin | None = None,
    ) -> _ComponentStack:
        """Train with an explicit early-stopping scope.

        :param mudataset: Source of response values and features.
        :param scope: Entity scope defining cell-line/drug indices to train on.
        :param early_stopping_scope: Entity scope for early-stopping samples.
        :param training_context: Optional runtime metadata.
        :param response_transformation: Optional fitted transformer applied to the
            training targets and to the early-stopping targets, which are training-time
            supervision and must live in the same space.
        :returns: Self after training.
        """
        self._reset()
        output = self._extract_response_pairs(mudataset, scope, response_transformation)
        output_earlystopping = (
            self._extract_response_pairs(mudataset, early_stopping_scope, response_transformation)
            if early_stopping_scope is not None
            else None
        )

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
        response = ResponseBatch.from_dataset(mudataset, scope)
        cell_line_input, drug_input = self._build_features_from_mudataset(
            mudataset,
            unique_entity_ids(response.cell_line_ids),
            unique_entity_ids(response.drug_ids),
        )
        return self.predict_from_features(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
