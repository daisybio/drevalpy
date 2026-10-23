"""Private featurizer/predictor pipeline for DRPModel."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.featurizers._matrix import unique_entity_ids
from drevalpy.models._components import build_components
from drevalpy.models._feature_preparation import transform_entities, usable_responses
from drevalpy.models._target_transform import (
    apply_target_transform,
    fit_target_transform,
    group_labels,
    validate_fit_scopes,
)
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.types import SplitMask
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.model_input_build import build_model_input_batch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource, FeatureSource, SampleFeatureSource

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


class PipelineState(Enum):
    """Training outcome owned by the pipeline."""

    UNTRAINED = "untrained"
    EMPTY = "empty"
    FITTED = "fitted"


class _ModelPipeline:
    """Fit featurizers on training entities, then train a predictor on featurized pairs."""

    def __init__(self, resolved: ResolvedModelConfig, response_transformation: Any = None) -> None:
        self._resolved = resolved
        self._response_prototype = response_transformation
        self._response_transform = None
        self._sample_featurizer = None
        self._drug_featurizer = None
        self._predictor = None
        self.state = PipelineState.UNTRAINED

    @property
    def resolved(self) -> ResolvedModelConfig:
        """Concrete configuration owned by this pipeline."""
        return self._resolved

    def reset(self) -> None:
        """Invalidate previous state before constructing components for one fit."""
        self.state = PipelineState.UNTRAINED
        self._response_transform = None
        self._sample_featurizer = self._drug_featurizer = self._predictor = None
        self._sample_featurizer, self._drug_featurizer, self._predictor = build_components(self.resolved)

    def _batch(
        self,
        response: ResponseBatch,
        sample_input: FeatureSource,
        drug_input: FeatureSource | None,
        *,
        fitting: bool = False,
        output_earlystopping: ResponseBatch | None = None,
        training_context: TrainingContext | None = None,
    ) -> ModelInputBatch:
        results = []
        for attr, featurizer, source in (
            ("sample_ids", self._sample_featurizer, sample_input),
            ("drug_ids", self._drug_featurizer, drug_input),
        ):
            ids = getattr(response, attr)
            es_ids = getattr(output_earlystopping, attr) if output_earlystopping is not None else None
            all_ids = np.concatenate([ids, es_ids]) if es_ids is not None else ids
            results.append(
                transform_entities(featurizer, source, all_ids, train_ids=ids if fitting else None, es_ids=es_ids)
            )
        cl, drug = results
        return build_model_input_batch(
            response,
            sample=cl,
            drug=drug if self._drug_featurizer is not None else None,
            early_stopping_response=output_earlystopping,
            training_context=training_context,
        )

    def _fit_featurizers_and_predictor(
        self,
        output: ResponseBatch,
        sample_input: FeatureSource,
        drug_input: FeatureSource | None = None,
        *,
        output_earlystopping: ResponseBatch | None = None,
        training_context: TrainingContext | None = None,
    ) -> _ModelPipeline:
        """Fit fresh components using only usable training responses.

        :param output: Training responses.
        :param sample_input: Sample feature source.
        :param drug_input: Optional drug feature source.
        :param output_earlystopping: Disjoint stopping responses.
        :param training_context: Checkpoint and logging context.
        :returns: This pipeline in EMPTY or FITTED state; failures leave it UNTRAINED.
        """
        output = usable_responses(output, sample_input, drug_input, self._sample_featurizer, self._drug_featurizer)
        if len(output) == 0:
            self.state = PipelineState.EMPTY
            return self
        if output_earlystopping is not None:
            output_earlystopping = usable_responses(
                output_earlystopping, sample_input, drug_input, self._sample_featurizer, self._drug_featurizer
            )
        batch = self._batch(
            output,
            sample_input,
            drug_input,
            fitting=True,
            output_earlystopping=output_earlystopping,
            training_context=training_context,
        )
        if not self._predictor._valid_pair_mask(batch).any():
            self.state = PipelineState.EMPTY
            return self
        self._predictor.fit(batch)
        self.state = PipelineState.FITTED
        return self

    def is_fitted(self) -> bool:
        """Return whether training or checkpoint restoration completed successfully.

        :returns: True only for a fitted pipeline.
        """
        return self.state == PipelineState.FITTED

    def component_state(self) -> dict[str, object]:
        """Return serializable state owned by the pipeline.

        :returns: Mapping with predictor and featurizer state dicts.
        """
        return {
            "response_prototype": self._response_prototype,
            "response_transform": self._response_transform,
            "predictor": self._predictor.get_state(),
            "sample_featurizer": (self._sample_featurizer.get_state() if self._sample_featurizer is not None else {}),
            "drug_featurizer": self._drug_featurizer.get_state() if self._drug_featurizer is not None else {},
        }

    def restore_component_state(self, state: dict[str, object]) -> None:
        """Restore state produced by ``component_state``.

        :param state: Serialized component state mapping.
        :raises ValueError: If predictor or featurizer state is not a mapping.
        """
        self.reset()
        if "response_transform" not in state or "response_prototype" not in state:
            raise ValueError("checkpoint is missing target transformation state; retrain the model")
        self._response_prototype = state["response_prototype"]
        self._response_transform = state["response_transform"]
        predictor_state = state.get("predictor", {})
        if not isinstance(predictor_state, dict):
            raise ValueError("predictor state is not a mapping")
        self._predictor.set_state(predictor_state)
        for key, featurizer in (
            ("sample_featurizer", self._sample_featurizer),
            ("drug_featurizer", self._drug_featurizer),
        ):
            value = state.get(key, {})
            if featurizer is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"{key} state is not a mapping")
                featurizer.set_state(value)

        if self._predictor.is_fitted():
            self.state = PipelineState.FITTED

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
        :returns: Flat ResponseBatch of (sample, drug, response) triples.
        """
        response = ResponseBatch.from_dataset(mudataset, scope, observed_only=True)
        return ResponseBatch(
            apply_target_transform(
                response_transformation,
                response.response,
                groups=group_labels(response_transformation, mudataset, response),
            ),
            response.sample_ids,
            response.drug_ids,
        )

    def _build_features_from_mudataset(
        self,
        mudataset: Dataset,
        sample_ids: np.ndarray,
        drug_ids: np.ndarray,
    ) -> tuple[FeatureSource, FeatureSource | None]:
        """Construct FeatureSource adapters from Dataset for the relevant entities.

        :param mudataset: Source of feature data.
        :param sample_ids: Unique sample IDs needed.
        :param drug_ids: Unique drug IDs needed.
        :returns: Tuple of (sample_source, drug_source).
        """
        cl_source = SampleFeatureSource(mudataset, sample_ids)
        drug_source = DrugFeatureSource(mudataset, drug_ids) if self._drug_featurizer is not None else None
        return cl_source, drug_source

    def fit(
        self,
        dataset: Dataset,
        scope: SplitMask,
        early_stopping_scope: SplitMask | None,
        *,
        training_context: TrainingContext,
    ) -> None:
        """Fit the complete pipeline after the public training call resets it."""
        validate_fit_scopes(dataset, scope, early_stopping_scope)
        raw = ResponseBatch.from_dataset(dataset, scope, observed_only=True)
        if len(raw) == 0:
            self.state = PipelineState.EMPTY
            return
        self._response_transform = fit_target_transform(
            self._response_prototype, raw.response, group_labels(self._response_prototype, dataset, raw)
        )
        mudataset = dataset
        response_transformation = self._response_transform
        output = self._extract_response_pairs(mudataset, scope, response_transformation)
        output_earlystopping = (
            self._extract_response_pairs(mudataset, early_stopping_scope, response_transformation)
            if early_stopping_scope is not None
            else None
        )

        all_cl_ids = unique_entity_ids(
            np.concatenate(
                [
                    output.sample_ids,
                    output_earlystopping.sample_ids if output_earlystopping else np.array([], dtype=str),
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

        sample_input, drug_input = self._build_features_from_mudataset(mudataset, all_cl_ids, all_drug_ids)

        self._fit_featurizers_and_predictor(
            output,
            sample_input,
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
        :param scope: Entity scope with sample/drug indices for prediction.
        :returns: Predicted response values aligned to scope.pairs.
        :raises RuntimeError: If the predictor has not been fitted.
        :raises ValueError: If the predictor returns anything but one value per pair.
        """
        if self.state == PipelineState.EMPTY:
            return np.full(len(scope), np.nan)
        if not self.is_fitted():
            raise RuntimeError("Model has not been trained; call train() or load() before predict()")
        validate_fit_scopes(mudataset, scope, None)
        response = ResponseBatch.from_dataset(mudataset, scope)
        sample_input, drug_input = self._build_features_from_mudataset(
            mudataset,
            unique_entity_ids(response.sample_ids),
            unique_entity_ids(response.drug_ids),
        )
        if len(response) == 0:
            return np.empty(0, dtype=float)
        batch = self._batch(response, sample_input, drug_input)
        predictions = np.asarray(self._predictor.predict(batch))
        if predictions.shape != (len(response),):
            raise ValueError(
                f"Predictor {type(self._predictor).__name__} returned shape {predictions.shape}; "
                f"expected one value per requested response position ({len(response)},)"
            )
        return apply_target_transform(
            self._response_transform,
            predictions,
            groups=group_labels(self._response_transform, mudataset, response),
            inverse=True,
        )
