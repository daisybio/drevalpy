"""Private featurizer/predictor pipeline for DRPModel.

``DRPModel`` keeps ``train``, ``predict``, ``save`` and ``load`` thin; each concern has its own
module, and new behaviour goes there rather than back into ``DRPModel``:

* ``spec.py``: the frozen ``ModelSpec``, reused across folds and tuning trials, so it holds no
  fitted state;
* this module: :class:`_ModelPipeline`, which owns all fitted state and the ``PipelineState``;
* ``_components.py``: building fresh featurizers and the predictor from a ``ResolvedModelConfig``;
* ``_feature_preparation.py``: featurizing the entities and dropping the pairs a component's
  missing-input policy rejects;
* ``response_transformation/``: the optional response transformation;
* ``mixins/_persistence_io.py``: the versioned single-ZIP checkpoint, which imports ``DRPModel``
  only under ``TYPE_CHECKING``;
* ``factory.py``: resolving zoo and spec names to ``ModelConfig`` objects;
* ``drevalpy/experiment/_tracking.py``: Weights & Biases tracking, outside ``DRPModel``.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.featurizers._matrix import unique_entity_ids
from drevalpy.models._components import build_components
from drevalpy.models._feature_preparation import transform_entities, usable_responses
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.models.response_transformation._pipeline_ops import (
    apply_response_transformation,
    fit_response_transformation,
    group_labels,
)
from drevalpy.types import SplitMask
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource, FeatureSource, SampleFeatureSource

if TYPE_CHECKING:
    from drevalpy.models.response_transformation import ResponseTransformer


def validate_fit_scopes(dataset: Dataset, scope: SplitMask, stopping: SplitMask | None) -> None:
    """Check dataset-backed selections without imposing split-mode entity rules.

    :param dataset: Dataset that *scope* and *stopping* index into.
    :param scope: Training or prediction positions.
    :param stopping: Optional disjoint stopping positions.
    :raises TypeError: If inputs do not meet the public model contract.
    :raises ValueError: If scope shapes differ or fitting selections overlap.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError("Model operations require a Dataset")
    for selection in (scope,) if stopping is None else (scope, stopping):
        if not isinstance(selection, SplitMask):
            raise TypeError("Model scope must be a SplitMask")
        if selection.shape != dataset.response_matrix.shape:
            raise ValueError("Model scope shape must match the dataset response matrix")
    if stopping is not None and np.any(scope.mask & stopping.mask):
        raise ValueError("Training and early-stopping scopes must be response-disjoint")


class PipelineState(Enum):
    """Training outcome owned by the pipeline."""

    UNTRAINED = "untrained"
    EMPTY = "empty"
    FITTED = "fitted"


class _ModelPipeline:
    """Fit featurizers on training entities, then train a predictor on featurized pairs."""

    def __init__(
        self, resolved: ResolvedModelConfig, response_transformation: ResponseTransformer | None = None
    ) -> None:
        self._resolved = resolved
        self._response_transformation = response_transformation
        self._fitted_response_transformation = None
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
        self._fitted_response_transformation = None
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
        return ModelInputBatch.from_response(
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
            "response_transformation": self._response_transformation,
            "fitted_response_transformation": self._fitted_response_transformation,
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
        if "fitted_response_transformation" not in state or "response_transformation" not in state:
            raise ValueError("checkpoint is missing response transformation state; retrain the model")
        self._response_transformation = state["response_transformation"]
        self._fitted_response_transformation = state["fitted_response_transformation"]
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
        dataset: Dataset,
        scope: SplitMask,
        response_transformation: ResponseTransformer | None = None,
    ) -> ResponseBatch:
        """Build a ResponseBatch from the Dataset for given pair indices.

        :param dataset: Source of response values.
        :param scope: SplitMask with 2D pair array.
        :param response_transformation: Optional *already fitted* transformer applied to
            the extracted responses. Pass it only for training-time extractions; the
            prediction and evaluation paths must read the raw matrix.
        :returns: Flat ResponseBatch of (sample, drug, response) triples.
        """
        response = ResponseBatch.from_dataset(dataset, scope, observed_only=True)
        return ResponseBatch(
            apply_response_transformation(
                response_transformation,
                response.response,
                groups=group_labels(response_transformation, dataset, response),
            ),
            response.sample_ids,
            response.drug_ids,
        )

    def _build_features_from_dataset(
        self,
        dataset: Dataset,
        sample_ids: np.ndarray,
        drug_ids: np.ndarray,
    ) -> tuple[FeatureSource, FeatureSource | None]:
        """Construct FeatureSource adapters from Dataset for the relevant entities.

        :param dataset: Source of feature data.
        :param sample_ids: Unique sample IDs needed.
        :param drug_ids: Unique drug IDs needed.
        :returns: Tuple of (sample_source, drug_source).
        """
        cl_source = SampleFeatureSource(dataset, sample_ids)
        drug_source = DrugFeatureSource(dataset, drug_ids) if self._drug_featurizer is not None else None
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
        self._fitted_response_transformation = fit_response_transformation(
            self._response_transformation, raw.response, group_labels(self._response_transformation, dataset, raw)
        )
        response_transformation = self._fitted_response_transformation
        output = self._extract_response_pairs(dataset, scope, response_transformation)
        output_earlystopping = (
            self._extract_response_pairs(dataset, early_stopping_scope, response_transformation)
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

        sample_input, drug_input = self._build_features_from_dataset(dataset, all_cl_ids, all_drug_ids)

        self._fit_featurizers_and_predictor(
            output,
            sample_input,
            drug_input,
            output_earlystopping=output_earlystopping,
            training_context=training_context,
        )

    def predict(
        self,
        dataset: Dataset,
        scope: SplitMask,
    ) -> np.ndarray:
        """Predict responses for the entities defined by an SplitMask.

        Returns one prediction per pair in scope. Pairs with missing features
        get NaN predictions (maintaining alignment with scope.pairs).

        :param dataset: Source of feature data and entity IDs.
        :param scope: Entity scope with sample/drug indices for prediction.
        :returns: Predicted response values aligned to scope.pairs.
        :raises RuntimeError: If the predictor has not been fitted.
        :raises ValueError: If the predictor returns anything but one value per pair.
        """
        if self.state == PipelineState.EMPTY:
            return np.full(len(scope), np.nan)
        if not self.is_fitted():
            raise RuntimeError("Model has not been trained; call train() or load() before predict()")
        validate_fit_scopes(dataset, scope, None)
        response = ResponseBatch.from_dataset(dataset, scope)
        sample_input, drug_input = self._build_features_from_dataset(
            dataset,
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
        return apply_response_transformation(
            self._fitted_response_transformation,
            predictions,
            groups=group_labels(self._fitted_response_transformation, dataset, response),
            inverse=True,
        )
