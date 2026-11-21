"""The public model runtime: one named recipe and its fitted featurizers, predictor and response transformation.

``DRPModel`` builds fresh components from its :class:`~drevalpy.models.spec.ModelSpec` on every
``train`` call and owns all fitted state. Related concerns live in their own modules:

* ``spec.py``: the frozen ``ModelSpec``, reused across folds and tuning trials;
* ``_feature_preparation.py``: featurizing the entities and dropping the pairs a component's
  missing-input policy rejects;
* ``response_transformation/``: the optional response transformation;
* ``_archive.py``: the versioned single-ZIP checkpoint;
* ``drevalpy/experiment/_tracking.py``: Weights & Biases tracking, outside ``DRPModel``.
"""

from __future__ import annotations

import tempfile
from contextlib import nullcontext
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath

from drevalpy._utils.accelerator import ensure_available
from drevalpy._utils.n_jobs import check_n_jobs
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.featurizers.featurizer import CONCAT
from drevalpy.models._feature_preparation import prepare_features, unique_entity_ids, usable_pairs
from drevalpy.models.config.immutable import thaw_value
from drevalpy.models.hyperparameters import PREDICTOR_SLOT, component_values
from drevalpy.models.response_transformation._pipeline_ops import (
    apply_response_transformation,
    fit_response_transformation,
    group_labels,
)
from drevalpy.models.spec import ModelSpec
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_block import FeaturizerOutput
from drevalpy.types.data.feature_source import FeatureSource, Side
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.data.response_table import ResponseTable
from drevalpy.types.data.selection_mask import SelectionMask

if TYPE_CHECKING:
    from collections.abc import Mapping

    from drevalpy.components.featurizers.featurizer import Featurizer
    from drevalpy.components.predictors.abstract.base import Predictor
    from drevalpy.models.config import FeaturizerConfig
    from drevalpy.models.response_transformation import ResponseTransformer


def validate_fit_scopes(dataset: Dataset, scope: SelectionMask, stopping: SelectionMask | None) -> None:
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
        if not isinstance(selection, SelectionMask):
            raise TypeError("Model scope must be a SelectionMask")
        if selection.shape != dataset.response_matrix.shape:
            raise ValueError("Model scope shape must match the dataset response matrix")
    if stopping is not None and np.any(scope.mask & stopping.mask):
        raise ValueError("Training and early-stopping scopes must be response-disjoint")


class ModelState(Enum):
    """Training outcome of a model."""

    UNTRAINED = "untrained"
    EMPTY = "empty"
    FITTED = "fitted"


def _build_featurizer(config: FeaturizerConfig, slot: str, values: Mapping[str, Any]) -> Featurizer:
    if config.name == CONCAT:
        children = [_build_featurizer(child, slot, values) for child in (config.featurizers or ())]
        return config.create_instance({"featurizers": children})
    return config.create_instance(component_values(values, slot, config.selector))


def _build_components(spec: ModelSpec) -> tuple[Featurizer | None, Featurizer | None, Predictor]:
    """Instantiate fresh featurizers and predictor with the spec's hyperparameters.

    :param spec: Model specification.
    :returns: Sample featurizer, optional drug featurizer and predictor.
    """
    config, values = spec.config, spec.hyperparameters
    roots = config.featurizers()
    sample, drug = (
        _build_featurizer(roots[side], f"{side}_featurizer", values) if side in roots else None
        for side in (Side.SAMPLE, Side.DRUG)
    )
    # A plain string: networks pass their hyperparameters to Lightning's ``save_hyperparameters``, and a checkpoint
    # holding the enum cannot be reloaded with ``torch.load(weights_only=True)``.
    predictor = config.predictor.create_instance(
        {
            **component_values(values, PREDICTOR_SLOT, config.predictor.name),
            "prediction_mode": str(config.prediction_mode.value),
        }
    )
    return sample, drug, predictor


class DRPModel:
    """Train, predict, and persist a model built from an immutable ``ModelSpec``.

    Predictions always have one value per requested response position, in the
    dataset's original response units. The model owns all fitted state, including
    feature preprocessing and response transformation.
    """

    def __init__(
        self,
        spec: ModelSpec,
        hyperparameters: Mapping[str, Any] | None = None,
        *,
        check_values: bool = True,
        response_transformation: ResponseTransformer | None = None,
    ) -> None:
        """Configure a fresh model without training any components.

        :param spec: Named preset or custom component configuration.
        :param hyperparameters: Optional overrides under short or qualified keys, applied with
            :meth:`ModelSpec.with_hyperparameters`.
        :param check_values: Passed to :meth:`ModelSpec.with_hyperparameters`; ``False`` accepts
            *hyperparameters* outside the search space.
        :param response_transformation: Cloneable, invertible response transformation prototype.
        :raises TypeError: If spec is not a ModelSpec.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError("DRPModel requires a ModelSpec; use ModelSpec.from_name(name) for a preset")
        self._spec = spec.with_hyperparameters(hyperparameters, check_values=check_values) if hyperparameters else spec
        self._response_transformation = response_transformation
        self._fitted_response_transformation: ResponseTransformer | None = None
        self._sample_featurizer: Featurizer | None = None
        self._drug_featurizer: Featurizer | None = None
        self._predictor: Predictor | None = None
        self._measure: str | None = None
        self.state = ModelState.UNTRAINED

    @property
    def spec(self) -> ModelSpec:
        """Unfitted specification, including the hyperparameter values, bound to this model."""
        return self._spec

    @property
    def name(self) -> str:
        """Public name used by results, checkpoints, and logging."""
        return self.spec.name

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Independent copy of the model's values under qualified keys."""
        return thaw_value(dict(self.spec.hyperparameters))

    def is_fitted(self) -> bool:
        """Return whether training or loading produced a fitted predictor.

        :returns: True only for a fitted model.
        """
        return self.state == ModelState.FITTED

    @property
    def measure(self) -> str | None:
        """The response measure the model was trained on, such as ``pEC50``; ``None`` before training."""
        return self._measure

    def _reset(self) -> None:
        self.state = ModelState.UNTRAINED
        self._measure = None
        self._fitted_response_transformation = None
        self._sample_featurizer = self._drug_featurizer = self._predictor = None
        self._sample_featurizer, self._drug_featurizer, self._predictor = _build_components(self.spec)

    def train(
        self,
        dataset: Dataset,
        scope: SelectionMask,
        *,
        early_stopping_scope: SelectionMask | None = None,
        model_checkpoint_dir: str | UPath | None = None,
        n_jobs: int = 1,
    ) -> None:
        """Start a fresh fit using training and optional disjoint stopping data.

        A model that requires an accelerator raises ``DrEvalError`` when torch finds none.

        :param dataset: Response matrix and raw features.
        :param scope: Training response positions.
        :param early_stopping_scope: Disjoint positions used only for stopping.
        :param model_checkpoint_dir: Directory for component training checkpoints. When omitted, a temporary
            directory is used for this fit and removed afterwards.
        :param n_jobs: Number of worker processes a single-drug predictor fits its per-drug models in; ``-1`` uses
            all CPUs. It changes the time a fit takes, not its result, and is ignored by every other model.
        """
        check_n_jobs(n_jobs)
        ensure_available(self.spec.accelerator, self.name)
        self._reset()
        directory = (
            tempfile.TemporaryDirectory(prefix="drevalpy-checkpoints-")
            if model_checkpoint_dir is None
            else nullcontext(model_checkpoint_dir)
        )
        with directory as checkpoint_dir:
            context = TrainingContext(checkpoint_dir=UPath(checkpoint_dir), n_jobs=n_jobs)
            self._fit(dataset, scope, early_stopping_scope, training_context=context)
        self._measure = dataset.measure

    def _fit(
        self,
        dataset: Dataset,
        scope: SelectionMask,
        early_stopping_scope: SelectionMask | None,
        *,
        training_context: TrainingContext,
    ) -> None:
        validate_fit_scopes(dataset, scope, early_stopping_scope)
        predictor = self._require_predictor()
        layers = predictor.response_layers
        raw = ResponseTable.from_dataset(dataset, scope, observed_only=True, layers=layers)
        if len(raw) == 0:
            self.state = ModelState.EMPTY
            return
        self._fitted_response_transformation = fit_response_transformation(
            self._response_transformation, raw.response, group_labels(self._response_transformation, dataset, raw)
        )
        output = self._transform_responses(dataset, raw)
        stopping = (
            self._transform_responses(
                dataset, ResponseTable.from_dataset(dataset, early_stopping_scope, observed_only=True, layers=layers)
            )
            if early_stopping_scope is not None
            else None
        )
        sample_input, drug_input = self._feature_sources(dataset)
        output, stopping = usable_pairs(
            output,
            stopping,
            (("sample_ids", self._sample_featurizer, sample_input), ("drug_ids", self._drug_featurizer, drug_input)),
        )
        if len(output) == 0:
            self.state = ModelState.EMPTY
            return
        data = self._predictor_input(
            output,
            sample_input,
            drug_input,
            fitting=True,
            early_stopping=stopping,
            training_context=training_context,
        )
        if not predictor._valid_pair_mask(data).any():
            self.state = ModelState.EMPTY
            return
        predictor.fit(data)
        self.state = ModelState.FITTED

    def _require_predictor(self) -> Predictor:
        if self._predictor is None:
            raise RuntimeError("Model has not been trained; call train() or load() before predict()")
        return self._predictor

    def _transform_responses(self, dataset: Dataset, response: ResponseTable) -> ResponseTable:
        """Apply the fitted response transformation to *response*, leaving layers untouched."""
        transformation = self._fitted_response_transformation
        return replace(
            response,
            response=apply_response_transformation(
                transformation, response.response, groups=group_labels(transformation, dataset, response)
            ),
        )

    def _feature_sources(self, dataset: Dataset) -> tuple[FeatureSource, FeatureSource | None]:
        sample_source = FeatureSource(dataset, Side.SAMPLE)
        drug_source = FeatureSource(dataset, Side.DRUG) if self._drug_featurizer is not None else None
        return sample_source, drug_source

    def _predictor_input(
        self,
        response: ResponseTable,
        sample_input: FeatureSource,
        drug_input: FeatureSource | None,
        *,
        fitting: bool = False,
        early_stopping: ResponseTable | None = None,
        training_context: TrainingContext | None = None,
    ) -> PredictorInput:
        results = []
        for attr, featurizer, source in (
            ("sample_ids", self._sample_featurizer, sample_input),
            ("drug_ids", self._drug_featurizer, drug_input),
        ):
            ids = getattr(response, attr)
            es_ids = getattr(early_stopping, attr) if early_stopping is not None else None
            all_ids = np.concatenate([ids, es_ids]) if es_ids is not None else ids
            if featurizer is None or source is None:
                results.append(FeaturizerOutput(unique_entity_ids(all_ids), {}))
                continue
            results.append(
                prepare_features(
                    featurizer,
                    source,
                    all_ids,
                    train_pair_ids=ids if fitting else None,
                    es_pair_ids=es_ids if fitting else None,
                )
            )
        sample, drug = results
        return PredictorInput(
            response,
            sample=sample,
            drug=drug if self._drug_featurizer is not None else None,
            early_stopping=early_stopping,
            training_context=training_context or TrainingContext(),
        )

    def predict(self, dataset: Dataset, scope: SelectionMask) -> np.ndarray:
        """Predict original-unit responses in scope order, preserving missing positions.

        Pairs with missing features get NaN predictions. A model that requires an accelerator raises
        ``DrEvalError`` when torch finds none.

        :param dataset: Response matrix and raw features.
        :param scope: Requested response positions, including unobserved responses.
        :returns: One prediction or NaN per position.
        :raises RuntimeError: If the model has not been trained or loaded.
        :raises ValueError: If the predictor returns anything but one value per pair.
        """
        ensure_available(self.spec.accelerator, self.name)
        if self.state == ModelState.EMPTY:
            return np.full(len(scope), np.nan)
        if not self.is_fitted():
            raise RuntimeError("Model has not been trained; call train() or load() before predict()")
        predictor = self._require_predictor()
        validate_fit_scopes(dataset, scope, None)
        response = ResponseTable.from_dataset(dataset, scope)
        sample_input, drug_input = self._feature_sources(dataset)
        if len(response) == 0:
            return np.empty(0, dtype=float)
        predictions = np.asarray(predictor.predict(self._predictor_input(response, sample_input, drug_input)))
        if predictions.shape != (len(response),):
            raise ValueError(
                f"Predictor {type(predictor).__name__} returned shape {predictions.shape}; "
                f"expected one value per requested response position ({len(response)},)"
            )
        return apply_response_transformation(
            self._fitted_response_transformation,
            predictions,
            groups=group_labels(self._fitted_response_transformation, dataset, response),
            inverse=True,
        )

    def save(self, path: str | UPath) -> None:
        """Save the spec and the fitted components as one versioned archive.

        :param path: Checkpoint path; .zip is appended when needed.
        :raises RuntimeError: If the model is not fitted.
        """
        from drevalpy.models._archive import save_archive

        if not self.is_fitted():
            raise RuntimeError(f"Cannot save model {self.name!r}: it must be trained first")
        save_archive(
            path,
            self.spec.dump(),
            {
                "sample_featurizer": self._sample_featurizer,
                "drug_featurizer": self._drug_featurizer,
                "predictor": self._predictor,
                "response_transformation": self._response_transformation,
                "fitted_response_transformation": self._fitted_response_transformation,
            },
            measure=self._measure,
        )

    @classmethod
    def load(cls, path: str | UPath) -> DRPModel:
        """Restore a complete fitted model, including its response transformation.

        Custom featurizers and predictors must be registered as for training.

        .. warning::
           Checkpoints are unpickled with ``joblib``, which can execute arbitrary code. Load
           checkpoints only from trusted sources.

        :param path: Checkpoint archive path.
        :returns: The fitted model.
        :raises CorruptedCheckpointError: If the archive's spec or components are invalid.
        """
        from drevalpy.models._archive import CorruptedCheckpointError, load_archive

        spec_data, components, measure = load_archive(path)
        try:
            spec = ModelSpec.load(spec_data)
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptedCheckpointError(f"checkpoint spec is invalid: {exc}") from exc
        if components.get("predictor") is None:
            raise CorruptedCheckpointError("checkpoint has no fitted predictor")
        model = cls(spec, response_transformation=components.get("response_transformation"))
        model._sample_featurizer = components.get("sample_featurizer")
        model._drug_featurizer = components.get("drug_featurizer")
        model._predictor = components["predictor"]
        model._fitted_response_transformation = components.get("fitted_response_transformation")
        model._measure = measure
        model.state = ModelState.FITTED
        return model
