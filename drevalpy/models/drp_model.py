"""Concrete config-backed drug response prediction model."""

from __future__ import annotations

import copy
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.data_loading import (
    load_cell_line_features_for_model_config,
    load_drug_features_for_model_config,
)
from drevalpy.components.registry import get_predictor
from drevalpy.components.training_context import TrainingContext
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._component_stack import _ComponentStack, build_component_stack
from drevalpy.models._drp_logging import _DRPLoggingMixin
from drevalpy.models.config import ModelConfig, ModelScope
from drevalpy.models.featurizer_mapping import cell_line_views_from_model_config, drug_views_from_model_config
from drevalpy.utils._pipeline_function import pipeline_function


class DRPModel(_DRPLoggingMixin):
    """Concrete experiment-facing model backed by an immutable ModelConfig."""

    _model_name: ClassVar[str] = "DRPModel"
    _base_model_config: ClassVar[ModelConfig | None] = None

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Materialize a fresh component stack from class defaults or flat overrides.

        :param hyperparameters: optional flat public overrides; ``None`` uses defaults
        """
        self._init_runtime_fields()
        from drevalpy.components.tuning.drp_hyperparameters import (
            config_from_public_hyperparameters,
            default_config_for_drp_model,
        )

        if hyperparameters is None:
            config = default_config_for_drp_model(type(self))
            self._apply_model_config(config if config is not None else self.model_config())
            return
        config = config_from_public_hyperparameters(type(self), hyperparameters)
        if config is None:
            msg = f"Cannot apply hyperparameters for model {self.get_model_name()!r}"
            raise ValueError(msg)
        self._apply_model_config(config)

    def _init_runtime_fields(self) -> None:
        self.wandb_project: str | None = None
        self.wandb_run: Any = None
        self.wandb_config: dict[str, Any] | None = None
        self._in_hyperparameter_tuning = False
        self._stack: _ComponentStack | None = None
        self._empty_training = False
        self._hyperparameters: dict[str, Any] = {}
        self._resolved_model_config: ModelConfig | None = None

    @classmethod
    def _unmaterialized(cls) -> DRPModel:
        """Return an empty instance without materializing a default stack."""
        instance = object.__new__(cls)
        instance._init_runtime_fields()
        return instance

    @classmethod
    def _from_resolved_config(cls, config: ModelConfig) -> DRPModel:
        """Construct an instance from an already-resolved structured config."""
        instance = cls._unmaterialized()
        instance._apply_model_config(config)
        return instance

    @classmethod
    @pipeline_function
    def get_model_name(cls) -> str:
        """Return the model identity for this class."""
        return cls._model_name

    @classmethod
    def model_config(cls) -> ModelConfig:
        """Return a defensive deep copy of the class base config."""
        if cls._base_model_config is None:
            msg = f"{cls.__name__} has no base ModelConfig; use construct_model(...)"
            raise RuntimeError(msg)
        return cls._base_model_config.model_copy(deep=True)

    @classmethod
    def supports_early_stopping(cls) -> bool:
        """Return whether the configured predictor supports early stopping."""
        predictor_class = get_predictor(cls.model_config().predictor.name)
        return bool(getattr(predictor_class, "supports_early_stopping", False))

    @classmethod
    def is_single_drug(cls) -> bool:
        """Return whether this model is scoped to a single drug."""
        return cls.model_config().scope == ModelScope.SINGLE_DRUG

    @classmethod
    @pipeline_function
    def get_structured_hyperparameter_space(cls) -> dict[str, Any]:
        """Return the merged structured hyperparameter space for this model."""
        from drevalpy.components.tuning.drp_hyperparameters import structured_space_for_drp_model

        return structured_space_for_drp_model(cls)

    @classmethod
    @pipeline_function
    def get_default_hyperparameters(cls) -> dict[str, Any]:
        """Return default hyperparameters used by ``cls()``."""
        from drevalpy.components.tuning.drp_hyperparameters import default_hyperparameters_for_drp_model

        return default_hyperparameters_for_drp_model(cls)

    @classmethod
    @pipeline_function
    def get_hyperparameter_set(cls) -> list[dict[str, Any]]:
        """Return the default hyperparameter configuration for this model."""
        return [cls.get_default_hyperparameters()]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Return a defensive copy of the instance hyperparameters."""
        return copy.deepcopy(self._hyperparameters)

    @property
    def early_stopping(self) -> bool:
        """Instance convenience accessor for early-stopping support."""
        return type(self).supports_early_stopping()

    @property
    def is_single_drug_model(self) -> bool:
        """Instance convenience accessor for single-drug scope."""
        return type(self).is_single_drug()

    @property
    def cell_line_views(self) -> list[str]:
        """Return required cell-line views derived from the resolved config."""
        config = self._resolved_model_config or self.model_config()
        return cell_line_views_from_model_config(config)

    @property
    def drug_views(self) -> list[str]:
        """Return required drug views derived from the resolved config."""
        config = self._resolved_model_config or self.model_config()
        return drug_views_from_model_config(config)

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        """Store a copy of hyperparameters and optionally log them to wandb."""
        import wandb

        self._hyperparameters = copy.deepcopy(hyperparameters)
        if not self.is_wandb_enabled():
            return
        if not self._in_hyperparameter_tuning:
            wandb.config.update({"hyperparameters": self._hyperparameters})

    def _apply_model_config(self, config: ModelConfig) -> None:
        from drevalpy.components.tuning.drp_hyperparameters import public_hyperparameters_from_config

        self._resolved_model_config = config.model_copy(deep=True)
        self.log_hyperparameters(public_hyperparameters_from_config(self._resolved_model_config))
        self._stack = build_component_stack(self._resolved_model_config)
        self._empty_training = False

    @pipeline_function
    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load cell-line features for the resolved model config."""
        config = self._resolved_model_config
        if config is None:
            raise RuntimeError("Model has not been constructed with a ModelConfig")
        return load_cell_line_features_for_model_config(
            config,
            data_path,
            dataset_name,
            model_name=self.get_model_name(),
        )

    @pipeline_function
    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """Load drug features for the resolved model config."""
        config = self._resolved_model_config
        if config is None:
            raise RuntimeError("Model has not been constructed with a ModelConfig")
        return load_drug_features_for_model_config(
            config,
            data_path,
            dataset_name,
            model_name=self.get_model_name(),
        )

    @pipeline_function
    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Train the private component stack on the given response data."""
        if self._stack is None:
            raise RuntimeError("Model has not been constructed with a component stack")
        if len(output) == 0:
            self._empty_training = True
            return
        self._empty_training = False
        self._stack.train(
            output,
            cell_line_input,
            drug_input,
            output_earlystopping=output_earlystopping,
            training_context=TrainingContext(
                checkpoint_dir=model_checkpoint_dir,
                logging_metadata={"model_name": self.get_model_name()},
            ),
        )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Predict responses for the given cell-line/drug pairs."""
        if self._empty_training:
            return np.full(len(cell_line_ids), np.nan)
        if self._stack is None:
            raise RuntimeError("Model has not been constructed with a component stack")
        if not self._stack.is_fitted():
            raise RuntimeError("Model has not been trained; call train() or load() before predict()")
        return self._stack.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)

    @pipeline_function
    def save(self, directory: str) -> None:
        """Persist model identity, config, and fitted component state."""
        from drevalpy.models._model_persistence import save_model

        save_model(self, directory)

    @classmethod
    def load(cls, directory: str) -> DRPModel:
        """Load a fitted model checkpoint into a new instance of this class."""
        from drevalpy.models._model_persistence import (
            CorruptedCheckpointError,
            IncompatibleModelCheckpointError,
            load_model_payload,
        )

        model_name, config, state = load_model_payload(directory)
        if model_name != cls.get_model_name():
            raise IncompatibleModelCheckpointError(
                f"checkpoint model_name {model_name!r} does not match {cls.get_model_name()!r}"
            )
        instance = cls._from_resolved_config(config)
        if instance._stack is None:
            raise CorruptedCheckpointError("failed to materialize component stack from checkpoint")
        try:
            instance._stack.restore_component_state(state)
        except (ValueError, RuntimeError) as exc:
            raise CorruptedCheckpointError(
                f"checkpoint component state is invalid: {exc}" if str(exc) else "checkpoint component state is invalid"
            ) from exc
        if not instance._stack.is_fitted():
            raise CorruptedCheckpointError("checkpoint did not restore a fitted predictor")
        instance._empty_training = False
        return instance

    def get_concatenated_features(
        self,
        cell_line_view: str | None,
        drug_view: str | None,
        cell_line_ids_output: np.ndarray,
        drug_ids_output: np.ndarray,
        cell_line_input: FeatureDataset | None,
        drug_input: FeatureDataset | None,
    ) -> np.ndarray:
        """Concatenate selected cell-line and drug feature views into matrix ``X``."""
        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids_output,
            drug_ids=drug_ids_output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        if drug_view is not None and drug_view not in inputs:
            raise ValueError(f"Expected drug_view '{drug_view}' to be in inputs, but it was not. Inputs: {inputs}")
        if cell_line_view is not None and cell_line_view not in inputs:
            raise ValueError(
                f"Expected cell_line_view '{cell_line_view}' to be in inputs, but it was not. Inputs: {inputs}"
            )

        cell_line_features = None if cell_line_view is None else inputs.get(cell_line_view)
        drug_features = None if drug_view is None else inputs.get(drug_view)

        if cell_line_features is not None and drug_features is not None:
            return np.concatenate((cell_line_features, drug_features), axis=1)
        if cell_line_features is not None:
            return cell_line_features
        if drug_features is not None:
            return drug_features
        raise ValueError("No features provided.")

    def get_feature_matrices(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset | None,
        drug_input: FeatureDataset | None,
    ) -> dict[str, np.ndarray]:
        """Return feature matrices for the model's required views."""
        cell_line_feature_matrices = {}
        if cell_line_input is not None:
            for cell_line_view in self.cell_line_views:
                if cell_line_view not in cell_line_input.view_names:
                    raise ValueError(f"Cell line input does not contain view {cell_line_view}")
                cell_line_feature_matrices[cell_line_view] = cell_line_input.get_feature_matrix(
                    view=cell_line_view, identifiers=cell_line_ids
                )
        drug_feature_matrices = {}
        if drug_input is not None:
            for drug_view in self.drug_views:
                if drug_view not in drug_input.view_names:
                    raise ValueError(f"Drug input does not contain view {drug_view}")
                drug_feature_matrices[drug_view] = drug_input.get_feature_matrix(view=drug_view, identifiers=drug_ids)
        return {**cell_line_feature_matrices, **drug_feature_matrices}
