"""Component-native DRPModel facade used by factory entries and construct_model."""

from __future__ import annotations

import copy
from typing import Any, ClassVar

from drevalpy.components.data_loading import (
    load_cell_line_features_for_model_config,
    load_drug_features_for_model_config,
)
from drevalpy.components.registry import get_predictor
from drevalpy.components.training_context import TrainingContext
from drevalpy.components.tuning.drp_hyperparameters import (
    config_from_public_hyperparameters,
    default_config_for_drp_model,
    default_hyperparameters_for_drp_model,
    public_hyperparameters_from_config,
    structured_space_for_drp_model,
)
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.composed_model import ComposedModel
from drevalpy.models.config import ModelConfig, ModelScope
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.featurizer_mapping import cell_line_views_from_model_config, drug_views_from_model_config


class NativeDRPModel(DRPModel):
    """Thin DRP facade with a single component stack as fitted-state source of truth."""

    _factory_name: ClassVar[str]
    _model_spec: ClassVar[str | None] = None

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Materialize the component stack from defaults or flat hyperparameters.

        :param hyperparameters: flat public overrides; ``None`` uses class defaults.
        """
        self._init_drp_base()
        self._init_runtime_fields()
        if hyperparameters is None:
            self._apply_model_config(self.default_model_config())
            return
        config = config_from_public_hyperparameters(type(self), hyperparameters)
        if config is None:
            self._apply_model_config(model_config_for_name(self._factory_name, hyperparameters))
            return
        self._apply_model_config(config)

    def _init_drp_base(self) -> None:
        """Initialize DRPModel fields without assigning the read-only hyperparameters property."""
        self.wandb_project: str | None = None
        self.wandb_run: Any = None
        self.wandb_config: dict[str, Any] | None = None
        self._in_hyperparameter_tuning: bool = False

    def _init_runtime_fields(self) -> None:
        self._composed: ComposedModel | None = None
        self._empty_training = False
        self._hyperparameters: dict[str, Any] = {}
        self._resolved_model_config: ModelConfig | None = None
        self._cell_line_views_list: list[str] = []
        self._drug_views_list: list[str] = []
        self._engine_preload_state: dict[str, Any] = {}

    @classmethod
    def _unmaterialized(cls) -> NativeDRPModel:
        """Return an empty instance without materializing a default stack."""
        instance = object.__new__(cls)
        instance._init_drp_base()
        instance._init_runtime_fields()
        return instance

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> NativeDRPModel:
        """Construct a facade from a resolved structured ``ModelConfig``."""
        instance = cls._unmaterialized()
        instance._apply_model_config(config)
        return instance

    @classmethod
    def get_model_name(cls) -> str:
        return cls._factory_name

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, Any]:
        return default_hyperparameters_for_drp_model(cls)

    @classmethod
    def get_structured_hyperparameter_space(cls) -> dict[str, Any]:
        return structured_space_for_drp_model(cls)

    @classmethod
    def default_model_config(cls) -> ModelConfig:
        config = default_config_for_drp_model(cls)
        if config is not None:
            return config
        if cls._model_spec is not None:
            return ModelConfig.from_spec(cls._model_spec)
        return model_config_for_name(cls._factory_name)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Return a defensive copy of the facade hyperparameters."""
        return copy.deepcopy(self._hyperparameters)

    @property
    def cell_line_views(self) -> list[str]:
        return list(self._cell_line_views_list)

    @property
    def drug_views(self) -> list[str]:
        return list(self._drug_views_list)

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        """Store a copy of hyperparameters and optionally log them to wandb."""
        import wandb

        self._hyperparameters = copy.deepcopy(hyperparameters)
        if not self.is_wandb_enabled():
            return
        if not self._in_hyperparameter_tuning:
            wandb.config.update({"hyperparameters": self._hyperparameters})

    def _resolved_config(self, hyperparameters: dict[str, Any] | None = None) -> ModelConfig:
        if self._resolved_model_config is not None and hyperparameters is None:
            return self._resolved_model_config.model_copy(deep=True)
        config = config_from_public_hyperparameters(type(self), hyperparameters)
        if config is not None:
            return config
        if self._model_spec is not None:
            return ModelConfig.from_spec(self._model_spec)
        return model_config_for_name(self._factory_name, hyperparameters or {})

    def _apply_model_config(self, config: ModelConfig) -> None:
        self._resolved_model_config = config.model_copy(deep=True)
        self.log_hyperparameters(public_hyperparameters_from_config(self._resolved_model_config))
        self._cell_line_views_list = cell_line_views_from_model_config(self._resolved_model_config)
        self._drug_views_list = drug_views_from_model_config(self._resolved_model_config)
        self.is_single_drug_model = self._resolved_model_config.scope == ModelScope.SINGLE_DRUG
        predictor_class = get_predictor(self._resolved_model_config.predictor.name)
        self.early_stopping = bool(getattr(predictor_class, "supports_early_stopping", False))
        self._composed = self._resolved_model_config.create_model()
        self._empty_training = False

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        # Prefer the already-resolved config; re-applying public flat HPs can reject
        # featurizer-local keys that were round-tripped into ``self.hyperparameters``.
        config = self._resolved_config()
        predictor_class = get_predictor(config.predictor.name)
        loader = getattr(predictor_class, "load_dataset_cell_line_features", None)
        if callable(loader):
            loaded = loader(
                data_path,
                dataset_name,
                hyperparameters=self.hyperparameters,
                model_name=self.get_model_name(),
            )
            if isinstance(loaded, tuple):
                features, preload = loaded
            else:
                features, preload = loaded, {}
            if isinstance(preload, dict):
                self._engine_preload_state.update(preload)
            return features
        return load_cell_line_features_for_model_config(
            config,
            data_path,
            dataset_name,
            model_name=self.get_model_name(),
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        config = self._resolved_config()
        predictor_class = get_predictor(config.predictor.name)
        loader = getattr(predictor_class, "load_dataset_drug_features", None)
        if callable(loader):
            loaded = loader(
                data_path,
                dataset_name,
                hyperparameters=self.hyperparameters,
                model_name=self.get_model_name(),
            )
            if isinstance(loaded, tuple):
                features, preload = loaded
            else:
                features, preload = loaded, {}
            if isinstance(preload, dict):
                self._engine_preload_state.update(preload)
            return features
        return load_drug_features_for_model_config(
            config,
            data_path,
            dataset_name,
            model_name=self.get_model_name(),
        )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        if self._composed is None:
            msg = "Model has not been constructed with a component stack"
            raise RuntimeError(msg)
        if len(output) == 0:
            self._empty_training = True
            return
        composed = self._composed
        self._empty_training = False
        set_preload = getattr(composed._predictor, "set_engine_preload_state", None)
        if callable(set_preload) and self._engine_preload_state:
            set_preload(self._engine_preload_state)
        composed.train(
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
        cell_line_ids,
        drug_ids,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ):
        if self._empty_training:
            import numpy as np

            return np.full(len(cell_line_ids), np.nan)
        if self._composed is None:
            msg = "Model has not been constructed with a component stack"
            raise RuntimeError(msg)
        if not self._composed.is_fitted():
            msg = "Model has not been trained; call train() or load() before predict()"
            raise RuntimeError(msg)
        return self._composed.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)

    def save(self, directory: str) -> None:
        if self._composed is None or not self._composed.is_fitted():
            msg = "Cannot save: component stack is not trained"
            raise RuntimeError(msg)
        self._composed.save(directory)

    @classmethod
    def load(cls, directory: str) -> NativeDRPModel:
        instance = cls._unmaterialized()
        composed = ComposedModel.load(directory)
        config = composed.config
        if config is None:
            raise RuntimeError("Loaded component stack did not contain a ModelConfig")
        instance._resolved_model_config = config
        instance._hyperparameters = public_hyperparameters_from_config(config)
        instance._cell_line_views_list = cell_line_views_from_model_config(config)
        instance._drug_views_list = drug_views_from_model_config(config)
        instance.is_single_drug_model = config.scope == ModelScope.SINGLE_DRUG
        predictor_class = get_predictor(config.predictor.name)
        instance.early_stopping = bool(getattr(predictor_class, "supports_early_stopping", False))
        instance._composed = composed
        instance._empty_training = False
        return instance


def create_native_drp_class(
    factory_name: str,
    *,
    spec: str | None = None,
    class_name: str | None = None,
    scope: ModelScope = ModelScope.MULTI_DRUG,
    validate_spec: bool = True,
    bases: tuple[type[NativeDRPModel], ...] = (NativeDRPModel,),
    class_dict: dict[str, Any] | None = None,
) -> type[NativeDRPModel]:
    """Create a component-native DRPModel subclass for a factory entry."""
    if spec is not None and validate_spec:
        from drevalpy.models.config import ModelConfig

        base_config = ModelConfig.from_spec(spec)
        base_config.validate()
    attrs: dict[str, Any] = {
        "_factory_name": factory_name,
        "_model_spec": spec,
        "is_single_drug_model": scope == ModelScope.SINGLE_DRUG,
    }
    if class_dict:
        attrs.update(class_dict)

    def get_model_name(cls) -> str:
        return factory_name

    attrs["get_model_name"] = classmethod(get_model_name)
    cls = type(class_name or factory_name, bases, attrs)
    cls.__module__ = "drevalpy.models"
    return cls
