"""Component-native DRPModel facade used by factory entries and construct_model."""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy.components.data_loading import (
    load_cell_line_features_for_model_config,
    load_drug_features_for_model_config,
)
from drevalpy.components.register_builtins import ensure_components_registered
from drevalpy.components.tuning.drp_hyperparameters import (
    config_from_public_hyperparameters,
    default_config_for_drp_model,
    default_hyperparameters_for_drp_model,
    public_hyperparameters_from_config,
    structured_space_for_drp_model,
)
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._component_bridge import (
    ComponentDRPBridge,
    save_component_stack,
)
from drevalpy.models._legacy_checkpoint_loaders import (
    has_component_stack,
    load_hyperparameters_json,
    load_native_checkpoint,
)
from drevalpy.models.config import ModelConfig
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.featurizer_mapping import cell_line_views_from_model_config, drug_views_from_model_config


class NativeDRPModel(DRPModel):
    """Thin DRP facade with a single component stack as fitted-state source of truth."""

    _factory_name: ClassVar[str]
    _model_spec: ClassVar[str | None] = None

    def __init__(self) -> None:
        super().__init__()
        self._bridge = ComponentDRPBridge()
        self.hyperparameters: dict[str, Any] = {}
        self._resolved_model_config: ModelConfig | None = None
        self._cell_line_views_list: list[str] = []
        self._drug_views_list: list[str] = []

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
    def cell_line_views(self) -> list[str]:
        return list(self._cell_line_views_list)

    @cell_line_views.setter
    def cell_line_views(self, views: list[str]) -> None:
        self._cell_line_views_list = list(views)

    @property
    def drug_views(self) -> list[str]:
        return list(self._drug_views_list)

    @drug_views.setter
    def drug_views(self, views: list[str]) -> None:
        self._drug_views_list = list(views)

    def _resolved_config(self, hyperparameters: dict[str, Any] | None = None) -> ModelConfig:
        if self._resolved_model_config is not None and hyperparameters is None:
            return self._resolved_model_config.model_copy(deep=True)
        config = config_from_public_hyperparameters(type(self), hyperparameters)
        if config is not None:
            return config
        if self._model_spec is not None:
            return ModelConfig.from_spec(self._model_spec)
        return model_config_for_name(self._factory_name, hyperparameters or {})

    def build_from_model_config(self, config: ModelConfig) -> None:
        ensure_components_registered()
        self._resolved_model_config = config.model_copy(deep=True)
        self.hyperparameters = public_hyperparameters_from_config(self._resolved_model_config)
        self.log_hyperparameters(self.hyperparameters)
        self._cell_line_views_list = cell_line_views_from_model_config(self._resolved_model_config)
        self._drug_views_list = drug_views_from_model_config(self._resolved_model_config)
        self._bridge.set_composed_config(self._resolved_model_config)

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        ensure_components_registered()
        config = config_from_public_hyperparameters(type(self), hyperparameters)
        if config is None:
            self.log_hyperparameters(hyperparameters)
            self.hyperparameters = dict(hyperparameters)
            self.build_from_model_config(model_config_for_name(self._factory_name, hyperparameters))
            return
        self.build_from_model_config(config)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        config = self._resolved_config(self.hyperparameters or None)
        return load_cell_line_features_for_model_config(
            config,
            data_path,
            dataset_name,
            model_name=self.get_model_name(),
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        config = self._resolved_config(self.hyperparameters or None)
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
        _ = model_checkpoint_dir
        if self._bridge.composed is None:
            self.build_from_model_config(self.default_model_config())
        self._bridge.train(
            output,
            cell_line_input,
            drug_input,
            output_earlystopping=output_earlystopping,
        )

    def predict(
        self,
        cell_line_ids,
        drug_ids,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ):
        if self._bridge.composed is None:
            msg = "Model has not been built; call build_model() before predict()"
            raise RuntimeError(msg)
        if not self._bridge.is_trained():
            msg = "Model has not been trained; call train() or load() before predict()"
            raise RuntimeError(msg)
        return self._bridge.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)

    def save(self, directory: str) -> None:
        if not self._bridge.is_trained():
            msg = "Cannot save: component stack is not trained"
            raise RuntimeError(msg)
        config = self._resolved_model_config or self._resolved_config(self.hyperparameters or None)
        save_component_stack(
            self._bridge,
            directory,
            hyperparameters=public_hyperparameters_from_config(config),
        )

    @classmethod
    def load(cls, directory: str) -> NativeDRPModel:
        instance = cls()
        if has_component_stack(directory):
            loaded_hp = load_native_checkpoint(instance, directory)
            instance.hyperparameters = loaded_hp or load_hyperparameters_json(directory)
            return instance
        return cls._load_legacy_checkpoint(directory)

    @classmethod
    def _load_legacy_checkpoint(cls, directory: str) -> NativeDRPModel:
        msg = f"{cls.__name__} does not support legacy checkpoint loading"
        raise NotImplementedError(msg)


def create_native_drp_class(
    factory_name: str,
    *,
    spec: str | None = None,
    bases: tuple[type[NativeDRPModel], ...] = (NativeDRPModel,),
    class_dict: dict[str, Any] | None = None,
) -> type[NativeDRPModel]:
    """Create a component-native DRPModel subclass for a factory entry."""
    if spec is not None:
        from drevalpy.models.config import ModelConfig

        base_config = ModelConfig.from_spec(spec)
        base_config.validate()
    attrs: dict[str, Any] = {
        "_factory_name": factory_name,
        "_model_spec": spec,
    }
    if class_dict:
        attrs.update(class_dict)

    def get_model_name(cls) -> str:
        return factory_name

    attrs["get_model_name"] = classmethod(get_model_name)
    cls = type(factory_name, bases, attrs)
    cls.__module__ = bases[0].__module__
    return cls
