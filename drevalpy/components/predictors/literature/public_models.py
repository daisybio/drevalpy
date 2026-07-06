"""Public DRPModel compatibility classes backed by the component stack."""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar, cast

import numpy as np

from drevalpy.components.predictors.literature.impl.dipk.dipk import DIPKModel as _DIPKModel
from drevalpy.components.predictors.literature.impl.druggnn.drug_gnn import DrugGNN as _DrugGNN
from drevalpy.components.predictors.literature.impl.molir.molir import MOLIR as _MOLIR
from drevalpy.components.predictors.literature.impl.pharmaformer.pharmaformer import (
    PharmaFormerModel as _PharmaFormerModel,
)
from drevalpy.components.predictors.literature.impl.precily.precily import PrecilyModel as _PrecilyImpl
from drevalpy.components.predictors.literature.impl.simple_neural_network.multi_view_neural_network import (
    MultiViewNeuralNetwork as _MultiViewNeuralNetwork,
)
from drevalpy.components.predictors.literature.impl.simple_neural_network.simple_neural_network import (
    SimpleNeuralNetwork as _SimpleNeuralNetwork,
)
from drevalpy.components.predictors.literature.impl.sparsego.sparsego import SparseGOModel as _SparseGOModel
from drevalpy.components.predictors.literature.impl.srmf.srmf import SRMF as _SRMFImpl  # noqa: N811
from drevalpy.components.predictors.literature.impl.superfeltr.superfeltr import SuperFELTR as _SuperFELTR
from drevalpy.components.predictors.literature.structured_predictors import StructuredLiteratureEnginePredictor
from drevalpy.components.register_builtins import ensure_components_registered
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._component_bridge import (
    _HYPERPARAMETERS_FILE,
    ComponentDRPBridge,
    restore_literature_to_components,
    save_component_stack,
)
from drevalpy.models._legacy_checkpoint_loaders import has_component_stack, load_native_checkpoint
from drevalpy.models._legacy_state_accessors import (
    literature_input_dims_from_bridge,
    literature_model_from_bridge,
)
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.factory import model_config_for_name

_SKIP_IMPL_STATE_COPY = frozenset(
    {
        "model",
        "gene_expression_scaler",
        "methylation_scaler",
        "methylation_pca",
        "proteomics_transformer",
        "input_dims",
    }
)


class LiteratureComponentDRPModel(DRPModel):
    """Route legacy experiment APIs through zoo-backed ComposedModel stacks."""

    _zoo_name: ClassVar[str]
    _impl_cls: ClassVar[type[Any] | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        impl_cls = cls.__dict__.get("_impl_cls")
        if impl_cls is not None:
            if "early_stopping" not in cls.__dict__:
                cls.early_stopping = getattr(impl_cls, "early_stopping", False)
            if "is_single_drug_model" not in cls.__dict__:
                cls.is_single_drug_model = getattr(impl_cls, "is_single_drug_model", False)

    def __init__(self) -> None:
        super().__init__()
        self._bridge = ComponentDRPBridge()
        self._preview_model: Any | None = None
        self._preview_featurizer: dict[str, Any] = {}
        self._input_dims: dict[str, int] = {}
        impl_cls = type(self)._impl_cls
        if impl_cls is not None:
            temp = impl_cls()
            for key, value in temp.__dict__.items():
                if key != "_bridge" and key not in _SKIP_IMPL_STATE_COPY:
                    setattr(self, key, value)
        if not hasattr(self, "hyperparameters"):
            self.hyperparameters = {}
        if impl_cls is not None:
            class_cell_line_views = getattr(impl_cls, "cell_line_views", [])
            if isinstance(class_cell_line_views, list) and class_cell_line_views:
                self._cell_line_views_list = list(class_cell_line_views)
            class_drug_views = getattr(impl_cls, "drug_views", [])
            if isinstance(class_drug_views, list) and class_drug_views:
                self._drug_views_list = list(class_drug_views)
            else:
                self._drug_views_list = []
        else:
            self._cell_line_views_list = []
            self._drug_views_list = []

    @classmethod
    def get_model_name(cls) -> str:
        return cls._zoo_name

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, Any]:
        if cls._impl_cls is not None:
            return cls._impl_cls.get_default_hyperparameters()
        return super().get_default_hyperparameters()

    @classmethod
    def get_hyperparameter_set(cls) -> list[dict[str, Any]]:
        return [cls.get_default_hyperparameters()]

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

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = dict(hyperparameters)
        ensure_components_registered()
        config = model_config_for_name(self._zoo_name, hyperparameters)
        self._bridge.set_composed_config(config)
        impl_cls = type(self)._impl_cls
        if impl_cls is not None:
            impl_cls.build_model(self, hyperparameters)
        self._publish_build_context()

    def _publish_build_context(self) -> None:
        composed = self._bridge.composed
        if composed is None:
            return
        predictor = composed._predictor
        if isinstance(predictor, StructuredLiteratureEnginePredictor):
            context = {
                name: value
                for name, value in vars(self).items()
                if not name.startswith("_") and name not in {"hyperparameters", "wandb_project"}
            }
            context["hyperparameters"] = dict(self.hyperparameters)
            predictor.set_build_context(context)

    @property
    def model(self):
        fitted = literature_model_from_bridge(self._bridge)
        if fitted is not None:
            return fitted
        return self._preview_model

    @model.setter
    def model(self, value: Any) -> None:
        self._preview_model = value

    @property
    def gene_expression_scaler(self):
        from drevalpy.models._legacy_state_accessors import sklearn_featurizer_state_from_bridge

        fitted = sklearn_featurizer_state_from_bridge(self._bridge).get("gene_expression_scaler")
        if fitted is not None:
            return fitted
        return self._preview_featurizer.get("gene_expression_scaler")

    @gene_expression_scaler.setter
    def gene_expression_scaler(self, value: Any) -> None:
        self._preview_featurizer["gene_expression_scaler"] = value

    @property
    def methylation_scaler(self):
        from drevalpy.models._legacy_state_accessors import sklearn_featurizer_state_from_bridge

        fitted = sklearn_featurizer_state_from_bridge(self._bridge).get("methylation_scaler")
        if fitted is not None:
            return fitted
        return self._preview_featurizer.get("methylation_scaler")

    @methylation_scaler.setter
    def methylation_scaler(self, value: Any) -> None:
        self._preview_featurizer["methylation_scaler"] = value

    @property
    def methylation_pca(self):
        from drevalpy.models._legacy_state_accessors import sklearn_featurizer_state_from_bridge

        fitted = sklearn_featurizer_state_from_bridge(self._bridge).get("methylation_pca")
        if fitted is not None:
            return fitted
        return self._preview_featurizer.get("methylation_pca")

    @methylation_pca.setter
    def methylation_pca(self, value: Any) -> None:
        self._preview_featurizer["methylation_pca"] = value

    @property
    def input_dims(self):
        dims = literature_input_dims_from_bridge(self._bridge)
        if dims is not None:
            return dims
        return getattr(self, "_input_dims", {})

    @input_dims.setter
    def input_dims(self, value: dict[str, int]) -> None:
        self._input_dims = dict(value)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        impl_cls = type(self)._impl_cls
        if impl_cls is None:
            msg = f"{type(self).__name__} does not implement load_cell_line_features"
            raise NotImplementedError(msg)
        return impl_cls.load_cell_line_features(self, data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        impl_cls = type(self)._impl_cls
        if impl_cls is None:
            msg = f"{type(self).__name__} does not implement load_drug_features"
            raise NotImplementedError(msg)
        return impl_cls.load_drug_features(self, data_path, dataset_name)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        _ = model_checkpoint_dir
        self._publish_build_context()
        self._bridge.train(
            output,
            cell_line_input,
            drug_input,
            output_earlystopping=output_earlystopping,
        )

    def _predict_via_impl(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        impl_cls = type(self)._impl_cls
        if impl_cls is None:
            return np.full(len(cell_line_ids), np.nan)
        return impl_cls.predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        if self._bridge.is_trained():
            return self._bridge.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)
        if getattr(self, "model", None) is not None:
            return self._predict_via_impl(cell_line_ids, drug_ids, cell_line_input, drug_input)
        impl_cls = type(self)._impl_cls
        if impl_cls is not None:
            return self._predict_via_impl(cell_line_ids, drug_ids, cell_line_input, drug_input)
        if self._bridge.composed is None:
            msg = "Model has not been built; call build_model() before predict()"
            raise RuntimeError(msg)
        msg = "Model has not been trained; call train() or load() before predict()"
        raise RuntimeError(msg)

    def save(self, directory: str) -> None:
        if self._bridge.is_trained():
            save_component_stack(self._bridge, directory, hyperparameters=self.hyperparameters)
            return
        impl_cls = type(self)._impl_cls
        if impl_cls is not None and "save" in impl_cls.__dict__:
            impl_cls.save(self, directory)
            return
        super().save(directory)

    @classmethod
    def load(cls, directory: str) -> LiteratureComponentDRPModel:
        hyperparameters_path = os.path.join(directory, _HYPERPARAMETERS_FILE)
        hyperparameters: dict[str, Any] = {}
        if os.path.exists(hyperparameters_path):
            with open(hyperparameters_path) as handle:
                hyperparameters = json.load(handle)
        if has_component_stack(directory):
            wrapper = cls()
            wrapper.build_model(hyperparameters)
            loaded_hp = load_native_checkpoint(wrapper, directory)
            wrapper.hyperparameters = loaded_hp or hyperparameters
            return wrapper
        if cls._impl_cls is None:
            msg = f"{cls.__name__}.load is not implemented"
            raise NotImplementedError(msg)
        loaded = cls._impl_cls.load(directory)
        wrapper = cls()
        for name, value in vars(loaded).items():
            if name.startswith("_"):
                continue
            setattr(wrapper, name, value)
        wrapper.hyperparameters = dict(getattr(loaded, "hyperparameters", {}))
        if hasattr(loaded, "cell_line_views"):
            wrapper.cell_line_views = cast(list[str], loaded.cell_line_views)
        if hasattr(loaded, "drug_views"):
            wrapper.drug_views = cast(list[str], loaded.drug_views)
        wrapper.build_model(dict(wrapper.hyperparameters))
        restore_literature_to_components(wrapper)
        return wrapper


class PrecilyModel(LiteratureComponentDRPModel):
    """Precily model component."""

    _zoo_name = "Precily"
    _impl_cls = _PrecilyImpl


class SRMF(LiteratureComponentDRPModel):
    """Srmf component."""

    _zoo_name = "SRMF"
    _impl_cls = _SRMFImpl


class DrugGNN(LiteratureComponentDRPModel):
    """Drug gnn component."""

    _zoo_name = "DrugGNN"
    _impl_cls = _DrugGNN


class SimpleNeuralNetwork(LiteratureComponentDRPModel):
    """Simple neural network component."""

    _zoo_name = "SimpleNeuralNetwork"
    _impl_cls = _SimpleNeuralNetwork


class MultiViewNeuralNetwork(LiteratureComponentDRPModel):
    """Multi view neural network component."""

    _zoo_name = "MultiViewNeuralNetwork"
    _impl_cls = _MultiViewNeuralNetwork


class MOLIR(LiteratureComponentDRPModel):
    """Molir component."""

    _zoo_name = "MOLIR"
    _impl_cls = _MOLIR
    is_single_drug_model = True


class SuperFELTR(LiteratureComponentDRPModel):
    """Super feltr component."""

    _zoo_name = "SuperFELTR"
    _impl_cls = _SuperFELTR
    is_single_drug_model = True


class PharmaFormerModel(LiteratureComponentDRPModel):
    """Pharma former model component."""

    _zoo_name = "PharmaFormer"
    _impl_cls = _PharmaFormerModel


class DIPKModel(LiteratureComponentDRPModel):
    """Dipkmodel component."""

    _zoo_name = "DIPK"
    _impl_cls = _DIPKModel


class SparseGOModel(LiteratureComponentDRPModel):
    """SparseGO literature model routed through the component stack."""

    _zoo_name = "SparseGO"
    _impl_cls = _SparseGOModel


__all__ = [
    "DIPKModel",
    "DrugGNN",
    "LiteratureComponentDRPModel",
    "MOLIR",
    "MultiViewNeuralNetwork",
    "PharmaFormerModel",
    "PrecilyModel",
    "SRMF",
    "SimpleNeuralNetwork",
    "SparseGOModel",
    "SuperFELTR",
]
