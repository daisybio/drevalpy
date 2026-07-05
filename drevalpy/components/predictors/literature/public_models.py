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
from drevalpy.components.state_helpers import state_str_list
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._component_bridge import (
    _COMPONENT_STACK_FILE,
    _HYPERPARAMETERS_FILE,
    ComponentDRPBridge,
    load_component_stack,
    restore_literature_to_components,
    save_component_stack,
    sync_literature_from_components,
)
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.factory import model_config_for_name


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
        impl_cls = type(self)._impl_cls
        if impl_cls is not None:
            temp = impl_cls()
            for key, value in temp.__dict__.items():
                if key != "_bridge":
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

    def _sync_impl_state_from_bridge(self) -> None:
        composed = self._bridge.composed
        if composed is None:
            return
        predictor = composed._predictor

        engine = getattr(predictor, "_engine", None)
        if engine is not None:
            for name, value in vars(engine).items():
                if name.startswith("_"):
                    continue
                setattr(self, name, value)

        component_model = getattr(predictor, "_model", None)
        if component_model is not None:
            self.model = component_model

        cell_featurizer = composed._cell_line_featurizer
        if cell_featurizer is not None:
            from drevalpy.components.featurizers.cell_line.concat import (
                ConcatFeaturizersCellLineFeaturizer,
            )

            if isinstance(cell_featurizer, ConcatFeaturizersCellLineFeaturizer):
                state = ConcatFeaturizersCellLineFeaturizer.collect_legacy_state(cell_featurizer)
            else:
                state = cell_featurizer.get_state()
            if "gene_expression_scaler" in state:
                self.gene_expression_scaler = state["gene_expression_scaler"]
            if "methylation_scaler" in state:
                self.methylation_scaler = state["methylation_scaler"]
            if "methylation_pca" in state:
                self.methylation_pca = state["methylation_pca"]
            views = state_str_list(state, "views")
            if views is not None:
                self.cell_line_views = views

        drug_featurizer = composed._drug_featurizer
        if drug_featurizer is not None and hasattr(drug_featurizer, "_view") and hasattr(self, "drug_views"):
            self.drug_views = [drug_featurizer._view]

        if composed._cell_line_matrix is not None and composed._cell_line_matrix.size:
            cell_dim = int(composed._cell_line_matrix.shape[1])
            if hasattr(self, "hyperparameters") and self.hyperparameters.get("input_dim_omic") is None:
                self.hyperparameters["input_dim_omic"] = cell_dim
        if composed._drug_matrix is not None and composed._drug_matrix.size:
            from drevalpy.models.composed_model import _matrix_feature_width

            drug_dim = _matrix_feature_width(composed._drug_matrix)
            if hasattr(self, "hyperparameters") and self.hyperparameters.get("input_dim_fp") is None:
                self.hyperparameters["input_dim_fp"] = drug_dim

        if hasattr(self, "input_dims"):
            view_dims = None
            if cell_featurizer is not None:
                if isinstance(cell_featurizer, ConcatFeaturizersCellLineFeaturizer):
                    view_dims = state.get("view_dims")
                else:
                    view_dims = getattr(cell_featurizer, "view_dims", None)
            if isinstance(view_dims, dict) and view_dims:
                self.input_dims = dict(view_dims)
                if drug_featurizer is not None and composed._drug_matrix is not None and composed._drug_matrix.size:
                    from drevalpy.models.composed_model import _matrix_feature_width

                    drug_view = self.drug_views[0] if self.drug_views else "fingerprints"
                    self.input_dims[drug_view] = _matrix_feature_width(composed._drug_matrix)
            elif cell_featurizer is not None and hasattr(cell_featurizer, "_view"):
                self.input_dims = {cell_featurizer._view: int(cell_featurizer.output_dim)}
                if drug_featurizer is not None and composed._drug_matrix is not None and composed._drug_matrix.size:
                    from drevalpy.models.composed_model import _matrix_feature_width

                    drug_view = self.drug_views[0] if self.drug_views else "fingerprints"
                    self.input_dims[drug_view] = _matrix_feature_width(composed._drug_matrix)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        _ = model_checkpoint_dir
        composed = self._bridge.composed
        predictor = composed._predictor if composed is not None else None
        if isinstance(predictor, StructuredLiteratureEnginePredictor):
            predictor._literature_host = self
        try:
            self._bridge.train(
                output,
                cell_line_input,
                drug_input,
                output_earlystopping=output_earlystopping,
            )
        finally:
            if isinstance(predictor, StructuredLiteratureEnginePredictor):
                predictor._literature_host = None
        self._sync_impl_state_from_bridge()

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
        stack_path = os.path.join(directory, _COMPONENT_STACK_FILE)
        if os.path.exists(stack_path):
            wrapper = cls()
            wrapper.build_model(hyperparameters)
            loaded_hp = load_component_stack(wrapper._bridge, directory)
            wrapper.hyperparameters = loaded_hp or hyperparameters
            sync_literature_from_components(wrapper)
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
