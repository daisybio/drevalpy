"""Public DRPModel compatibility classes backed by the component stack."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.models.factory import model_config_for_name
from drevalpy.components.register_builtins import ensure_components_registered
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._component_bridge import ComponentDRPBridge, restore_literature_to_components
from drevalpy.models.drp_model import DRPModel

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
from drevalpy.components.predictors.literature.impl.srmf.srmf import SRMF as _SRMFImpl
from drevalpy.components.predictors.literature.impl.superfeltr.superfeltr import SuperFELTR as _SuperFELTR


class LiteratureComponentDRPModel(DRPModel):
    """Route legacy experiment APIs through zoo-backed :class:`ComposedModel` stacks."""

    _zoo_name: ClassVar[str]

    def __init__(self) -> None:
        super().__init__()
        self._bridge = ComponentDRPBridge()
        self.hyperparameters: dict[str, Any] = {}

    @classmethod
    def get_model_name(cls) -> str:
        return cls._zoo_name

    @classmethod
    def get_hyperparameter_set(cls) -> list[dict[str, Any]]:
        for base in cls.__bases__:
            if base is LiteratureComponentDRPModel:
                continue
            if issubclass(base, DRPModel):
                return base.get_hyperparameter_set()
        return super().get_hyperparameter_set()

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = dict(hyperparameters)
        ensure_components_registered()
        config = model_config_for_name(self._zoo_name, hyperparameters)
        self._bridge.set_composed_config(config)
        for base in self.__class__.__bases__:
            if base is LiteratureComponentDRPModel:
                continue
            if issubclass(base, DRPModel):
                base.build_model(self, hyperparameters)
                break

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
            state = cell_featurizer.get_state()
            if "gene_expression_scaler" in state:
                self.gene_expression_scaler = state["gene_expression_scaler"]
            if "methylation_scaler" in state:
                self.methylation_scaler = state["methylation_scaler"]
            if "methylation_pca" in state:
                self.methylation_pca = state["methylation_pca"]
            if "views" in state:
                self.cell_line_views = list(state["views"])

        drug_featurizer = composed._drug_featurizer
        if drug_featurizer is not None and hasattr(drug_featurizer, "_view") and "drug_views" in vars(self):
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
            view_dims = getattr(cell_featurizer, "view_dims", None)
            if view_dims:
                self.input_dims = dict(view_dims)
                if drug_featurizer is not None and composed._drug_matrix is not None and composed._drug_matrix.size:
                    from drevalpy.models.composed_model import _matrix_feature_width

                    drug_view = self.drug_views[0] if getattr(self, "drug_views", None) else "fingerprints"
                    self.input_dims[drug_view] = _matrix_feature_width(composed._drug_matrix)
            elif cell_featurizer is not None and hasattr(cell_featurizer, "_view"):
                self.input_dims = {cell_featurizer._view: int(cell_featurizer.output_dim)}
                if drug_featurizer is not None and composed._drug_matrix is not None and composed._drug_matrix.size:
                    from drevalpy.models.composed_model import _matrix_feature_width

                    drug_view = self.drug_views[0] if getattr(self, "drug_views", None) else "fingerprints"
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
        self._bridge.train(
            output,
            cell_line_input,
            drug_input,
            output_earlystopping=output_earlystopping,
        )
        self._sync_impl_state_from_bridge()

    def _impl_base(self) -> type[DRPModel] | None:
        for base in self.__class__.__bases__:
            if base is LiteratureComponentDRPModel:
                continue
            if issubclass(base, DRPModel):
                return base
        return None

    def _bridge_is_trained(self) -> bool:
        return self._bridge.is_trained()

    def _predict_via_impl(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        impl_base = self._impl_base()
        if impl_base is None:
            return np.full(len(cell_line_ids), np.nan)
        return impl_base.predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        if self._bridge_is_trained():
            return self._bridge.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)
        if getattr(self, "model", None) is not None:
            return self._predict_via_impl(cell_line_ids, drug_ids, cell_line_input, drug_input)
        return self._bridge.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)

    @classmethod
    def load(cls, directory: str) -> LiteratureComponentDRPModel:
        impl_base = None
        for base in cls.__bases__:
            if base is LiteratureComponentDRPModel:
                continue
            if issubclass(base, DRPModel):
                impl_base = base
                break
        if impl_base is None:
            msg = f"{cls.__name__}.load is not implemented"
            raise NotImplementedError(msg)
        loaded = impl_base.load(directory)
        wrapper = cls()
        for name, value in vars(loaded).items():
            if name.startswith("_"):
                continue
            setattr(wrapper, name, value)
        wrapper.build_model(dict(wrapper.hyperparameters))
        restore_literature_to_components(wrapper)
        return wrapper


class PrecilyModel(LiteratureComponentDRPModel, _PrecilyImpl):
    _zoo_name = "Precily"


class SRMF(LiteratureComponentDRPModel, _SRMFImpl):
    _zoo_name = "SRMF"


class DrugGNN(LiteratureComponentDRPModel, _DrugGNN):
    _zoo_name = "DrugGNN"


class SimpleNeuralNetwork(LiteratureComponentDRPModel, _SimpleNeuralNetwork):
    _zoo_name = "SimpleNeuralNetwork"


class MultiViewNeuralNetwork(LiteratureComponentDRPModel, _MultiViewNeuralNetwork):
    _zoo_name = "MultiViewNeuralNetwork"


class MOLIR(LiteratureComponentDRPModel, _MOLIR):
    _zoo_name = "MOLIR"
    is_single_drug_model = True


class SuperFELTR(LiteratureComponentDRPModel, _SuperFELTR):
    _zoo_name = "SuperFELTR"
    is_single_drug_model = True


class PharmaFormerModel(LiteratureComponentDRPModel, _PharmaFormerModel):
    _zoo_name = "PharmaFormer"


class DIPKModel(LiteratureComponentDRPModel, _DIPKModel):
    _zoo_name = "DIPK"


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
    "SuperFELTR",
]
