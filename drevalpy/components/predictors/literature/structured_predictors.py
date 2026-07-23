"""Native structured literature predictors backed by component-owned engines."""

from __future__ import annotations

from typing import Any, ClassVar

import joblib
import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase
from drevalpy.components.predictors.literature._feature_dataset_from_batch import (
    feature_dataset_from_blocks,
    merge_feature_dataset,
)
from drevalpy.components.predictors.literature._metadata import (
    DIPK_METADATA,
    MOLIR_METADATA,
    PHARMAFORMER_METADATA,
    PRECIILY_METADATA,
    SPARSEGO_METADATA,
    SRMF_METADATA,
    SUPERFELTR_METADATA,
)
from drevalpy.components.predictors.literature.impl.dipk.dipk import DIPKModel
from drevalpy.components.predictors.literature.impl.molir.molir import MOLIR
from drevalpy.components.predictors.literature.impl.pharmaformer.pharmaformer import PharmaFormerModel
from drevalpy.components.predictors.literature.impl.precily.precily import PrecilyModel
from drevalpy.components.predictors.literature.impl.sparsego.sparsego import SparseGOModel
from drevalpy.components.predictors.literature.impl.srmf.srmf import SRMF
from drevalpy.components.predictors.literature.impl.superfeltr.superfeltr import SuperFELTR
from drevalpy.components.predictors.structured import StructuredPredictor
from drevalpy.components.registry import register_predictor
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.config import PredictionMode


class StructuredLiteratureEnginePredictor(StructuredPredictor):
    """Train a component-owned literature engine on featurizer-produced blocks."""

    _engine_cls: ClassVar[type[LiteratureEngineBase]]
    requires_raw_feature_datasets: ClassVar[bool] = False
    requires_drug_featurizer: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    _ENGINE_PRELOAD_ATTRS: ClassVar[tuple[str, ...]] = (
        "layer_connections",
        "gene2id_mapping_ont",
        "ontology_gene_order",
        "gene_dim_input",
        "model",
    )

    def __init__(self) -> None:
        self._hyperparameters: dict[str, Any] = {}
        self._engine: LiteratureEngineBase | None = None
        self._engine_preload_state: dict[str, Any] = {}

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        _ = input_dims
        self._hyperparameters = dict(hyperparameters)

    def set_engine_preload_state(self, state: dict[str, Any]) -> None:
        self._engine_preload_state = dict(state)

    @classmethod
    def load_dataset_cell_line_features(
        cls,
        data_path: str,
        dataset_name: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> tuple[FeatureDataset, dict[str, Any]]:
        _ = model_name
        engine = cls._engine_cls()
        if hyperparameters:
            engine.hyperparameters = dict(hyperparameters)
        features = engine.load_cell_line_features(data_path, dataset_name)
        preload = {
            attr: getattr(engine, attr) for attr in cls._ENGINE_PRELOAD_ATTRS if getattr(engine, attr, None) is not None
        }
        return features, preload

    @classmethod
    def load_dataset_drug_features(
        cls,
        data_path: str,
        dataset_name: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> FeatureDataset | None:
        _ = model_name
        engine = cls._engine_cls()
        if hyperparameters:
            engine.hyperparameters = dict(hyperparameters)
        features = engine.load_drug_features(data_path, dataset_name)
        if hyperparameters is not None:
            hyperparameters.update(dict(engine.hyperparameters))
        return features

    def _materialize_inputs(
        self,
        batch: ModelInputBatch,
    ) -> tuple[FeatureDataset, FeatureDataset | None]:
        cell_line_input = batch.cell_line_input
        drug_input = batch.drug_input
        if cell_line_input is None:
            msg = "structured literature predictor requires cell_line_input"
            raise RuntimeError(msg)
        cell_lines = cell_line_input
        if batch.cell_line_blocks:
            cell_lines = merge_feature_dataset(cell_line_input, batch.cell_line_blocks, batch.cell_line_entity_ids)
        if not self.requires_drug_featurizer:
            return cell_lines, None
        if drug_input is None:
            msg = "structured literature predictor requires drug_input"
            raise RuntimeError(msg)
        if batch.drug_blocks and batch.drug_entity_ids is not None:
            drugs = merge_feature_dataset(drug_input, batch.drug_blocks, batch.drug_entity_ids)
        elif batch.drug_entity_ids is not None:
            drugs = feature_dataset_from_blocks(batch.drug_entity_ids, batch.drug_blocks, fallback=drug_input)
        else:
            drugs = drug_input
        return cell_lines, drugs

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "structured literature predictor requires response"
            raise RuntimeError(msg)
        if batch.cell_line_features.size == 0 and not self.requires_raw_feature_datasets:
            msg = "cell_line featurizer produced no features"
            raise ValueError(msg)
        cell_line_input = batch.cell_line_input
        drug_input = batch.drug_input
        if self.requires_raw_feature_datasets:
            if cell_line_input is None:
                msg = "structured literature predictor requires cell_line_input"
                raise RuntimeError(msg)
            cell_lines = cell_line_input
            drugs = None if not self.requires_drug_featurizer else drug_input
        else:
            cell_lines, drugs = self._materialize_inputs(batch)
        output = DrugResponseDataset(
            response=batch.response,
            cell_line_ids=batch.cell_line_ids,
            drug_ids=batch.drug_ids,
        )
        hyperparameters = dict(self._hyperparameters)
        engine = self._engine_cls()
        for name, value in self._engine_preload_state.items():
            setattr(engine, name, value)
        engine.build_model(hyperparameters)
        engine.train(
            output,
            cell_lines,
            drugs,
            output_earlystopping=batch.early_stopping_response,
            model_checkpoint_dir=batch.training_context.checkpoint_dir,
        )
        self._engine = engine

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        cell_line_input = batch.cell_line_input
        if self._engine is None or cell_line_input is None:
            return np.full(batch.n_pairs, np.nan, dtype=np.float64)
        drug_input = batch.drug_input
        if self.requires_raw_feature_datasets:
            drugs = None if not self.requires_drug_featurizer else drug_input
            return self._engine.predict(
                batch.cell_line_ids,
                batch.drug_ids,
                cell_line_input,
                drugs,
            )
        cell_lines, drugs = self._materialize_inputs(batch)
        return self._engine.predict(
            batch.cell_line_ids,
            batch.drug_ids,
            cell_lines,
            drugs,
        )

    def is_fitted(self) -> bool:
        return self._engine is not None

    def get_state(self) -> dict[str, object]:
        if self._engine is None:
            return {}
        import io

        buffer = io.BytesIO()
        joblib.dump(self._engine, buffer)
        return {
            "hyperparameters": dict(self._hyperparameters),
            "engine": buffer.getvalue(),
        }

    def set_state(self, state: dict[str, object]) -> None:
        import io

        engine_blob = state.get("engine")
        if isinstance(engine_blob, (bytes, bytearray)):
            loaded = joblib.load(io.BytesIO(engine_blob))
            if isinstance(loaded, LiteratureEngineBase):
                self._engine = loaded
        hyperparameters = state.get("hyperparameters")
        if isinstance(hyperparameters, dict):
            self._hyperparameters = dict(hyperparameters)


@register_predictor(
    "precily",
    description="Precily pathway + SMILESVec model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **PRECIILY_METADATA,
)
class PrecilyPredictor(StructuredLiteratureEnginePredictor):
    """Precily predictor component."""

    _engine_cls = PrecilyModel


@register_predictor(
    "srmf",
    description="SRMF matrix factorization model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **SRMF_METADATA,
)
class SRMFPredictor(StructuredLiteratureEnginePredictor):
    """Srmfpredictor component."""

    _engine_cls = SRMF


@register_predictor(
    "molir",
    description="MOLIR single-drug multi-omics model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **MOLIR_METADATA,
)
class MOLIRPredictor(StructuredLiteratureEnginePredictor):
    """Molirpredictor component."""

    requires_drug_featurizer: ClassVar[bool] = False
    requires_raw_feature_datasets: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = True
    _engine_cls = MOLIR


@register_predictor(
    "superfeltr",
    description="SuperFELTR single-drug multi-omics model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **SUPERFELTR_METADATA,
)
class SuperFELTRPredictor(StructuredLiteratureEnginePredictor):
    """Super feltrpredictor component."""

    requires_drug_featurizer: ClassVar[bool] = False
    requires_raw_feature_datasets: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = True
    _engine_cls = SuperFELTR


@register_predictor(
    "pharmaFormer",
    description="PharmaFormer landmark genes + BPE PharmaFormer model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **PHARMAFORMER_METADATA,
)
class PharmaFormerPredictor(StructuredLiteratureEnginePredictor):
    """Pharma former predictor component."""

    requires_raw_feature_datasets: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = True
    _engine_cls = PharmaFormerModel


@register_predictor(
    "dipk",
    description="DIPK BIONIC + MolGNet model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **DIPK_METADATA,
)
class DIPKPredictor(StructuredLiteratureEnginePredictor):
    """Dipkpredictor component."""

    requires_raw_feature_datasets: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = True
    _engine_cls = DIPKModel


@register_predictor(
    "sparsego",
    description="SparseGO GO-structured visible neural network.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **SPARSEGO_METADATA,
)
class SparseGOPredictor(StructuredLiteratureEnginePredictor):
    """SparseGO predictor component."""

    requires_raw_feature_datasets: ClassVar[bool] = True
    _engine_cls = SparseGOModel
