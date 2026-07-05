"""Native structured literature predictors backed by component-owned engines."""

from __future__ import annotations

from typing import Any, ClassVar

import joblib
import numpy as np

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.pair_batch import PairBatch
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
    _use_raw_inputs: ClassVar[bool] = False
    requires_drug_featurizer: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self) -> None:
        self._hyperparameters: dict[str, Any] = {}
        self._engine: LiteratureEngineBase | None = None

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        _ = input_dims
        self._hyperparameters = dict(hyperparameters)

    def _materialize_inputs(
        self,
        batch: PairBatch,
        *,
        cell_line_input: FeatureDataset | None,
        drug_input: FeatureDataset | None,
    ) -> tuple[FeatureDataset, FeatureDataset | None]:
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

    def _engine_from_host(self) -> LiteratureEngineBase:
        engine = self._engine_cls()
        host = getattr(self, "_literature_host", None)
        if host is not None:
            for name, value in vars(host).items():
                if name.startswith("_"):
                    continue
                setattr(engine, name, value)
        return engine

    def fit_structured(
        self,
        batch: PairBatch,
        *,
        output: DrugResponseDataset | None = None,
        cell_line_input: FeatureDataset | None = None,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        if output is None:
            msg = "structured literature predictor requires output"
            raise RuntimeError(msg)
        if batch.cell_line_features.size == 0:
            msg = "cell_line featurizer produced no features"
            raise ValueError(msg)
        if self._use_raw_inputs:
            if cell_line_input is None:
                msg = "structured literature predictor requires cell_line_input"
                raise RuntimeError(msg)
            cell_lines = cell_line_input
            drugs = None if not self.requires_drug_featurizer else drug_input
        else:
            cell_lines, drugs = self._materialize_inputs(
                batch,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
            )
        host = getattr(self, "_literature_host", None)
        hyperparameters = {**self._hyperparameters, **getattr(host, "hyperparameters", {})}
        engine = self._engine_from_host()
        engine.build_model(hyperparameters)
        engine.train(
            output,
            cell_lines,
            drugs,
            output_earlystopping=output_earlystopping,
        )
        self._engine = engine

    def predict_structured(
        self,
        batch: PairBatch,
        *,
        cell_line_input: FeatureDataset | None = None,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        if self._engine is None or cell_line_input is None:
            return np.full(len(batch.cell_line_ids), np.nan, dtype=np.float64)
        if self._use_raw_inputs:
            drugs = None if not self.requires_drug_featurizer else drug_input
            return self._engine.predict(
                batch.cell_line_ids,
                batch.drug_ids,
                cell_line_input,
                drugs,
            )
        cell_lines, drugs = self._materialize_inputs(
            batch,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
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
    **PRECIILY_METADATA,
)
class PrecilyPredictor(StructuredLiteratureEnginePredictor):
    """Precily predictor component."""

    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, view="pathways")
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, view="smilesvec")
    _engine_cls = PrecilyModel


@register_predictor(
    "srmf",
    description="SRMF matrix factorization model.",
    **SRMF_METADATA,
)
class SRMFPredictor(StructuredLiteratureEnginePredictor):
    """Srmfpredictor component."""

    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(
        kind=FeatureKind.DENSE, view="gene_expression"
    )
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, view="fingerprints")
    _engine_cls = SRMF


@register_predictor(
    "molir",
    description="MOLIR single-drug multi-omics model.",
    **MOLIR_METADATA,
)
class MOLIRPredictor(StructuredLiteratureEnginePredictor):
    """Molirpredictor component."""

    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, scope="multi_view")
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)
    requires_drug_featurizer: ClassVar[bool] = False
    _use_raw_inputs: ClassVar[bool] = True
    _engine_cls = MOLIR


@register_predictor(
    "superfeltr",
    description="SuperFELTR single-drug multi-omics model.",
    **SUPERFELTR_METADATA,
)
class SuperFELTRPredictor(StructuredLiteratureEnginePredictor):
    """Super feltrpredictor component."""

    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, scope="multi_view")
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)
    requires_drug_featurizer: ClassVar[bool] = False
    _use_raw_inputs: ClassVar[bool] = True
    _engine_cls = SuperFELTR


@register_predictor(
    "pharmaFormer",
    description="PharmaFormer landmark genes + BPE PharmaFormer model.",
    **PHARMAFORMER_METADATA,
)
class PharmaFormerPredictor(StructuredLiteratureEnginePredictor):
    """Pharma former predictor component."""

    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(
        kind=FeatureKind.DENSE, view="gene_expression"
    )
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, view="bpe_smiles")
    _use_raw_inputs: ClassVar[bool] = True
    _engine_cls = PharmaFormerModel


@register_predictor(
    "dipk",
    description="DIPK BIONIC + MolGNet model.",
    **DIPK_METADATA,
)
class DIPKPredictor(StructuredLiteratureEnginePredictor):
    """Dipkpredictor component."""

    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, scope="multi_view")
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, view="molgnet_features")
    _use_raw_inputs: ClassVar[bool] = True
    _engine_cls = DIPKModel


@register_predictor(
    "sparsego",
    description="SparseGO GO-structured visible neural network.",
    **SPARSEGO_METADATA,
)
class SparseGOPredictor(StructuredLiteratureEnginePredictor):
    """SparseGO predictor component."""

    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(
        kind=FeatureKind.DENSE, view="gene_expression"
    )
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, view="fingerprints")
    _use_raw_inputs: ClassVar[bool] = True
    _engine_cls = SparseGOModel
