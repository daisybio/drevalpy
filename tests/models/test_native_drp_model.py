"""Tests for the shared NativeDRPModel facade."""

from __future__ import annotations

import tempfile

import numpy as np

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._native_drp_model import create_native_drp_class
from drevalpy.models.legacy_checkpoint_migration import migrate_checkpoint_to_component_stack


def _synthetic_data() -> tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0])},
            "d2": {"fingerprints": np.array([0.0, 1.0])},
        }
    )
    return response, cell_line_input, drug_input


def test_native_drp_class_supports_factory_lifecycle() -> None:
    register_builtin_components()
    NativeElasticNet = create_native_drp_class("ElasticNet", spec="geneExpression:fingerprints:elasticNet")
    model = NativeElasticNet()
    assert model.get_model_name() == "ElasticNet"
    model.build_model({"alpha": 0.1, "l1_ratio": 0.5})
    response, cell_line_input, drug_input = _synthetic_data()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = NativeElasticNet.load(tmp)
        loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)


def test_native_naive_class_round_trip() -> None:
    register_builtin_components()
    NativeNaive = create_native_drp_class("NaiveDrugMeanPredictor", spec="NaiveDrugMeanPredictor")
    model = NativeNaive()
    model.build_model({})
    response = DrugResponseDataset(
        response=np.array([1.0, 3.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    model.train(response, FeatureDataset(features={}), FeatureDataset(features={}))
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        migrate_checkpoint_to_component_stack(model, tmp, output_directory=tmp)
        loaded = NativeNaive.load(tmp)
    assert loaded._bridge.is_trained()
