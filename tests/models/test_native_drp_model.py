"""Tests for the shared NativeDRPModel facade."""

from __future__ import annotations

import tempfile

import numpy as np

from drevalpy.models._native_drp_model import create_native_drp_class
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    identity_cell_line_features,
    identity_drug_features,
    multi_drug_response,
)


def test_native_drp_class_supports_factory_lifecycle() -> None:
    NativeElasticNet = create_native_drp_class("ElasticNet", spec="ElasticNet", validate_spec=False)
    model = NativeElasticNet()
    assert model.get_model_name() == "ElasticNet"
    model.build_model({"alpha": 0.1, "l1_ratio": 0.5})
    response = multi_drug_response()
    cell_line_input = cell_line_gene_expression()
    drug_input = drug_fingerprints()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = NativeElasticNet.load(tmp)
        loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)


def test_native_naive_class_round_trip() -> None:
    NativeNaive = create_native_drp_class("NaiveDrugMeanPredictor", spec="NaiveDrugMeanPredictor", validate_spec=False)
    model = NativeNaive()
    model.build_model({})
    response = multi_drug_response()
    cell_line_input = identity_cell_line_features()
    drug_input = identity_drug_features()
    model.train(response, cell_line_input, drug_input)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = NativeNaive.load(tmp)
    assert loaded._composed is not None
    assert loaded._composed.is_fitted()
