"""Public DRPModel lifecycle and persistence contract."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from drevalpy.models import DRPModel
from drevalpy.models._pipeline import PipelineState
from tests.models.synthetic_fixtures import (
    lso_split_masks,
    small_data_spec,
    synthetic_mudataset_gene_expression_fingerprints,
)


def test_constructor_requires_a_model_spec() -> None:
    with pytest.raises(TypeError, match="ModelSpec"):
        DRPModel("ElasticNet")  # type: ignore[arg-type]


def test_properties_expose_name_config_and_independent_hyperparameters() -> None:
    model = DRPModel(small_data_spec("ElasticNet"), {"alpha": 0.1, "l1_ratio": 0.5})

    assert model.name == "ElasticNet"
    assert model.config.predictor_values()["alpha"] == 0.1
    values = model.hyperparameters
    values["alpha"] = 99
    assert model.hyperparameters["alpha"] == 0.1


def test_train_predict_and_load_preserve_original_response_units(tmp_path) -> None:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lso_split_masks()
    model = DRPModel(small_data_spec("ElasticNet"), response_transformation=StandardScaler())

    model.train(dataset, split.train, model_checkpoint_dir=tmp_path)
    predictions = model.predict(dataset, split.test)
    model.save(tmp_path / "model")
    restored = DRPModel.load(tmp_path / "model")

    assert model._pipeline.state is PipelineState.FITTED
    assert predictions.shape == (2,)
    np.testing.assert_allclose(restored.predict(dataset, split.test), predictions)


def test_train_without_a_checkpoint_dir_uses_a_temporary_one_and_removes_it(tmp_path, monkeypatch) -> None:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lso_split_masks()
    model = DRPModel(small_data_spec("ElasticNet"))
    seen = []
    original_fit = model._pipeline.fit

    def spy(*args, training_context, **kwargs):
        seen.append(training_context.checkpoint_dir)
        assert training_context.checkpoint_dir.is_dir()
        return original_fit(*args, training_context=training_context, **kwargs)

    monkeypatch.setattr(model._pipeline, "fit", spy)
    monkeypatch.chdir(tmp_path)

    model.train(dataset, split.train)

    assert len(seen) == 1
    assert not seen[0].exists()
    assert not (tmp_path / "checkpoints").exists()
    assert list(tmp_path.iterdir()) == []
