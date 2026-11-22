"""Tests for the SuperFELTR predictor, encoders, regressor and per-module fit."""

from __future__ import annotations

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.literature.superfeltr import predictor as superfeltr_predictor
from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.components.predictors.literature.superfeltr.utils import SuperFELTEncoder, SuperFELTRegressor
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.data.response_table import ResponseTable
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_pickle_round_trip,
    assert_training_checkpoint_score,
    assert_training_predictions,
    finite_predictions,
    seed_literature,
    single_drug_omics_batch,
)


def test_superfeltr_predictor_registry_name() -> None:
    assert get_predictor("superfeltr") is SuperFELTRPredictor


class TestTinySuperFELTRFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_fits_triplet_batches_and_round_trips_through_pickle_without_feature_names(
        self, seed: int, tmp_path, monkeypatch
    ) -> None:
        checkpoints: list = []
        fit = superfeltr_predictor.fit_omics_model
        monkeypatch.setattr(
            superfeltr_predictor,
            "fit_omics_model",
            lambda *args, **kwargs: checkpoints.append(fit(*args, **kwargs)) or checkpoints[-1],
        )
        seed_literature(seed)
        batch = single_drug_omics_batch(tmp_path)
        predictor = SuperFELTRPredictor(
            hyperparameters={
                "epochs": 1,
                "mini_batch": 2,
                "out_dim_expr_encoder": 3,
                "out_dim_mutation_encoder": 3,
                "out_dim_cnv_encoder": 2,
                "dropout_rate": 0.0,
            }
        )
        predictor.fit(batch)
        # The regressor is fitted last; its best checkpoint is no longer kept on the fitted model.
        regressor_checkpoint = checkpoints[-1]
        assert regressor_checkpoint is not None
        assert np.isfinite(float(regressor_checkpoint.best_model_score))
        assert_training_checkpoint_score("superfeltr", seed, regressor_checkpoint.best_model_score)
        expected = finite_predictions(predictor, batch)
        assert_training_predictions("superfeltr", seed, expected)
        assert_frozen_reference("superfeltr", seed)

        assert_pickle_round_trip(predictor, batch, expected)


EXPR_DIM = 6
MUT_DIM = 5
CNV_DIM = 4
OUT_EXPR = 4
OUT_MUT = 3
OUT_CNV = 2
MINI_BATCH = 2
N_ENTITIES = 6
N_PAIRS = 2 * MINI_BATCH + 2

BASE_HPAMS: dict[str, int | float | dict] = {
    "dropout_rate": 0.1,
    "margin": 1.0,
    "learning_rate": 0.01,
    "weight_decay": 0.01,
    "out_dim_expr_encoder": OUT_EXPR,
    "out_dim_mutation_encoder": OUT_MUT,
    "out_dim_cnv_encoder": OUT_CNV,
    "epochs": 1,
    "mini_batch": MINI_BATCH,
}

RANGES = (0.1, 1.0)


def _hpams(**overrides: object) -> dict[str, int | float | dict]:
    merged = dict(BASE_HPAMS)
    merged.update(overrides)  # type: ignore[arg-type]
    return merged


def _encoder(omic_type: str = "expression", input_size: int = EXPR_DIM, **overrides: object) -> SuperFELTEncoder:
    return SuperFELTEncoder(
        input_size=input_size,
        hpams=_hpams(**overrides),
        omic_type=omic_type,
        ranges=RANGES,
    )


def _omics() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return (
        rng.normal(size=(N_ENTITIES, EXPR_DIM)).astype(np.float32),
        rng.normal(size=(N_ENTITIES, MUT_DIM)).astype(np.float32),
        rng.normal(size=(N_ENTITIES, CNV_DIM)).astype(np.float32),
    )


def _omics_batch(tmp_path, *, with_val: bool = False) -> PredictorInput:
    expr, mut, cnv = _omics()
    entity_ids = np.array([f"s{i}" for i in range(N_ENTITIES)])
    pair_ids = entity_ids[np.arange(N_PAIRS) % N_ENTITIES]
    pairs = ResponseTable(
        response=np.linspace(0.0, 1.0, N_PAIRS, dtype=np.float32),
        sample_ids=pair_ids,
        drug_ids=np.repeat("d1", N_PAIRS),
    )
    sample = FeaturizerOutput(
        entity_ids=entity_ids,
        blocks={
            "gene_expression": FeatureBlock(expr),
            "mutations": FeatureBlock(mut),
            "copy_number_variation_gistic": FeatureBlock(cnv),
        },
    )
    return PredictorInput(
        pairs=pairs,
        sample=sample,
        early_stopping=pairs if with_val else None,
        training_context=TrainingContext(checkpoint_dir=tmp_path),
    )


def _regressor(**overrides: object) -> SuperFELTRegressor:
    encoders = (
        _encoder("expression", EXPR_DIM),
        _encoder("mutation", MUT_DIM),
        _encoder("copy_number_variation_gistic", CNV_DIM),
    )
    return SuperFELTRegressor(
        input_size=OUT_EXPR + OUT_MUT + OUT_CNV,
        hpams=_hpams(**overrides),
        encoders=encoders,
    )


@pytest.mark.parametrize("hyperparameter", ["dropout_rate", "margin", "learning_rate", "weight_decay"])
def test_encoder_rejects_non_float_hyperparameters(hyperparameter: str) -> None:
    with pytest.raises(ValueError, match="must be floats"):
        _encoder(**{hyperparameter: 1})


@pytest.mark.parametrize("hyperparameter", ["out_dim_expr_encoder", "out_dim_mutation_encoder", "out_dim_cnv_encoder"])
def test_encoder_rejects_non_integer_output_sizes(hyperparameter: str) -> None:
    with pytest.raises(ValueError, match="must be ints"):
        _encoder(**{hyperparameter: 4.0})


@pytest.mark.parametrize(
    ("omic_type", "width", "out_size"),
    [
        ("expression", EXPR_DIM, OUT_EXPR),
        ("mutation", MUT_DIM, OUT_MUT),
        ("copy_number_variation_gistic", CNV_DIM, OUT_CNV),
    ],
)
def test_encoder_reads_its_own_view_and_output_width(omic_type: str, width: int, out_size: int) -> None:
    encoder = _encoder(omic_type, width)
    views = [torch.randn(3, EXPR_DIM), torch.randn(3, MUT_DIM), torch.randn(3, CNV_DIM)]

    assert encoder.encode[0].out_features == out_size
    assert encoder.training_step([*views, torch.randn(3, 1)], 0).ndim == 0


def test_encoder_rejects_an_unknown_omic_type() -> None:
    with pytest.raises(KeyError):
        _encoder("proteomics", EXPR_DIM)


def test_encoder_structure_forward_and_optimizer() -> None:
    encoder = _encoder()
    assert [type(layer) for layer in encoder.encode] == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout]
    assert (encoder.positive_range, encoder.negative_range) == RANGES
    encoder.eval()
    with torch.no_grad():
        assert encoder(torch.randn(3, EXPR_DIM)).shape == (3, OUT_EXPR)
    optimizer = encoder.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)


def test_encoder_triplet_loss_is_a_non_negative_scalar() -> None:
    loss = _encoder()._compute_loss(torch.randn(4, OUT_EXPR), torch.tensor([0.0, 0.5, 1.0, 2.0]))

    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_regressor_rejects_non_float_hyperparameters() -> None:
    with pytest.raises(ValueError, match="must be floats"):
        _regressor(learning_rate=1)


def test_regressor_freezes_its_encoders_and_optimizes_only_the_head() -> None:
    regressor = _regressor()
    assert isinstance(regressor.encoders, nn.ModuleList)
    assert all(not e.training for e in regressor.encoders)
    assert all(not p.requires_grad for e in regressor.encoders for p in e.parameters())

    regressor.train()
    assert regressor.training
    assert all(not e.training for e in regressor.encoders)

    optimizer = regressor.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adagrad)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in regressor.regressor.parameters()}

    regressor.to(torch.float64)
    assert {p.dtype for e in regressor.encoders for p in e.parameters()} == {torch.float64}


def test_regressor_forward_and_predict() -> None:
    regressor = _regressor()
    regressor.eval()
    with torch.no_grad():
        assert regressor(torch.randn(3, OUT_EXPR + OUT_MUT + OUT_CNV)).shape == (3, 1)
    encoded = regressor._encode_and_concatenate(
        torch.randn(3, EXPR_DIM), torch.randn(3, MUT_DIM), torch.randn(3, CNV_DIM)
    )
    assert encoded.shape == (3, OUT_EXPR + OUT_MUT + OUT_CNV)

    regressor.train()
    preds = regressor.predict(*_omics())
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (N_ENTITIES,)
    assert np.isfinite(preds).all()
    assert regressor.training is False


@pytest.mark.parametrize("step", ["training_step", "validation_step"])
def test_regressor_steps_return_a_scalar_loss(step: str) -> None:
    batch = [torch.randn(3, EXPR_DIM), torch.randn(3, MUT_DIM), torch.randn(3, CNV_DIM), torch.randn(3, 1)]

    assert getattr(_regressor(), step)(batch, 0).ndim == 0


def _fit_module(module, tmp_path, *, with_val: bool = False, **overrides):
    predictor = SuperFELTRPredictor(hyperparameters={**_hpams(), **overrides})
    return predictor._fit_module(module, _omics_batch(tmp_path, with_val=with_val))


@pytest.mark.parametrize("hyperparameter", ["epochs", "mini_batch"])
def test_fit_rejects_non_integer_epochs_and_mini_batch(hyperparameter: str, tmp_path) -> None:
    with pytest.raises(ValueError, match="must be integers"):
        _fit_module(_encoder(), tmp_path, **{hyperparameter: 1.0})


@pytest.mark.parametrize(("with_val", "monitor"), [(False, "train_loss"), (True, "val_loss")])
def test_fit_encoder_monitors_the_available_loss_and_writes_a_checkpoint(with_val, monitor, tmp_path) -> None:
    checkpoint = _fit_module(_encoder(), tmp_path, with_val=with_val)

    assert isinstance(checkpoint, pl.callbacks.ModelCheckpoint)
    assert checkpoint.monitor == monitor
    assert checkpoint.best_model_path.endswith(".ckpt")
    assert str(checkpoint.dirpath).startswith(str(tmp_path))
    assert "version-" in str(checkpoint.dirpath)
    restored = SuperFELTEncoder.load_from_checkpoint(checkpoint.best_model_path, map_location="cpu")
    assert restored.omic_type == "expression"
    assert restored.encode[0].out_features == OUT_EXPR


def test_fit_regressor_writes_a_checkpoint(tmp_path) -> None:
    checkpoint = _fit_module(_regressor(), tmp_path)

    assert checkpoint.best_model_path


def test_fit_runs_on_the_cpu_without_a_default_logger(tmp_path, monkeypatch) -> None:
    real_trainer = pl.Trainer
    seen: list[dict[str, object]] = []

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_trainer(*args, **kwargs)

    monkeypatch.setattr(pl, "Trainer", spy)

    _fit_module(_encoder(), tmp_path / "ckpt")

    (kwargs,) = seen
    assert kwargs["accelerator"] == "cpu"
    assert kwargs["logger"] is False
    assert not (tmp_path / "lightning_logs").exists()
