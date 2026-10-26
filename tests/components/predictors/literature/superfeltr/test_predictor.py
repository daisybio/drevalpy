"""Tests for the SuperFELTR predictor, encoders, regressor and training entry point."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit
from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.components.predictors.literature.superfeltr.utils import (
    SuperFELTEncoder,
    SuperFELTRegressor,
    train_superfeltr_model,
)
from drevalpy.registry.predictor import get as get_predictor
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_state_round_trip,
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
    def test_fits_triplet_batches_and_round_trips_state_without_feature_names(self, seed: int, tmp_path) -> None:
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
        fitted = predictor._models["d1"]
        assert fitted.best_checkpoint is not None
        assert np.isfinite(float(fitted.best_checkpoint.best_model_score))
        assert_training_checkpoint_score("superfeltr", seed, fitted.best_checkpoint.best_model_score)
        expected = finite_predictions(predictor, batch)
        assert_training_predictions("superfeltr", seed, expected)
        assert_frozen_reference("superfeltr", seed)

        assert_state_round_trip(predictor, batch, expected)


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


def _pairs() -> tuple[np.ndarray, np.ndarray]:
    response = np.linspace(0.0, 1.0, N_PAIRS, dtype=np.float32)
    pair_idx = np.arange(N_PAIRS, dtype=np.int64) % N_ENTITIES
    return response, pair_idx


def _split(response: np.ndarray | None = None) -> OmicsSplit:
    expr, mut, cnv = _omics()
    default_response, pair_idx = _pairs()
    return OmicsSplit(
        gene_expression=expr,
        mutations=mut,
        copy_number=cnv,
        response=default_response if response is None else response,
        pair_idx=pair_idx,
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
    ("omic_type", "out_size", "index"),
    [("expression", OUT_EXPR, 0), ("mutation", OUT_MUT, 1), ("copy_number_variation_gistic", OUT_CNV, 2)],
)
def test_encoder_branches_on_the_omic_type(omic_type: str, out_size: int, index: int) -> None:
    encoder = _encoder(omic_type, EXPR_DIM)
    tensors = (torch.zeros(2, 1), torch.ones(2, 1), torch.full((2, 1), 2.0))

    assert encoder.encode[0].out_features == out_size
    torch.testing.assert_close(encoder._get_omic_data(*tensors), tensors[index])


def test_encoder_rejects_an_unknown_omic_type() -> None:
    with pytest.raises(KeyError):
        _encoder("proteomics", EXPR_DIM)
    encoder = _encoder()
    encoder.omic_type = "proteomics"
    with pytest.raises(ValueError, match="not recognized"):
        encoder._get_omic_data(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1))


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


@pytest.mark.parametrize("hyperparameter", ["epochs", "mini_batch"])
def test_train_rejects_non_integer_epochs_and_mini_batch(hyperparameter: str, tmp_path) -> None:
    with pytest.raises(ValueError, match="must be integers"):
        train_superfeltr_model(
            model=_encoder(), hpams=_hpams(**{hyperparameter: 1.0}), train=_split(), model_checkpoint_dir=tmp_path
        )


@pytest.mark.parametrize(("with_val", "monitor"), [(False, "train_loss"), (True, "val_loss")])
def test_train_encoder_monitors_the_available_loss_and_writes_a_checkpoint(with_val, monitor, tmp_path) -> None:
    checkpoint = train_superfeltr_model(
        model=_encoder(),
        hpams=_hpams(),
        train=_split(),
        val=_split() if with_val else None,
        patience=1,
        model_checkpoint_dir=tmp_path,
    )

    assert isinstance(checkpoint, pl.callbacks.ModelCheckpoint)
    assert checkpoint.monitor == monitor
    assert checkpoint.best_model_path.endswith(".ckpt")
    assert str(checkpoint.dirpath).startswith(str(tmp_path))
    assert "version-" in str(checkpoint.dirpath)
    restored = SuperFELTEncoder.load_from_checkpoint(checkpoint.best_model_path, map_location="cpu")
    assert restored.omic_type == "expression"
    assert restored.encode[0].out_features == OUT_EXPR


def test_train_regressor_accepts_a_two_dimensional_response(tmp_path) -> None:
    response, _ = _pairs()

    checkpoint = train_superfeltr_model(
        model=_regressor(),
        hpams=_hpams(),
        train=_split(response=response.reshape(-1, 1)),
        patience=1,
        model_checkpoint_dir=tmp_path,
    )

    assert checkpoint.best_model_path


@pytest.mark.parametrize("make_model", [_encoder, _regressor])
def test_train_without_checkpoint_dir_uses_a_temporary_directory(make_model, tmp_path) -> None:
    checkpoint = train_superfeltr_model(
        model=make_model(), hpams=_hpams(), train=_split(), patience=1, model_checkpoint_dir=None
    )

    assert checkpoint.best_model_path
    assert not Path(checkpoint.best_model_path).exists()
    assert not (tmp_path / "superfeltr_checkpoints").exists()


def test_train_runs_on_the_cpu_without_a_default_logger(tmp_path, monkeypatch) -> None:
    real_trainer = pl.Trainer
    seen: list[dict[str, object]] = []

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_trainer(*args, **kwargs)

    monkeypatch.setattr(pl, "Trainer", spy)

    train_superfeltr_model(
        model=_encoder(), hpams=_hpams(), train=_split(), patience=1, model_checkpoint_dir=tmp_path / "ckpt"
    )

    (kwargs,) = seen
    assert kwargs["accelerator"] == "cpu"
    assert kwargs["logger"] is False
    assert not (tmp_path / "lightning_logs").exists()
