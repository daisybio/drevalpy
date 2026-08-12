"""Tests for the SuperFELTR encoders, regressor, and training entry point.

Two constraints shape the fixtures here: the encoder's ``nn.BatchNorm1d``
requires a batch of more than one row, and ``train_superfeltr_model`` builds its
training loader with ``drop_last=True``, so the pair count must be at least
``2 * mini_batch``. Every fit runs on CPU for a single epoch with
``wandb_project=None`` so no ``WandbLogger`` import is triggered.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from drevalpy.components.predictors.literature.superfeltr.utils import (
    SuperFELTEncoder,
    SuperFELTRegressor,
    train_superfeltr_model,
)

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


@pytest.fixture(autouse=True)
def _trainer_logs_in_tmp_path(monkeypatch, tmp_path) -> None:
    """Keep Lightning's run logs inside ``tmp_path``.

    The trainer is deliberately *not* pinned to CPU: ``SuperFELTRegressor``
    registers its encoders in an ``nn.ModuleList``, so Lightning moves them along
    with the regressor and the fit works on whatever accelerator is present.
    """
    monkeypatch.chdir(tmp_path)


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


@pytest.mark.parametrize(
    "hyperparameter",
    ["dropout_rate", "margin", "learning_rate", "weight_decay"],
)
def test_encoder_rejects_non_float_hyperparameters(hyperparameter: str) -> None:
    with pytest.raises(ValueError, match="must be floats"):
        _encoder(**{hyperparameter: 1})


@pytest.mark.parametrize(
    ("omic_type", "expected"),
    [
        pytest.param("expression", OUT_EXPR, id="expression"),
        pytest.param("mutation", OUT_MUT, id="mutation"),
        pytest.param("copy_number_variation_gistic", OUT_CNV, id="cnv"),
    ],
)
def test_encoder_output_size_branches_on_the_omic_type(omic_type: str, expected: int) -> None:
    encoder = _encoder(omic_type, EXPR_DIM)

    assert encoder.encode[0].out_features == expected


def test_encoder_rejects_an_unknown_omic_type() -> None:
    with pytest.raises(KeyError):
        _encoder("proteomics", EXPR_DIM)


@pytest.mark.parametrize(
    "hyperparameter",
    ["out_dim_expr_encoder", "out_dim_mutation_encoder", "out_dim_cnv_encoder"],
)
def test_encoder_rejects_non_integer_output_sizes(hyperparameter: str) -> None:
    with pytest.raises(ValueError, match="must be ints"):
        _encoder(**{hyperparameter: 4.0})


def test_encoder_places_batch_norm_before_the_activation() -> None:
    encoder = _encoder()

    layer_types = [type(layer) for layer in encoder.encode]

    assert layer_types == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout]


def test_encoder_stores_the_triplet_ranges() -> None:
    encoder = _encoder()

    assert (encoder.positive_range, encoder.negative_range) == RANGES


def test_encoder_forward_projects_to_the_output_size() -> None:
    encoder = _encoder()
    encoder.eval()

    with torch.no_grad():
        encoded = encoder(torch.randn(3, EXPR_DIM))

    assert encoded.shape == (3, OUT_EXPR)


def test_encoder_configures_adam_with_the_requested_learning_rate() -> None:
    encoder = _encoder()

    optimizer = encoder.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("omic_type", "expected_index"),
    [
        pytest.param("expression", 0, id="expression"),
        pytest.param("mutation", 1, id="mutation"),
        pytest.param("copy_number_variation_gistic", 2, id="cnv"),
    ],
)
def test_encoder_selects_its_own_omic_from_the_batch(omic_type: str, expected_index: int) -> None:
    encoder = _encoder(omic_type, EXPR_DIM)
    tensors = (torch.zeros(2, 1), torch.ones(2, 1), torch.full((2, 1), 2.0))

    selected = encoder._get_omic_data(*tensors)

    torch.testing.assert_close(selected, tensors[expected_index])


def test_encoder_omic_selection_rejects_an_unrecognized_type() -> None:
    encoder = _encoder()
    encoder.omic_type = "proteomics"

    with pytest.raises(ValueError, match="not recognized"):
        encoder._get_omic_data(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1))


def test_encoder_triplet_loss_is_a_non_negative_scalar() -> None:
    encoder = _encoder()

    loss = encoder._compute_loss(torch.randn(4, OUT_EXPR), torch.tensor([0.0, 0.5, 1.0, 2.0]))

    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_regressor_rejects_non_float_hyperparameters() -> None:
    with pytest.raises(ValueError, match="must be floats"):
        _regressor(learning_rate=1)


def test_regressor_puts_its_encoders_in_eval_mode() -> None:
    regressor = _regressor()

    assert all(not encoder.training for encoder in regressor.encoders)


def test_regressor_registers_its_encoders_as_submodules() -> None:
    regressor = _regressor()

    assert isinstance(regressor.encoders, nn.ModuleList)
    assert [name for name, _ in regressor.named_children() if name == "encoders"] == ["encoders"]


def test_regressor_moves_its_encoders_with_the_rest_of_the_model() -> None:
    regressor = _regressor()

    regressor.to(torch.float64)

    encoder_dtypes = {parameter.dtype for encoder in regressor.encoders for parameter in encoder.parameters()}
    assert encoder_dtypes == {torch.float64}
    assert next(regressor.regressor.parameters()).dtype == torch.float64


def test_regressor_keeps_encoders_in_eval_mode_when_switched_to_train() -> None:
    regressor = _regressor()

    regressor.train()

    assert regressor.training
    assert all(not encoder.training for encoder in regressor.encoders)


def test_regressor_freezes_its_encoders() -> None:
    regressor = _regressor()

    assert all(not parameter.requires_grad for encoder in regressor.encoders for parameter in encoder.parameters())


def test_regressor_optimizes_only_the_regression_head() -> None:
    regressor = _regressor()

    optimizer = regressor.configure_optimizers()

    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in regressor.regressor.parameters()}


def test_regressor_forward_returns_one_scalar_per_row() -> None:
    regressor = _regressor()
    regressor.eval()

    with torch.no_grad():
        output = regressor(torch.randn(3, OUT_EXPR + OUT_MUT + OUT_CNV))

    assert output.shape == (3, 1)


def test_regressor_concatenates_all_three_encoder_outputs() -> None:
    regressor = _regressor()

    encoded = regressor._encode_and_concatenate(
        torch.randn(3, EXPR_DIM),
        torch.randn(3, MUT_DIM),
        torch.randn(3, CNV_DIM),
    )

    assert encoded.shape == (3, OUT_EXPR + OUT_MUT + OUT_CNV)


def test_regressor_predict_returns_a_flat_numpy_array() -> None:
    regressor = _regressor()
    expr, mut, cnv = _omics()

    preds = regressor.predict(expr, mut, cnv)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (N_ENTITIES,)
    assert np.isfinite(preds).all()


def test_regressor_predict_leaves_the_module_in_eval_mode() -> None:
    regressor = _regressor()
    regressor.train()
    expr, mut, cnv = _omics()

    regressor.predict(expr, mut, cnv)

    assert regressor.training is False


def test_regressor_configures_adagrad() -> None:
    regressor = _regressor()

    optimizer = regressor.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adagrad)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_regressor_training_step_returns_a_scalar_loss() -> None:
    regressor = _regressor()
    batch = [
        torch.randn(3, EXPR_DIM),
        torch.randn(3, MUT_DIM),
        torch.randn(3, CNV_DIM),
        torch.randn(3, 1),
    ]

    loss = regressor.training_step(batch, 0)

    assert loss.ndim == 0


def test_regressor_validation_step_returns_a_scalar_loss() -> None:
    regressor = _regressor()
    batch = [
        torch.randn(3, EXPR_DIM),
        torch.randn(3, MUT_DIM),
        torch.randn(3, CNV_DIM),
        torch.randn(3, 1),
    ]

    loss = regressor.validation_step(batch, 0)

    assert loss.ndim == 0


@pytest.mark.parametrize("hyperparameter", ["epochs", "mini_batch"])
def test_train_rejects_non_integer_epochs_and_mini_batch(hyperparameter: str, tmp_path) -> None:
    response, pair_idx = _pairs()
    expr, mut, cnv = _omics()

    with pytest.raises(ValueError, match="must be integers"):
        train_superfeltr_model(
            model=_encoder(),
            hpams=_hpams(**{hyperparameter: 1.0}),
            gene_expression=expr,
            mutations=mut,
            copy_number=cnv,
            response=response,
            pair_idx=pair_idx,
            model_checkpoint_dir=tmp_path,
            wandb_project=None,
        )


def test_train_requires_the_other_validation_omics_alongside_expression(tmp_path) -> None:
    response, pair_idx = _pairs()
    expr, mut, cnv = _omics()

    with pytest.raises(ValueError, match="val_mutations and val_copy_number are required"):
        train_superfeltr_model(
            model=_encoder(),
            hpams=_hpams(),
            gene_expression=expr,
            mutations=mut,
            copy_number=cnv,
            response=response,
            pair_idx=pair_idx,
            val_gene_expression=expr,
            val_response=response,
            val_pair_idx=pair_idx,
            model_checkpoint_dir=tmp_path,
            wandb_project=None,
        )


def test_train_encoder_without_validation_monitors_the_training_loss(tmp_path) -> None:
    response, pair_idx = _pairs()
    expr, mut, cnv = _omics()

    checkpoint = train_superfeltr_model(
        model=_encoder(),
        hpams=_hpams(),
        gene_expression=expr,
        mutations=mut,
        copy_number=cnv,
        response=response,
        pair_idx=pair_idx,
        patience=1,
        model_checkpoint_dir=tmp_path,
        wandb_project=None,
    )

    assert isinstance(checkpoint, pl.callbacks.ModelCheckpoint)
    assert checkpoint.monitor == "train_loss"


def test_train_encoder_with_validation_monitors_the_validation_loss(tmp_path) -> None:
    response, pair_idx = _pairs()
    expr, mut, cnv = _omics()

    checkpoint = train_superfeltr_model(
        model=_encoder(),
        hpams=_hpams(),
        gene_expression=expr,
        mutations=mut,
        copy_number=cnv,
        response=response,
        pair_idx=pair_idx,
        val_gene_expression=expr,
        val_mutations=mut,
        val_copy_number=cnv,
        val_response=response,
        val_pair_idx=pair_idx,
        patience=1,
        model_checkpoint_dir=tmp_path,
        wandb_project=None,
    )

    assert checkpoint.monitor == "val_loss"


def test_train_writes_a_checkpoint_under_a_unique_versioned_directory(tmp_path) -> None:
    response, pair_idx = _pairs()
    expr, mut, cnv = _omics()

    checkpoint = train_superfeltr_model(
        model=_encoder(),
        hpams=_hpams(),
        gene_expression=expr,
        mutations=mut,
        copy_number=cnv,
        response=response,
        pair_idx=pair_idx,
        patience=1,
        model_checkpoint_dir=tmp_path,
        wandb_project=None,
    )

    assert checkpoint.best_model_path
    assert checkpoint.best_model_path.endswith(".ckpt")
    assert str(checkpoint.dirpath).startswith(str(tmp_path))
    assert "version-" in str(checkpoint.dirpath)


def test_train_regressor_accepts_a_two_dimensional_response(tmp_path) -> None:
    response, pair_idx = _pairs()
    expr, mut, cnv = _omics()

    checkpoint = train_superfeltr_model(
        model=_regressor(),
        hpams=_hpams(),
        gene_expression=expr,
        mutations=mut,
        copy_number=cnv,
        response=response.reshape(-1, 1),
        pair_idx=pair_idx,
        patience=1,
        model_checkpoint_dir=tmp_path,
        wandb_project=None,
    )

    assert checkpoint.best_model_path


def test_trained_encoder_checkpoint_is_loadable(tmp_path) -> None:
    response, pair_idx = _pairs()
    expr, mut, cnv = _omics()
    checkpoint = train_superfeltr_model(
        model=_encoder(),
        hpams=_hpams(),
        gene_expression=expr,
        mutations=mut,
        copy_number=cnv,
        response=response,
        pair_idx=pair_idx,
        patience=1,
        model_checkpoint_dir=tmp_path,
        wandb_project=None,
    )

    restored = SuperFELTEncoder.load_from_checkpoint(checkpoint.best_model_path, map_location="cpu")

    assert restored.omic_type == "expression"
    assert restored.encode[0].out_features == OUT_EXPR
