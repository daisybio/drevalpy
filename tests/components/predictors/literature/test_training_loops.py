"""Tests for the shared checkpointed early-stopping loop and the eval-time pair inference helper."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from drevalpy.components.predictors.literature._early_stopping import (
    EarlyStoppingRun,
    run_epoch,
    train_with_early_stopping,
)
from drevalpy.components.predictors.literature._pair_predict import (
    PairEvalSpec,
    concatenated_forward,
    predict_pairs,
    require_drug_pair_idx,
)
from drevalpy.experiment._tracking import tracking_run
from drevalpy.types.data.batch.feature_block import numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests._fake_wandb import install_fake_wandb
from tests.components.predictors.literature._helpers import two_by_two_batch

pytestmark = pytest.mark.slow


class _CountingModel:
    """A one-parameter torch module that records how it was driven."""

    def __init__(self) -> None:
        import torch
        from torch import nn

        self._module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self._module.weight.fill_(0.0)
        self.saved_weights: list[float] = []
        self.reload_count = 0
        self.devices: list[object] = []

    def set_weight(self, value: float) -> None:
        import torch

        with torch.no_grad():
            self._module.weight.fill_(value)

    def weight(self) -> float:
        return float(self._module.weight.item())

    def state_dict(self):
        self.saved_weights.append(self.weight())
        return self._module.state_dict()

    def load_state_dict(self, state):
        self.reload_count += 1
        return self._module.load_state_dict(state)

    def to(self, device):
        self.devices.append(device)
        return self


def _run(tmp_path, *, epochs: int, patience: int, verbose: bool = False) -> EarlyStoppingRun:
    return EarlyStoppingRun(
        epochs=epochs,
        patience=patience,
        checkpoint_dir=tmp_path,
        model_name="Scripted",
        verbose=verbose,
    )


def _drive(model, run: EarlyStoppingRun, val_losses: list[float]) -> list[int]:
    import torch

    reached: list[int] = []

    def train_epoch() -> float:
        epoch = len(reached)
        reached.append(epoch)
        model.set_weight(float(epoch))
        return 1.0

    def val_epoch() -> float:
        return val_losses[len(reached) - 1]

    train_with_early_stopping(model, run, train_epoch, val_epoch, torch.device("cpu"))
    return reached


@pytest.mark.parametrize(
    ("epochs", "patience", "losses", "reached", "saved"),
    [
        (4, 2, [4.0, 3.0, 2.0, 1.0], [0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0]),
        (10, 2, [1.0, 2.0, 3.0, 4.0], [0, 1, 2], [0.0]),
        (10, 2, [5.0, 6.0, 4.0, 7.0, 8.0], [0, 1, 2, 3, 4], [0.0, 2.0]),
    ],
    ids=["improving", "patience-stops", "improvement-resets-patience"],
)
def test_patience_counts_consecutive_non_improving_epochs(tmp_path, epochs, patience, losses, reached, saved) -> None:
    model = _CountingModel()

    assert _drive(model, _run(tmp_path, epochs=epochs, patience=patience), losses) == reached
    assert model.saved_weights == saved


def test_the_best_epochs_weights_are_reloaded_and_the_model_moved_to_the_device(tmp_path) -> None:
    import torch

    model = _CountingModel()

    _drive(model, _run(tmp_path, epochs=4, patience=4), [5.0, 6.0, 4.0, 7.0])

    assert model.reload_count == 1
    assert model.weight() == pytest.approx(2.0)
    assert model.devices == [torch.device("cpu")]


def test_checkpoints_are_written_per_model_and_runs_do_not_collide(tmp_path) -> None:
    directory = tmp_path / "nested" / "checkpoints"

    for _ in range(2):
        _drive(_CountingModel(), _run(directory, epochs=1, patience=1), [1.0])

    written = list(directory.glob("*.pth"))
    assert len(written) == 2
    assert all(p.name.startswith("version-") and p.name.endswith("_best_Scripted_model.pth") for p in written)


def test_progress_is_silent_unless_the_run_is_verbose(tmp_path, caplog) -> None:
    with caplog.at_level(logging.INFO):
        _drive(_CountingModel(), _run(tmp_path, epochs=1, patience=1), [1.0])
    assert caplog.messages == []

    with caplog.at_level(logging.INFO):
        _drive(_CountingModel(), _run(tmp_path, epochs=10, patience=1, verbose=True), [1.0, 2.0])
    output = "\n".join(caplog.messages)
    assert "Scripted: Epoch [1/10] Training Loss:" in output
    assert "Scripted: Epoch [1/10] Validation Loss:" in output
    assert "Scripted: Reloading the best model" in output
    assert "Scripted: Early stopping triggered at epoch 2" in caplog.messages


def test_each_epoch_is_logged_to_the_active_tracking_run_only(tmp_path, monkeypatch) -> None:
    fake = install_fake_wandb(monkeypatch)
    _drive(_CountingModel(), _run(tmp_path, epochs=2, patience=2), [2.0, 1.0])
    assert fake.runs == []

    with tracking_run("proj", name="r"):
        _drive(_CountingModel(), _run(tmp_path, epochs=2, patience=2), [2.0, 1.0])

    assert fake.runs[0].logged == [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 2.0},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 1.0},
    ]


def _linear_model_and_loader():
    import torch
    from torch import nn

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    loader = [(torch.tensor([[1.0]]), torch.tensor([[2.0]])), (torch.tensor([[2.0]]), torch.tensor([[4.0]]))]
    return model, loader


def _squared_error(model):
    def batch_loss(batch):
        inputs, targets = batch
        return ((model(inputs) - targets) ** 2).mean()

    return batch_loss


def test_run_epoch_in_train_mode_updates_the_parameters_and_returns_the_mean_loss() -> None:
    import torch

    model, loader = _linear_model_and_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.eval()

    mean_loss = run_epoch(model, loader, _squared_error(model), optimizer)

    assert model.training
    assert mean_loss == pytest.approx((4.0 + (0.8 - 4.0) ** 2) / 2)
    assert model.weight.item() != 0.0


def test_run_epoch_in_eval_mode_leaves_parameters_and_gradients_untouched() -> None:
    model, loader = _linear_model_and_loader()
    model.train()
    grad_modes: list[bool] = []

    def batch_loss(batch):
        import torch

        grad_modes.append(torch.is_grad_enabled())
        return _squared_error(model)(batch)

    mean_loss = run_epoch(model, loader, batch_loss, None)

    assert not model.training
    assert mean_loss == pytest.approx((4.0 + 16.0) / 2)
    assert model.weight.item() == 0.0
    assert model.weight.grad is None
    assert grad_modes == [False, False]


def test_run_epoch_over_an_empty_loader_returns_zero() -> None:
    model, _ = _linear_model_and_loader()

    assert run_epoch(model, [], _squared_error(model), None) == 0.0


SAMPLE_VALUES = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
DRUG_VALUES = np.array([[10.0], [20.0]], dtype=np.float32)


def _batch(*, with_drug_pair_idx: bool = True) -> ModelInputBatch:
    return two_by_two_batch(
        sample_blocks={"view": numeric_feature_block(SAMPLE_VALUES)},
        drug_blocks={"view": numeric_feature_block(DRUG_VALUES)},
        drug_pair_idx=np.array([0, 1, 0, 1]) if with_drug_pair_idx else None,
    )


def _spec(batch_size: int = 2):
    import torch

    return PairEvalSpec(
        sample_blocks=(SAMPLE_VALUES,),
        drug_blocks=(DRUG_VALUES,),
        batch_size=batch_size,
        device=torch.device("cpu"),
    )


class _SumModel:
    def __init__(self) -> None:
        self.training = True
        self.seen_shapes: list[tuple[int, ...]] = []

    def eval(self) -> None:
        self.training = False

    def __call__(self, *tensors):
        import torch

        self.seen_shapes.append(tuple(tensors[0].shape))
        return torch.stack([tensor.sum(dim=1) for tensor in tensors]).sum(dim=0)


class _WidthModel:
    def eval(self) -> None:
        return None

    def __call__(self, tensor):
        import torch

        return torch.full((tensor.shape[0],), float(tensor.shape[1]))


def test_require_drug_pair_idx_passes_arrays_and_rejects_none() -> None:
    indices = np.array([0, 1])
    assert require_drug_pair_idx(indices) is indices
    with pytest.raises(RuntimeError, match="drug_pair_idx is required for this predictor"):
        require_drug_pair_idx(None)


def test_predict_pairs_returns_float64_in_pair_order_and_switches_to_eval() -> None:
    model = _SumModel()

    predictions = predict_pairs(model, _batch(), _spec())

    np.testing.assert_allclose(predictions, [13.0, 23.0, 17.0, 27.0])
    assert predictions.dtype == np.float64
    assert model.training is False


def test_a_batch_size_below_the_pair_count_still_covers_every_pair() -> None:
    model = _SumModel()

    assert len(predict_pairs(model, _batch(), _spec(batch_size=3))) == 4
    assert [shape[0] for shape in model.seen_shapes] == [3, 1]


def test_a_batch_without_drug_pair_indices_is_rejected() -> None:
    with pytest.raises(ValueError, match="drug feature result requires pair indices"):
        _batch(with_drug_pair_idx=False)


def test_a_multi_element_output_row_is_flattened() -> None:
    class _ColumnModel(_SumModel):
        def __call__(self, *tensors):
            return super().__call__(*tensors).reshape(-1, 1)

    assert predict_pairs(_ColumnModel(), _batch(), _spec()).shape == (4,)


def test_a_batch_with_no_pairs_yields_an_empty_float64_array() -> None:
    empty = two_by_two_batch(
        response=ResponseBatch(
            response=np.empty(0), sample_ids=np.empty(0, dtype=str), drug_ids=np.empty(0, dtype=str)
        ),
        sample_blocks={"view": numeric_feature_block(SAMPLE_VALUES)},
        drug_blocks={"view": numeric_feature_block(DRUG_VALUES)},
        sample_pair_idx=np.empty(0, dtype=np.intp),
        drug_pair_idx=np.empty(0, dtype=np.intp),
    )

    predictions = predict_pairs(_SumModel(), empty, _spec())

    assert predictions.shape == (0,)
    assert predictions.dtype == np.float64


def test_concatenated_forward_joins_blocks_feature_wise_otherwise_they_stay_separate() -> None:
    predictions = predict_pairs(_WidthModel(), _batch(), _spec(), forward=concatenated_forward(_WidthModel()))
    np.testing.assert_allclose(predictions, np.full(4, 3.0))

    model = _SumModel()
    predict_pairs(model, _batch(), _spec(batch_size=4))
    assert model.seen_shapes == [(4, 2)]
