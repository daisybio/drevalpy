"""Tests for the shared checkpointed early-stopping loop.

``literature/_early_stopping.py`` replaced the loop PharmaFormer and DIPK each ran by
hand around their own epoch functions. The behaviour that matters is when it stops:
patience counts *consecutive* non-improving epochs and resets on any improvement, and
whatever weights were best get reloaded at the end - including the case where the
first epoch was the best one and every later epoch was worse.

The epoch callables here return scripted losses rather than training anything, so the
loop's decisions are observable without a real fit.
"""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.literature._early_stopping import (
    EarlyStoppingRun,
    train_with_early_stopping,
)

#: ``save_torch_payload``/``load_state_dict`` round-trip real torch checkpoints.
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
    """Run the loop over scripted validation losses, recording the epochs reached.

    Each epoch stamps the model's weight with its index, so the reloaded weight
    identifies which epoch's checkpoint won.

    :param model: The scripted model.
    :param run: Run configuration.
    :param val_losses: One validation loss per epoch.
    :returns: The epoch indices the loop actually reached.
    """
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


def test_a_monotonically_improving_run_uses_the_whole_epoch_budget(tmp_path) -> None:
    model = _CountingModel()

    reached = _drive(model, _run(tmp_path, epochs=4, patience=2), [4.0, 3.0, 2.0, 1.0])

    assert reached == [0, 1, 2, 3]
    assert model.saved_weights == [0.0, 1.0, 2.0, 3.0]


def test_patience_stops_the_run_early(tmp_path) -> None:
    model = _CountingModel()

    reached = _drive(model, _run(tmp_path, epochs=10, patience=2), [1.0, 2.0, 3.0, 4.0])

    # Epoch 0 is best; epochs 1 and 2 exhaust patience.
    assert reached == [0, 1, 2]


def test_an_improvement_resets_the_patience_counter(tmp_path) -> None:
    model = _CountingModel()

    reached = _drive(model, _run(tmp_path, epochs=10, patience=2), [5.0, 6.0, 4.0, 7.0, 8.0])

    # Epoch 1 does not improve, epoch 2 does and resets, epochs 3-4 then exhaust patience.
    assert reached == [0, 1, 2, 3, 4]


def test_only_improving_epochs_are_checkpointed(tmp_path) -> None:
    model = _CountingModel()

    _drive(model, _run(tmp_path, epochs=4, patience=4), [5.0, 6.0, 4.0, 7.0])

    assert model.saved_weights == [0.0, 2.0]


def test_the_best_epochs_weights_are_reloaded_at_the_end(tmp_path) -> None:
    model = _CountingModel()

    _drive(model, _run(tmp_path, epochs=4, patience=4), [5.0, 6.0, 4.0, 7.0])

    assert model.reload_count == 1
    assert model.weight() == pytest.approx(2.0)


def test_the_model_is_moved_back_onto_the_target_device(tmp_path) -> None:
    import torch

    model = _CountingModel()

    _drive(model, _run(tmp_path, epochs=1, patience=1), [1.0])

    assert model.devices == [torch.device("cpu")]


def test_the_checkpoint_directory_is_created_and_named_per_model(tmp_path) -> None:
    model = _CountingModel()
    directory = tmp_path / "nested" / "checkpoints"

    _drive(model, _run(directory, epochs=1, patience=1), [1.0])

    written = list(directory.glob("*.pth"))
    assert len(written) == 1
    assert written[0].name.startswith("version-")
    assert written[0].name.endswith("_best_Scripted_model.pth")


def test_two_runs_in_one_directory_do_not_collide(tmp_path) -> None:
    """The randomised ``version-`` prefix is what keeps concurrent fits apart."""
    for _ in range(2):
        _drive(_CountingModel(), _run(tmp_path, epochs=1, patience=1), [1.0])

    assert len(list(tmp_path.glob("*.pth"))) == 2


def test_progress_is_silent_unless_the_run_is_verbose(tmp_path, capsys) -> None:
    _drive(_CountingModel(), _run(tmp_path, epochs=1, patience=1), [1.0])

    assert capsys.readouterr().out == ""


def test_a_verbose_run_reports_both_losses_and_the_reload(tmp_path, capsys) -> None:
    _drive(_CountingModel(), _run(tmp_path, epochs=1, patience=1, verbose=True), [1.0])

    output = capsys.readouterr().out
    assert "Scripted: Epoch [1/1] Training Loss:" in output
    assert "Scripted: Epoch [1/1] Validation Loss:" in output
    assert "Scripted: Reloading the best model" in output


def test_a_verbose_run_announces_the_early_stop(tmp_path, capsys) -> None:
    _drive(_CountingModel(), _run(tmp_path, epochs=10, patience=1, verbose=True), [1.0, 2.0])

    assert "Scripted: Early stopping triggered at epoch 2" in capsys.readouterr().out
