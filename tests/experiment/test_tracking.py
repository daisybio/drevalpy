"""A fake ``wandb`` in ``sys.modules`` records what tracking does; with no project nothing is imported."""

from __future__ import annotations

import sys

import pytest

from drevalpy.experiment import _tracking
from tests._fake_wandb import FakeWandb, install_fake_wandb
from tests._import_shims import block_imports


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> FakeWandb:
    return install_fake_wandb(monkeypatch)


def test_importing_the_module_does_not_import_wandb() -> None:
    assert "wandb" not in vars(_tracking)


def _no_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    block_imports(monkeypatch, "wandb")


def test_without_a_project_nothing_is_imported_or_started(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_wandb(monkeypatch)
    with _tracking.tracking_run(None, name="x", config={"a": 1}, tags=["t"]):
        _tracking.log_summary({"RMSE": 1.0})
        _tracking.log_metrics({"loss": 1.0})
    _tracking.require_tracking(None)
    assert _tracking.active_run() is None
    assert "wandb" not in sys.modules


def test_a_missing_wandb_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_wandb(monkeypatch)
    with pytest.raises(ImportError, match="pip install wandb"), _tracking.tracking_run("proj", name="x"):
        pass
    with pytest.raises(ImportError, match="pip install wandb"):
        _tracking.require_tracking("proj")


def test_a_run_is_started_with_its_settings_and_finished_on_exit(fake_wandb: FakeWandb) -> None:
    with _tracking.tracking_run("proj", name="run1", config={"lr": 0.1}, tags=["a", "b"]):
        assert fake_wandb.run.finished is False
        assert _tracking.active_run() is fake_wandb.run
    (run,) = fake_wandb.runs
    assert (run.project, run.name, run.config, run.tags) == ("proj", "run1", {"lr": 0.1}, ["a", "b"])
    assert run.finished
    assert run.exit_code is None
    assert fake_wandb.run is None


def test_an_exception_marks_the_run_failed_and_propagates(fake_wandb: FakeWandb) -> None:
    with pytest.raises(RuntimeError, match="boom"), _tracking.tracking_run("proj", name="run1"):
        raise RuntimeError("boom")
    assert fake_wandb.runs[0].finished
    assert fake_wandb.runs[0].exit_code == 1


def test_a_previous_run_is_finished_before_the_next_starts(fake_wandb: FakeWandb) -> None:
    fake_wandb.init(project="proj", name="stale", config={}, tags=[])
    with _tracking.tracking_run("proj", name="fresh"):
        pass
    assert fake_wandb.runs[0].finished
    assert fake_wandb.run_names() == ["stale", "fresh"]


def test_active_run_is_none_without_wandb_or_an_open_run(
    monkeypatch: pytest.MonkeyPatch, fake_wandb: FakeWandb
) -> None:
    assert _tracking.active_run() is None
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    assert _tracking.active_run() is None


def test_logging_goes_to_the_active_run_and_is_a_no_op_without_one(fake_wandb: FakeWandb) -> None:
    _tracking.log_summary({"test_RMSE": 1.5})
    _tracking.log_metrics({"epoch": 1})
    assert fake_wandb.runs == []

    with _tracking.tracking_run("proj", name="r"):
        _tracking.log_summary({"test_RMSE": 1.5})
        _tracking.log_metrics({"epoch": 1, "val_loss": 0.5})
        _tracking.log_metrics({"epoch": 2, "val_loss": 0.4})
    assert fake_wandb.runs[0].summary == {"test_RMSE": 1.5}
    assert fake_wandb.runs[0].logged == [{"epoch": 1, "val_loss": 0.5}, {"epoch": 2, "val_loss": 0.4}]
