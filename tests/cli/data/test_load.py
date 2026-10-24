"""Tests for :mod:`drevalpy.cli.data.load`; ``drevalpy.data.load`` is patched (downloads belong to tests/data)."""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.errors import UnknownNameError
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain

runner = make_runner()


def _patch_loader(monkeypatch: pytest.MonkeyPatch, worker) -> None:
    patch_worker(monkeypatch, "drevalpy.data.datasets._load", "load", worker)
    patch_worker(monkeypatch, "drevalpy.data", "load", worker)


@pytest.fixture()
def dataset() -> FakeDataset:
    return FakeDataset(name="TOYv1")


@pytest.fixture()
def loader(monkeypatch: pytest.MonkeyPatch, dataset: FakeDataset) -> Recorder:
    recorder = Recorder(return_value=dataset)
    _patch_loader(monkeypatch, recorder)
    return recorder


@pytest.mark.parametrize("argv", [["data", "load"], ["data", "load", "TOYv1"]], ids=["none", "missing-output"])
def test_missing_positional_arguments_are_usage_errors_and_load_nothing(loader: Recorder, argv: list[str]) -> None:
    assert runner.invoke(app, argv, env=HELP_ENV).exit_code == 2
    assert loader.call_count == 0


def test_loads_by_name_and_writes_to_the_requested_path(
    loader: Recorder, dataset: FakeDataset, tmp_path: UPath
) -> None:
    out = tmp_path / "nested" / "deeper" / "out.h5mu"

    result = runner.invoke(app, ["data", "load", "TOYv1", str(out)])

    assert result.exit_code == 0, result.output
    assert loader.args == ("TOYv1",)
    assert out.parent.is_dir()
    assert dataset.mdata.written == [str(out)]
    assert f"Wrote TOYv1 to {out}" in plain(result.output)


def test_a_path_is_forwarded_unchanged(loader: Recorder, tmp_path: UPath) -> None:
    source = tmp_path / "local.h5mu"

    runner.invoke(app, ["data", "load", str(source), str(tmp_path / "out.h5mu")])

    assert loader.args == (str(source),)


def test_a_loader_error_is_reported_without_a_traceback_and_writes_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
) -> None:
    def boom(name: str) -> FakeDataset:
        raise UnknownNameError(f"Unknown dataset {name!r}")

    _patch_loader(monkeypatch, boom)
    out = tmp_path / "out.h5mu"

    result = runner.invoke(app, ["data", "load", "NoSuchDataset", str(out)])

    assert result.exit_code == 1
    assert plain(result.output).strip() == "Error: Unknown dataset 'NoSuchDataset'"
    assert not out.exists()
