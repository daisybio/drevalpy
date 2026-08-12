"""Tests for :mod:`drevalpy.cli.data.load`, the ``drevalpy data load`` command.

:func:`drevalpy.data.datasets.load.load` resolves registered dataset names by
downloading them, so it is patched in every test here; the download path itself
belongs to ``tests/data/datasets``.
"""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain

runner = make_runner()


@pytest.fixture()
def dataset() -> FakeDataset:
    """The dataset stub the patched loader returns."""
    return FakeDataset(name="TOYv1")


@pytest.fixture()
def loader(monkeypatch: pytest.MonkeyPatch, dataset: FakeDataset) -> Recorder:
    """Patch :func:`drevalpy.data.datasets.load.load`.

    Args:
        monkeypatch: Fixture used to replace the source-module worker.
        dataset: Stub the loader hands back.

    Returns:
        Recorder standing in for the dataset loader.
    """
    recorder = Recorder(return_value=dataset)
    patch_worker(monkeypatch, "drevalpy.data.datasets.load", "load", recorder)
    patch_worker(monkeypatch, "drevalpy.data", "load", recorder)
    return recorder


class TestArguments:
    """Both positional arguments are required."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["data", "load"], id="none"),
            pytest.param(["data", "load", "TOYv1"], id="missing-output"),
        ],
    )
    def test_missing_positional_arguments_are_usage_errors(self, loader: Recorder, argv: list[str]) -> None:
        result = runner.invoke(app, argv, env=HELP_ENV)

        assert result.exit_code == 2

    def test_nothing_is_loaded_on_a_usage_error(self, loader: Recorder) -> None:
        runner.invoke(app, ["data", "load", "TOYv1"], env=HELP_ENV)

        assert loader.call_count == 0


class TestLoading:
    """The name argument is forwarded verbatim and the mdata is written out."""

    def test_exits_cleanly(self, loader: Recorder, tmp_path: UPath) -> None:
        result = runner.invoke(app, ["data", "load", "TOYv1", str(tmp_path / "out.h5mu")])

        assert result.exit_code == 0, result.output

    def test_forwards_the_dataset_name(self, loader: Recorder, tmp_path: UPath) -> None:
        runner.invoke(app, ["data", "load", "TOYv1", str(tmp_path / "out.h5mu")])

        assert loader.args == ("TOYv1",)

    def test_forwards_a_path_unchanged(self, loader: Recorder, tmp_path: UPath) -> None:
        """A ``.h5mu`` path is a valid ``name``; the command must not rewrite it."""
        source = tmp_path / "local.h5mu"

        runner.invoke(app, ["data", "load", str(source), str(tmp_path / "out.h5mu")])

        assert loader.args == (str(source),)

    def test_creates_the_output_parent_directory(self, loader: Recorder, tmp_path: UPath) -> None:
        out = tmp_path / "nested" / "deeper" / "out.h5mu"
        runner.invoke(app, ["data", "load", "TOYv1", str(out)])

        assert out.parent.is_dir()

    def test_writes_to_the_requested_path(self, loader: Recorder, dataset: FakeDataset, tmp_path: UPath) -> None:
        out = tmp_path / "out.h5mu"
        runner.invoke(app, ["data", "load", "TOYv1", str(out)])

        assert dataset.mdata.written == [str(out)]

    def test_echoes_the_dataset_name_and_destination(self, loader: Recorder, tmp_path: UPath) -> None:
        out = tmp_path / "out.h5mu"
        result = runner.invoke(app, ["data", "load", "TOYv1", str(out)])

        assert f"Wrote TOYv1 to {out}" in plain(result.output)


class TestLoaderFailure:
    """An unresolvable dataset name surfaces rather than writing a broken file."""

    def test_loader_error_propagates(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        def boom(name: str) -> FakeDataset:
            raise KeyError(name)

        patch_worker(monkeypatch, "drevalpy.data.datasets.load", "load", boom)
        patch_worker(monkeypatch, "drevalpy.data", "load", boom)

        result = runner.invoke(app, ["data", "load", "NoSuchDataset", str(tmp_path / "out.h5mu")])

        assert isinstance(result.exception, KeyError)

    def test_no_output_file_is_written(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        def boom(name: str) -> FakeDataset:
            raise KeyError(name)

        patch_worker(monkeypatch, "drevalpy.data.datasets.load", "load", boom)
        patch_worker(monkeypatch, "drevalpy.data", "load", boom)
        out = tmp_path / "out.h5mu"

        runner.invoke(app, ["data", "load", "NoSuchDataset", str(out)])

        assert not out.exists()
