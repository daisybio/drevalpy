"""Tests for :mod:`drevalpy.cli.experiments.randomization`."""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain

runner = make_runner()

RANDOMIZED = (("SVRC", "gene_expression"), ("SVRC", "methylation"))


@pytest.fixture()
def constructed() -> list[str]:
    """Collect the model names handed to ``construct_model``."""
    return []


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch, constructed: list[str]) -> Recorder:
    """Patch every lazy import of ``randomization_cmd``.

    Args:
        monkeypatch: Fixture used to replace the source-module workers.
        constructed: List that receives every requested model name.

    Returns:
        Recorder standing in for
        :func:`drevalpy.experiment.randomization.randomization`, returning two
        datasets whose ``randomization`` tags drive the output filenames.
    """

    def fake_construct_model(name: str) -> type:
        constructed.append(name)
        return type(f"Stub{name}", (), {})

    recorder = Recorder(return_value=[FakeDataset(randomization=tag) for tag in RANDOMIZED])
    monkeypatch.setattr("drevalpy.models.construct_model", fake_construct_model)
    monkeypatch.setattr("drevalpy.types.data.dataset.Dataset.load", classmethod(lambda cls, path: FakeDataset()))
    patch_worker(monkeypatch, "drevalpy.experiment.randomization", "randomization", recorder)
    return recorder


def _invoke(tmp_path: UPath, *extra: str):
    return runner.invoke(
        app,
        ["experiments", "randomization", "ElasticNet", "TOYv1", str(tmp_path / "randomized"), *extra],
    )


class TestArguments:
    """All three positional arguments are required."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["experiments", "randomization"], id="none"),
            pytest.param(["experiments", "randomization", "ElasticNet"], id="model-only"),
            pytest.param(["experiments", "randomization", "ElasticNet", "TOYv1"], id="missing-output-dir"),
        ],
    )
    def test_missing_positional_arguments_are_usage_errors(self, worker: Recorder, argv: list[str]) -> None:
        result = runner.invoke(app, argv, env=HELP_ENV)

        assert result.exit_code == 2


class TestForwarding:
    """Options map onto :func:`randomization`'s signature."""

    def test_exits_cleanly(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert result.exit_code == 0, result.output

    def test_constructs_the_requested_model(self, worker: Recorder, tmp_path: UPath, constructed: list[str]) -> None:
        _invoke(tmp_path)

        assert constructed == ["ElasticNet"]

    def test_passes_model_class_dataset_and_modes_positionally(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        model_class, dataset, modes = worker.args
        assert model_class.__name__ == "StubElasticNet"
        assert isinstance(dataset, FakeDataset)
        assert modes == ["SVRC"]

    def test_modes_default_to_svrc(self, worker: Recorder, tmp_path: UPath) -> None:
        """``modes=None`` is replaced by ``["SVRC"]`` rather than forwarded."""
        _invoke(tmp_path)

        assert worker.args[2] == ["SVRC"]

    def test_repeated_modes_are_forwarded_in_order(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "-m", "SVCC", "-m", "SVRD")

        assert worker.args[2] == ["SVCC", "SVRD"]

    def test_randomization_type_and_seed_defaults(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs == {"randomization_type": "permutation", "random_state": 42}

    @pytest.mark.parametrize("flag", ["--randomization-type", "-t"], ids=["long", "short"])
    def test_randomization_type_override(self, worker: Recorder, tmp_path: UPath, flag: str) -> None:
        _invoke(tmp_path, flag, "invariant")

        assert worker.kwargs["randomization_type"] == "invariant"

    def test_random_state_override(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--random-state", "7")

        assert worker.kwargs["random_state"] == 7

    def test_non_integer_random_state_is_a_usage_error(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "--random-state", "seed")

        assert result.exit_code == 2


class TestOutput:
    """Filenames come from each dataset's ``randomization`` tag."""

    def test_creates_the_output_directory_including_parents(self, worker: Recorder, tmp_path: UPath) -> None:
        out_dir = tmp_path / "nested" / "randomized"
        runner.invoke(app, ["experiments", "randomization", "ElasticNet", "TOYv1", str(out_dir)])

        assert out_dir.is_dir()

    def test_writes_one_file_per_randomized_dataset(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        written = sorted(p.name for p in (tmp_path / "randomized").glob("*.h5mu"))
        assert written == sorted(f"{mode}:{view}.h5mu" for mode, view in RANDOMIZED)

    def test_untagged_dataset_falls_back_to_unknown(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        monkeypatch.setattr("drevalpy.models.construct_model", lambda name: type("Stub", (), {}))
        monkeypatch.setattr("drevalpy.types.data.dataset.Dataset.load", classmethod(lambda cls, path: FakeDataset()))
        patch_worker(
            monkeypatch,
            "drevalpy.experiment.randomization",
            "randomization",
            Recorder(return_value=[FakeDataset(randomization=None)]),
        )

        _invoke(tmp_path)

        assert (tmp_path / "randomized" / "unknown:0.h5mu").exists()

    def test_echoes_the_dataset_count_and_destination(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        expected = f"Wrote {len(RANDOMIZED)} randomized datasets to {tmp_path / 'randomized'}"
        assert expected in plain(result.output)

    def test_empty_result_still_reports_zero(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        monkeypatch.setattr("drevalpy.models.construct_model", lambda name: type("Stub", (), {}))
        monkeypatch.setattr("drevalpy.types.data.dataset.Dataset.load", classmethod(lambda cls, path: FakeDataset()))
        patch_worker(monkeypatch, "drevalpy.experiment.randomization", "randomization", Recorder(return_value=[]))

        result = _invoke(tmp_path)

        assert "Wrote 0 randomized datasets" in plain(result.output)
