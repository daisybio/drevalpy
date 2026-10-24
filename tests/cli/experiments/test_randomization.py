"""Tests for :mod:`drevalpy.cli.experiments.randomization`."""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.cli.main import app
from tests.cli._helpers import (
    HELP_ENV,
    FakeDataset,
    Recorder,
    make_runner,
    patch_worker,
    plain,
    write_synthetic_fold,
)

runner = make_runner()

RANDOMIZED = (("SVRC", "gene_expression"), ("SVRC", "methylation"))


@pytest.fixture()
def constructed() -> list[str]:
    return []


@pytest.fixture()
def loader(monkeypatch: pytest.MonkeyPatch) -> Recorder:

    recorder = Recorder(return_value=FakeDataset())
    patch_worker(monkeypatch, "drevalpy.data.datasets._load", "load", recorder)
    patch_worker(monkeypatch, "drevalpy.data", "load", recorder)
    return recorder


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch, constructed: list[str], loader: Recorder) -> Recorder:

    def fake_construct_model(name: str) -> type:
        constructed.append(name)
        return type(f"Stub{name}", (), {})

    recorder = Recorder(return_value=[FakeDataset(randomization=tag) for tag in RANDOMIZED])
    monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", fake_construct_model)
    patch_worker(monkeypatch, "drevalpy.experiment._randomization", "randomization", recorder)
    return recorder


def _invoke(tmp_path: UPath, *extra: str):
    return runner.invoke(
        app,
        ["experiments", "randomization", "ElasticNet", "TOYv1", str(tmp_path / "randomized"), *extra],
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["experiments", "randomization"],
        ["experiments", "randomization", "ElasticNet"],
        ["experiments", "randomization", "ElasticNet", "TOYv1"],
    ],
    ids=["none", "model-only", "missing-output-dir"],
)
def test_missing_positional_arguments_are_usage_errors(worker: Recorder, argv: list[str]) -> None:
    assert runner.invoke(app, argv, env=HELP_ENV).exit_code == 2


def test_forwards_the_model_dataset_and_defaults(
    worker: Recorder, loader: Recorder, tmp_path: UPath, constructed: list[str]
) -> None:
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert constructed == ["ElasticNet"]
    assert loader.args == ("TOYv1",)
    model_class, dataset, modes = worker.args
    assert model_class.__name__ == "StubElasticNet"
    assert isinstance(dataset, FakeDataset)
    assert modes == ["SVRC"]
    assert worker.kwargs == {"randomization_type": "permutation", "random_state": 42}


def test_repeated_modes_are_forwarded_in_order_upper_cased(worker: Recorder, tmp_path: UPath) -> None:
    _invoke(tmp_path, "-m", "SVCC", "-m", "svrd")

    assert worker.args[2] == ["SVCC", "SVRD"]


@pytest.mark.parametrize("flag", ["--randomization-type", "-t"])
def test_randomization_type_and_seed_overrides(worker: Recorder, tmp_path: UPath, flag: str) -> None:
    _invoke(tmp_path, flag, "invariant", "--seed", "7")

    assert worker.kwargs == {"randomization_type": "invariant", "random_state": 7}


@pytest.mark.parametrize(
    "extra", [["-m", "SVXX"], ["--seed", "seed"], ["--random-state", "7"]], ids=["mode", "seed", "renamed-option"]
)
def test_invalid_options_are_usage_errors_before_the_worker(
    worker: Recorder, tmp_path: UPath, extra: list[str]
) -> None:
    result = _invoke(tmp_path, *extra)

    assert result.exit_code == 2
    assert worker.call_count == 0


def test_writes_one_file_per_randomized_dataset_and_echoes_the_count(worker: Recorder, tmp_path: UPath) -> None:
    out_dir = tmp_path / "nested" / "randomized"

    result = runner.invoke(app, ["experiments", "randomization", "ElasticNet", "TOYv1", str(out_dir)])

    written = sorted(p.name for p in out_dir.glob("*.h5mu"))
    assert written == sorted(f"{mode}__{view}.h5mu" for mode, view in RANDOMIZED)
    assert f"Wrote {len(RANDOMIZED)} randomized datasets to {out_dir}" in plain(result.output)


@pytest.mark.parametrize(
    ("datasets", "filename", "message"),
    [([FakeDataset(randomization=None)], "unknown__0.h5mu", "Wrote 1"), ([], None, "Wrote 0 randomized datasets")],
    ids=["untagged-falls-back-to-unknown", "empty-result"],
)
def test_untagged_and_empty_results(
    monkeypatch: pytest.MonkeyPatch,
    loader: Recorder,
    tmp_path: UPath,
    datasets: list[FakeDataset],
    filename: str | None,
    message: str,
) -> None:
    monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", lambda name: type("Stub", (), {}))
    patch_worker(monkeypatch, "drevalpy.experiment._randomization", "randomization", Recorder(return_value=datasets))

    result = _invoke(tmp_path)

    assert message in plain(result.output)
    if filename:
        assert (tmp_path / "randomized" / filename).exists()


def test_an_unknown_model_is_reported_and_creates_no_output_directory(
    monkeypatch: pytest.MonkeyPatch, loader: Recorder, tmp_path: UPath
) -> None:
    from drevalpy.models.spec import UnknownModelError

    def unknown(name: str) -> None:
        raise UnknownModelError(name, "no such model")

    monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", unknown)

    result = _invoke(tmp_path)

    assert result.exit_code == 1
    assert plain(result.output).startswith("Error: ")
    assert not (tmp_path / "randomized").exists()


class TestOnRealFiles:
    """The command on real ``.h5mu`` files; ``single`` records a file it wrote as a randomization run."""

    pytestmark = pytest.mark.slow

    @pytest.fixture()
    def written(self, tmp_path: UPath) -> tuple[UPath, UPath, UPath]:
        dataset, fold = write_synthetic_fold(runner, tmp_path)
        result = runner.invoke(
            app, ["experiments", "randomization", "ElasticNet", str(dataset), str(tmp_path / "rand")]
        )
        assert result.exit_code == 0, result.output
        return dataset, fold, tmp_path / "rand" / "SVRC__gene_expression.h5mu"

    def test_the_file_loads_with_its_identity_and_provenance(self, written: tuple[UPath, UPath, UPath]) -> None:
        from drevalpy.experiment._randomization import randomization
        from drevalpy.models import ModelSpec
        from drevalpy.types.data.dataset import Dataset

        dataset, _, path = written
        loaded = Dataset.load(path)
        in_process = randomization(ModelSpec.from_name("ElasticNet"), Dataset.load(dataset), ["SVRC"])[0]

        assert loaded.randomization == ("SVRC", "gene_expression")
        assert loaded.mdata.uns["randomization_provenance"] == in_process.mdata.uns["randomization_provenance"]

    def test_single_records_a_randomization_run_that_aggregate_keeps_apart(
        self, written: tuple[UPath, UPath, UPath], tmp_path: UPath
    ) -> None:
        from drevalpy.types.results import ExperimentResult, RunResult

        dataset, fold, path = written
        for source, name in ((dataset, "plain"), (path, "randomized")):
            result = runner.invoke(
                app, ["single", "ElasticNet", str(source), str(fold), str(tmp_path / f"{name}.npz"), "--no-hpo"]
            )
            assert result.exit_code == 0, result.output
        aggregated = runner.invoke(
            app,
            ["aggregate", str(tmp_path / "plain.npz"), str(tmp_path / "randomized.npz"), "-o", str(tmp_path / "agg")],
        )

        model = ExperimentResult.load(str(tmp_path / "agg")).models[0]
        assert aggregated.exit_code == 0, aggregated.output
        assert RunResult.load(str(tmp_path / "randomized.npz")).randomization == ("SVRC", "gene_expression")
        assert [run.randomization for run in model.select_runs(randomization="all")] == [
            None,
            ("SVRC", "gene_expression"),
        ]
        assert [run.randomization for run in model.select_runs()] == [None]
        assert model.n_folds == 1

    def test_a_cached_dataset_is_randomized_by_name(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        from drevalpy.data import synthetic

        monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path / "cache"))
        synthetic().save(tmp_path / "cache" / "CACHED.h5mu")

        result = runner.invoke(app, ["experiments", "randomization", "ElasticNet", "CACHED", str(tmp_path / "rand")])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "rand" / "SVRC__gene_expression.h5mu").is_file()

    def test_a_randomization_the_dataset_cannot_perform_writes_no_file(self, tmp_path: UPath) -> None:
        dataset, _ = write_synthetic_fold(runner, tmp_path)

        result = runner.invoke(
            app, ["experiments", "randomization", "ElasticNet", str(dataset), str(tmp_path / "rand"), "-m", "SVRD"]
        )

        assert result.exit_code == 0, result.output
        assert list((tmp_path / "rand").glob("*.h5mu")) == []
        assert "Wrote 0 randomized datasets" in plain(result.output)
