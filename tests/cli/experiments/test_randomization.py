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
    """Collect the model names handed to ``construct_model``."""
    return []


@pytest.fixture()
def loader(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Patch :func:`drevalpy.data.load`, which resolves a registered name or a path.

    Args:
        monkeypatch: Fixture used to replace the source-module worker.

    Returns:
        Recorder standing in for the dataset loader.
    """
    recorder = Recorder(return_value=FakeDataset())
    patch_worker(monkeypatch, "drevalpy.data.datasets._load", "load", recorder)
    patch_worker(monkeypatch, "drevalpy.data", "load", recorder)
    return recorder


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch, constructed: list[str], loader: Recorder) -> Recorder:
    """Patch every lazy import of ``randomization_cmd``.

    Args:
        monkeypatch: Fixture used to replace the source-module workers.
        constructed: List that receives every requested model name.
        loader: Dataset loader stub, so no file is read.

    Returns:
        Recorder standing in for
        :func:`drevalpy.experiment.randomization`, returning two
        datasets whose ``randomization`` tags drive the output filenames.
    """

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

    def test_resolves_the_dataset_argument_like_run_does(
        self, worker: Recorder, loader: Recorder, tmp_path: UPath
    ) -> None:
        """A registered name is as valid as a path, so the argument reaches ``load`` verbatim."""
        _invoke(tmp_path)

        assert loader.args == ("TOYv1",)

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
        _invoke(tmp_path, "-m", "SVCC", "-m", "svrd")

        assert worker.args[2] == ["SVCC", "SVRD"]

    def test_an_unknown_mode_is_a_usage_error(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "-m", "SVXX")

        assert result.exit_code == 2
        assert worker.call_count == 0

    def test_randomization_type_and_seed_defaults(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs == {"randomization_type": "permutation", "random_state": 42}

    @pytest.mark.parametrize("flag", ["--randomization-type", "-t"], ids=["long", "short"])
    def test_randomization_type_override(self, worker: Recorder, tmp_path: UPath, flag: str) -> None:
        _invoke(tmp_path, flag, "invariant")

        assert worker.kwargs["randomization_type"] == "invariant"

    def test_seed_override(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--seed", "7")

        assert worker.kwargs["random_state"] == 7

    def test_non_integer_seed_is_a_usage_error(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "--seed", "seed")

        assert result.exit_code == 2

    def test_the_renamed_random_state_option_is_rejected(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "--random-state", "7")

        assert result.exit_code == 2
        assert worker.call_count == 0


class TestOutput:
    """Filenames come from each dataset's ``randomization`` tag."""

    def test_creates_the_output_directory_including_parents(self, worker: Recorder, tmp_path: UPath) -> None:
        out_dir = tmp_path / "nested" / "randomized"
        runner.invoke(app, ["experiments", "randomization", "ElasticNet", "TOYv1", str(out_dir)])

        assert out_dir.is_dir()

    def test_writes_one_file_per_randomized_dataset(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        written = sorted(p.name for p in (tmp_path / "randomized").glob("*.h5mu"))
        assert written == sorted(f"{mode}__{view}.h5mu" for mode, view in RANDOMIZED)

    def test_untagged_dataset_falls_back_to_unknown(
        self, monkeypatch: pytest.MonkeyPatch, loader: Recorder, tmp_path: UPath
    ) -> None:
        monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", lambda name: type("Stub", (), {}))
        patch_worker(
            monkeypatch,
            "drevalpy.experiment._randomization",
            "randomization",
            Recorder(return_value=[FakeDataset(randomization=None)]),
        )

        _invoke(tmp_path)

        assert (tmp_path / "randomized" / "unknown__0.h5mu").exists()

    def test_echoes_the_dataset_count_and_destination(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        expected = f"Wrote {len(RANDOMIZED)} randomized datasets to {tmp_path / 'randomized'}"
        assert expected in plain(result.output)

    def test_empty_result_still_reports_zero(
        self, monkeypatch: pytest.MonkeyPatch, loader: Recorder, tmp_path: UPath
    ) -> None:
        monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", lambda name: type("Stub", (), {}))
        patch_worker(monkeypatch, "drevalpy.experiment._randomization", "randomization", Recorder(return_value=[]))

        result = _invoke(tmp_path)

        assert "Wrote 0 randomized datasets" in plain(result.output)


class TestUnknownModel:
    def test_is_reported_and_creates_no_output_directory(
        self, monkeypatch: pytest.MonkeyPatch, loader: Recorder, tmp_path: UPath
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
    """The command on real ``.h5mu`` files; a file it writes is a randomization run once ``single`` has trained on it.

    Each case reads and writes real synthetic data, which costs over 0.2s.
    """

    pytestmark = pytest.mark.slow

    @pytest.fixture()
    def written(self, tmp_path: UPath) -> tuple[UPath, UPath, UPath]:
        """The synthetic dataset, its first fold and the SVRC file the command wrote for ElasticNet."""
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
        """``data synthetic`` files hold SMILES, not the fingerprint ElasticNet derives from them."""
        dataset, _ = write_synthetic_fold(runner, tmp_path)

        result = runner.invoke(
            app, ["experiments", "randomization", "ElasticNet", str(dataset), str(tmp_path / "rand"), "-m", "SVRD"]
        )

        assert result.exit_code == 0, result.output
        assert list((tmp_path / "rand").glob("*.h5mu")) == []
        assert "Wrote 0 randomized datasets" in plain(result.output)
