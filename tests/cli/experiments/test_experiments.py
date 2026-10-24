"""Tests for the ``drevalpy experiments`` command group: ``robustness`` and ``randomization``."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli.experiments import experiments_app
from drevalpy.cli.experiments.randomization import randomization_cmd
from drevalpy.cli.experiments.robustness import robustness_cmd
from drevalpy.cli.main import app
from drevalpy.types import SplitMask, SplitMasks
from tests.cli._helpers import (
    FakeDataset,
    GroupSurface,
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


def _invoke_randomization(tmp_path: UPath, *extra: str):
    return runner.invoke(
        app,
        ["experiments", "randomization", "ElasticNet", "TOYv1", str(tmp_path / "randomized"), *extra],
    )


def test_forwards_the_model_dataset_and_defaults(
    worker: Recorder, loader: Recorder, tmp_path: UPath, constructed: list[str]
) -> None:
    result = _invoke_randomization(tmp_path)

    assert result.exit_code == 0, result.output
    assert constructed == ["ElasticNet"]
    assert loader.args == ("TOYv1",)
    model_class, dataset, modes = worker.args
    assert model_class.__name__ == "StubElasticNet"
    assert isinstance(dataset, FakeDataset)
    assert modes == ["SVRC"]
    assert worker.kwargs == {"randomization_type": "permutation", "random_state": 42}


def test_repeated_modes_are_forwarded_in_order_upper_cased(worker: Recorder, tmp_path: UPath) -> None:
    _invoke_randomization(tmp_path, "-m", "SVCC", "-m", "svrd")

    assert worker.args[2] == ["SVCC", "SVRD"]


@pytest.mark.parametrize("flag", ["--randomization-type", "-t"])
def test_randomization_type_and_seed_overrides(worker: Recorder, tmp_path: UPath, flag: str) -> None:
    _invoke_randomization(tmp_path, flag, "invariant", "--seed", "7")

    assert worker.kwargs == {"randomization_type": "invariant", "random_state": 7}


@pytest.mark.parametrize(
    "extra", [["-m", "SVXX"], ["--seed", "seed"], ["--random-state", "7"]], ids=["mode", "seed", "renamed-option"]
)
def test_invalid_options_are_usage_errors_before_the_worker(
    worker: Recorder, tmp_path: UPath, extra: list[str]
) -> None:
    result = _invoke_randomization(tmp_path, *extra)

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

    result = _invoke_randomization(tmp_path)

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

    result = _invoke_randomization(tmp_path)

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


N_PERMUTATIONS_DEFAULT = 5


def _write_fold(path: UPath, fold_index: int) -> None:
    rng = np.random.default_rng(fold_index)
    assignment = rng.integers(0, 3, size=(3, 3))
    SplitMasks(
        train=SplitMask(assignment == 0),
        test=SplitMask(assignment == 1),
        val=SplitMask(assignment == 2),
        metadata={"fold_index": fold_index},
    ).save(path)


@pytest.fixture()
def splits_dir(tmp_path: UPath) -> UPath:
    path = tmp_path / "splits"
    path.mkdir()
    for fold_index in range(2):
        _write_fold(path / f"fold_{fold_index}.npz", fold_index)
    return path


def _invoke_robustness(splits_dir: UPath, out_dir: UPath, *extra: str):
    return runner.invoke(app, ["experiments", "robustness", str(splits_dir), str(out_dir), *extra])


@pytest.mark.parametrize("kind", ["empty", "no-npz", "absent"])
def test_a_directory_without_folds_is_an_exit_one_error_and_creates_no_output(tmp_path: UPath, kind: str) -> None:
    splits = tmp_path / "splits"
    if kind != "absent":
        splits.mkdir()
    if kind == "no-npz":
        (splits / "notes.txt").write_text("not a fold")

    result = _invoke_robustness(splits, tmp_path / "out")

    assert result.exit_code == 1
    assert f"Error: No .npz files found in {splits}" in plain(result.output)
    assert "Traceback" not in result.output
    assert not (tmp_path / "out").exists()


def test_default_generates_one_variant_per_fold_and_trial(splits_dir: UPath, tmp_path: UPath) -> None:
    out_dir = tmp_path / "nested" / "out"

    result = _invoke_robustness(splits_dir, out_dir)

    assert result.exit_code == 0, result.output
    assert len(list(out_dir.glob("*.npz"))) == 2 * N_PERMUTATIONS_DEFAULT


@pytest.mark.parametrize("flag", ["--n-permutations", "-n"])
def test_permutation_count_names_files_after_fold_and_trial_and_echoes_the_count(
    splits_dir: UPath, tmp_path: UPath, flag: str
) -> None:
    out_dir = tmp_path / "out"

    result = _invoke_robustness(splits_dir, out_dir, flag, "2")

    assert sorted(p.name for p in out_dir.glob("*.npz")) == [
        "fold_0_trial_0.npz",
        "fold_0_trial_1.npz",
        "fold_1_trial_0.npz",
        "fold_1_trial_1.npz",
    ]
    assert f"Wrote 4 robustness splits to {out_dir}" in plain(result.output)


def test_variants_record_their_trial_and_preserve_metadata_and_masks(splits_dir: UPath, tmp_path: UPath) -> None:
    out_dir = tmp_path / "out"
    _invoke_robustness(splits_dir, out_dir, "-n", "2")

    variant = SplitMasks.load(out_dir / "fold_1_trial_1.npz")
    original = SplitMasks.load(splits_dir / "fold_1.npz")
    assert variant.metadata["robustness_trial"] == 1
    assert variant.metadata["fold_index"] == 1
    np.testing.assert_array_equal(variant.train.mask, original.train.mask)


def test_zero_permutations_writes_nothing_and_a_non_integer_count_is_a_usage_error(
    splits_dir: UPath, tmp_path: UPath
) -> None:
    _invoke_robustness(splits_dir, tmp_path / "out", "-n", "0")

    assert list((tmp_path / "out").glob("*.npz")) == []
    assert _invoke_robustness(splits_dir, tmp_path / "out2", "-n", "several").exit_code == 2


class TestRobustnessOnRealFiles:
    """A shuffled split trains through ``single`` and is recorded under its robustness trial."""

    pytestmark = pytest.mark.slow

    def test_single_records_the_robustness_trial(self, tmp_path: UPath) -> None:
        from drevalpy.types.results import RunResult

        dataset, fold = write_synthetic_fold(runner, tmp_path)
        shuffled = runner.invoke(app, ["experiments", "robustness", str(fold.parent), str(tmp_path / "rob"), "-n", "2"])
        result = runner.invoke(
            app,
            [
                "single",
                "ElasticNet",
                str(dataset),
                str(tmp_path / "rob" / "fold_0_trial_1.npz"),
                str(tmp_path / "r.npz"),
                "--no-hpo",
            ],
        )

        assert shuffled.exit_code == 0, shuffled.output
        assert result.exit_code == 0, result.output
        run = RunResult.load(str(tmp_path / "r.npz"))
        assert run.robustness_trial == 1
        assert run.randomization is None


class TestGroup(GroupSurface):
    group = experiments_app
    path = "experiments"
    callbacks = {"robustness": robustness_cmd, "randomization": randomization_cmd}
