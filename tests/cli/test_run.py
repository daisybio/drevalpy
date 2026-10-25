"""Tests for :mod:`drevalpy.cli.run`, the ``drevalpy run`` command."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sklearn.preprocessing import StandardScaler
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.data import CurveQuality
from drevalpy.errors import InvalidInputError
from drevalpy.models import Combinations, SkippedCombination
from drevalpy.types.results.failure import ExperimentFailedError, RunFailure
from tests.cli._helpers import HELP_ENV, Recorder, make_runner, patch_worker, plain
from tests.synthetic import make_run_failure

runner = make_runner()


class StubExperiment:
    def __init__(self, failures: tuple[RunFailure, ...] = ()) -> None:
        """Record saved paths."""
        self.saved: list[str] = []
        self.failures = failures

    def save(self, path: str) -> None:
        self.saved.append(path)

    def __repr__(self) -> str:
        """Return a marker the output test can look for."""
        return "STUB-EXPERIMENT-REPR"


@pytest.fixture()
def constructed() -> list[str]:
    return []


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch, constructed: list[str]) -> Recorder:
    def fake_construct_model(name: str) -> SimpleNamespace:
        constructed.append(name)
        return SimpleNamespace(name=name)

    recorder = Recorder(return_value=StubExperiment())
    monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", fake_construct_model)
    patch_worker(monkeypatch, "drevalpy._run", "run", recorder)
    return recorder


def _invoke(tmp_path: UPath, *extra: str) -> Any:
    return runner.invoke(
        app, ["run", "ElasticNet", "--dataset", "TOYv1", "--output-dir", str(tmp_path / "results"), *extra]
    )


def test_a_minimal_invocation_forwards_the_documented_defaults(worker: Recorder, tmp_path: UPath) -> None:
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert [spec.name for spec in worker.kwargs["models"]] == ["ElasticNet"]
    assert {key: value for key, value in worker.kwargs.items() if key != "models"} == {
        "dataset": ["TOYv1"],
        "split_mode": "LPO",
        "hyperparameter_tuning": True,
        "hpo_metric": "RMSE",
        "hpo_num_trials": 16,
        "hpo_random_state": None,
        "randomization_modes": None,
        "randomization_type": "permutation",
        "randomization_random_state": None,
        "robustness_repeats": 0,
        "seed": 42,
        "n_jobs": 1,
        "response_transformation": None,
        "quality": CurveQuality.default(),
        "wandb_project": None,
    }


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        pytest.param(["--no-hpo"], {"hyperparameter_tuning": False}, id="no-hpo"),
        pytest.param(["-s", "LSO"], {"split_mode": "LSO"}, id="split-mode"),
        pytest.param(
            ["--randomization-mode", "SVRC", "--randomization-mode", "svcd"],
            {"randomization_modes": ["SVRC", "SVCD"]},
            id="repeated-randomization-modes",
        ),
        pytest.param(["--randomization-type", "invariant"], {"randomization_type": "invariant"}, id="rand-type"),
        pytest.param(["--randomization-random-state", "17"], {"randomization_random_state": 17}, id="rand-state"),
        pytest.param(
            ["--hpo-metric", "Pearson", "--hpo-trials", "3", "--hpo-random-state", "7"],
            {"hpo_metric": "Pearson", "hpo_num_trials": 3, "hpo_random_state": 7},
            id="hpo",
        ),
        pytest.param(["--n-jobs", "3"], {"n_jobs": 3}, id="n-jobs"),
        pytest.param(["--robustness-repeats", "4"], {"robustness_repeats": 4}, id="robustness"),
        pytest.param(
            ["--seed", "5"],
            {"seed": 5, "hpo_random_state": None, "randomization_random_state": None},
            id="seed",
        ),
        pytest.param(["--wandb-project", "drevalpy"], {"wandb_project": "drevalpy"}, id="wandb"),
        pytest.param(["--no-quality-filter"], {"quality": CurveQuality.off()}, id="no-quality-filter"),
    ],
)
def test_option_forwarding(worker: Recorder, tmp_path: UPath, args: list[str], expected: dict[str, Any]) -> None:
    _invoke(tmp_path, *args)

    assert {key: worker.kwargs[key] for key in expected} == expected


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--randomization-mode", "SVXX"], id="unknown-randomization-mode"),
        pytest.param(["-r", "SVRC"], id="removed-short-randomization-flag"),
        pytest.param(["--precomputed-only"], id="removed-precompute-flag"),
        pytest.param(["--hpo-trials", "many"], id="non-integer-trials"),
        pytest.param(["--response-transformation", "logarithmic"], id="unknown-transformation"),
        pytest.param(["--no-quality-filter", "--min-abs-fold-change", "0.7"], id="threshold-without-filter"),
    ],
)
def test_bad_options_are_usage_errors_and_nothing_runs(worker: Recorder, tmp_path: UPath, args: list[str]) -> None:
    result = _invoke(tmp_path, *args)

    assert result.exit_code == 2
    assert worker.call_count == 0


def test_models_are_variadic_and_the_dataset_is_required(worker: Recorder, tmp_path: UPath, constructed) -> None:
    runner.invoke(
        app,
        ["run", "ElasticNet", "RandomForest", "-d", "GDSC2", "-d", "CTRPv2", "-s", "LStO", "-o", str(tmp_path / "r")],
    )

    assert constructed == ["ElasticNet", "RandomForest"]
    assert worker.kwargs["dataset"] == ["GDSC2", "CTRPv2"]
    assert worker.kwargs["split_mode"] == "LStO"
    worker.calls.clear()
    assert runner.invoke(app, ["run", "--dataset", "TOYv1"], env=HELP_ENV).exit_code == 2
    assert runner.invoke(app, ["run", "ElasticNet"], env=HELP_ENV).exit_code == 2
    assert worker.call_count == 0


def test_a_recipe_string_is_handed_to_from_name(worker: Recorder, tmp_path: UPath, constructed) -> None:
    recipe = "pca[gene_expression]:fingerprints:elasticNet"

    runner.invoke(app, ["run", recipe, "-d", "TOYv1", "--output-dir", str(tmp_path / "results")])

    assert constructed == [recipe]
    assert [spec.name for spec in worker.kwargs["models"]] == [recipe]


class TestResponseTransformation:
    def test_a_choice_is_forwarded_as_its_unfitted_transformer(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--response-transformation", "standard")

        transformer = worker.kwargs["response_transformation"]
        assert isinstance(transformer, StandardScaler)
        assert not hasattr(transformer, "mean_")


class TestQualityOptions:
    def test_thresholds_replace_the_defaults(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--min-relevance-score", "2.5", "--min-abs-fold-change", "0.7")

        quality = worker.kwargs["quality"]
        assert (quality.min_relevance_score, quality.min_abs_fold_change) == (2.5, 0.7)


class TestGroupBy:
    def test_prints_the_grouped_summary_only_when_asked(
        self, worker: Recorder, tmp_path: UPath, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []

        def fake_format(experiment: Any, group_by: Any, dataset: Any = None) -> str:
            seen.append(group_by)
            return "GROUPED-SUMMARY"

        monkeypatch.setattr("drevalpy.cli.run.format_grouped_summary", fake_format)

        plain_result = _invoke(tmp_path)
        grouped = _invoke(tmp_path, "--group-by", "drug")

        assert plain(plain_result.output).strip().endswith("STUB-EXPERIMENT-REPR")
        assert "GROUPED-SUMMARY" in plain(grouped.output)
        assert seen == ["drug"]


class TestOutput:
    def test_creates_parents_saves_and_reports(self, worker: Recorder, tmp_path: UPath) -> None:
        out_dir = tmp_path / "nested" / "results"

        result = runner.invoke(app, ["run", "ElasticNet", "--dataset", "TOYv1", "--output-dir", str(out_dir)])

        assert out_dir.is_dir()
        assert worker.return_value.saved == [str(out_dir)]
        assert f"Wrote experiment results to {out_dir}" in plain(result.output)
        assert "STUB-EXPERIMENT-REPR" in plain(result.output)

    def test_tolerates_a_pre_existing_output_directory(self, worker: Recorder, tmp_path: UPath) -> None:
        (tmp_path / "results").mkdir()

        assert _invoke(tmp_path).exit_code == 0

    def test_an_unknown_model_is_reported_without_a_traceback_and_nothing_runs(
        self, worker: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        from drevalpy.models.spec import UnknownModelError

        def unknown(name: str) -> None:
            raise UnknownModelError(name, "no such model")

        monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", unknown)

        result = _invoke(tmp_path)

        assert result.exit_code == 1
        assert plain(result.output).startswith("Error: ")
        assert "Traceback" not in result.output
        assert worker.call_count == 0
        assert not (tmp_path / "results").exists()


class TestGrid:
    """``-S/-D/-P`` expand through ``drevalpy.models.combinations``."""

    @pytest.fixture()
    def grid(self, monkeypatch: pytest.MonkeyPatch) -> Recorder:
        specs = [SimpleNamespace(name="raw[gene_expression]:fingerprints:ridge"), SimpleNamespace(name="ElasticNet")]
        skipped = [SkippedCombination("raw[gene_expression]:fingerprints:naive", "naive takes no features")]
        recorder = Recorder(return_value=Combinations(specs=specs, skipped=skipped))
        monkeypatch.setattr("drevalpy.models.combinations", recorder)
        return recorder

    def _invoke_grid(self, tmp_path: UPath, *args: str) -> Any:
        return runner.invoke(app, ["run", "-d", "TOYv1", "--output-dir", str(tmp_path / "results"), *args])

    def test_the_flags_reach_combinations_and_skipped_ones_are_printed(
        self, worker: Recorder, grid: Recorder, tmp_path: UPath
    ) -> None:
        result = self._invoke_grid(
            tmp_path, "-S", "raw[gene_expression]", "-S", "pca[gene_expression]", "-D", "fingerprints", "-P", "ridge"
        )

        assert result.exit_code == 0, result.output
        assert grid.kwargs == {
            "sample": ["raw[gene_expression]", "pca[gene_expression]"],
            "drug": ["fingerprints"],
            "predictors": ["ridge"],
        }
        assert "raw[gene_expression]:fingerprints:naive: naive takes no features" in plain(result.output)
        assert [spec.name for spec in worker.kwargs["models"]] == [
            "raw[gene_expression]:fingerprints:ridge",
            "ElasticNet",
        ]

    def test_the_long_option_names_work(self, worker: Recorder, grid: Recorder, tmp_path: UPath) -> None:
        self._invoke_grid(
            tmp_path, "--sample-featurizer", "raw[gene_expression]", "--drug-featurizer", "fingerprints",
            "--predictor", "ridge",
        )  # fmt: skip

        assert grid.kwargs["predictors"] == ["ridge"]

    def test_grid_specs_are_added_to_the_model_arguments(
        self, worker: Recorder, grid: Recorder, tmp_path: UPath
    ) -> None:
        runner.invoke(
            app, ["run", "ElasticNet", "RandomForest", "-d", "TOYv1", "-P", "ridge", "-o", str(tmp_path / "results")]
        )

        assert [spec.name for spec in worker.kwargs["models"]] == [
            "ElasticNet",
            "RandomForest",
            "raw[gene_expression]:fingerprints:ridge",
        ]

    def test_an_invalid_grid_is_reported_as_an_error(
        self, worker: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        def nothing_valid(**kwargs: Any) -> Combinations:
            raise InvalidInputError("no valid combination")

        monkeypatch.setattr("drevalpy.models.combinations", nothing_valid)

        result = self._invoke_grid(tmp_path, "-P", "ridge")

        assert result.exit_code == 1
        assert "Error: no valid combination" in plain(result.output)
        assert worker.call_count == 0

    def test_featurizer_flags_without_a_predictor_are_a_usage_error(self, worker: Recorder, tmp_path: UPath) -> None:
        result = self._invoke_grid(tmp_path, "-S", "pca[gene_expression]")

        assert result.exit_code == 2
        assert worker.call_count == 0


class TestFailures:
    """Recorded failures are printed after the results are saved, and set exit code 1."""

    FAILURE = make_run_failure("ElasticNet", 1)

    def test_failures_set_exit_code_1_are_listed_and_the_results_are_saved(
        self, worker: Recorder, tmp_path: UPath
    ) -> None:
        experiment = worker.return_value = StubExperiment(failures=(self.FAILURE,))

        result = _invoke(tmp_path)

        assert result.exit_code == 1
        assert "1 run(s) failed" in plain(result.output)
        assert "ElasticNet fold 1: RuntimeError: boom" in plain(result.output)
        assert experiment.saved == [str(tmp_path / "results")]

    def test_every_run_failing_lists_the_failures_and_exits_1(
        self, worker: Recorder, tmp_path: UPath, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def all_failed(**kwargs: Any) -> None:
            raise ExperimentFailedError([self.FAILURE])

        patch_worker(monkeypatch, "drevalpy._run", "run", all_failed)

        result = _invoke(tmp_path)

        assert result.exit_code == 1
        assert "Error: 1 run(s) failed:\n  ElasticNet fold 1: RuntimeError: boom" in plain(result.output)
