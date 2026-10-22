"""Tests for :mod:`drevalpy.cli.run`, the ``drevalpy run`` command."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.data import CurveQuality
from drevalpy.models import Combinations, SkippedCombination
from drevalpy.types.results.failure import ExperimentFailedError, RunFailure
from tests.cli._helpers import HELP_ENV, Recorder, make_runner, patch_worker, plain

runner = make_runner()


class StubExperiment:
    """Stand-in for the ``ExperimentResult`` that ``run`` returns."""

    def __init__(self, failures: tuple[RunFailure, ...] = ()) -> None:
        """Start with an empty save log.

        Args:
            failures: Failures the stub reports, like `ExperimentResult.failures`.
        """
        self.saved: list[str] = []
        self.failures = failures

    def save(self, path: str) -> None:
        """Record the directory the command asked for.

        Args:
            path: Destination directory chosen by the command.
        """
        self.saved.append(path)

    def __repr__(self) -> str:
        """Sentinel text the command is expected to echo verbatim."""
        return "STUB-EXPERIMENT-REPR"


@pytest.fixture()
def constructed() -> list[str]:
    """Collect the model names handed to ``construct_model``."""
    return []


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch, constructed: list[str]) -> Recorder:
    """Patch both lazy imports of ``run_cmd`` and hand back the ``run`` recorder.

    Args:
        monkeypatch: Fixture used to replace the source-module workers.
        constructed: List that receives every requested model name.

    Returns:
        Recorder standing in for :func:`drevalpy.run`.
    """

    def fake_construct_model(name: str) -> SimpleNamespace:
        constructed.append(name)
        return SimpleNamespace(name=name)

    recorder = Recorder(return_value=StubExperiment())
    monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", fake_construct_model)
    patch_worker(monkeypatch, "drevalpy._run", "run", recorder)
    return recorder


def _invoke(tmp_path: UPath, *extra: str) -> Any:
    out_dir = tmp_path / "results"
    return runner.invoke(app, ["run", "ElasticNet", "--dataset", "TOYv1", "--output-dir", str(out_dir), *extra])


class TestDefaults:
    """A minimal invocation forwards the documented default hyperparameters."""

    def test_exits_cleanly(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert result.exit_code == 0, result.output

    def test_forwards_dataset_and_split_mode(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs["dataset"] == ["TOYv1"]
        assert worker.kwargs["split_mode"] == "LPO"

    def test_forwards_hpo_defaults(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs["hyperparameter_tuning"] is True
        assert worker.kwargs["hpo_metric"] == "RMSE"
        assert worker.kwargs["hpo_num_trials"] == 16
        assert worker.kwargs["hpo_random_state"] is None

    def test_forwards_experiment_defaults(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs["randomization_modes"] is None
        assert worker.kwargs["randomization_type"] == "permutation"
        assert worker.kwargs["randomization_random_state"] is None
        assert worker.kwargs["robustness_trials"] == 0
        assert worker.kwargs["seed"] == 42

    def test_forwards_no_response_transformation(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs["response_transformation"] is None

    def test_passes_the_constructed_specs_not_names(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert [spec.name for spec in worker.kwargs["models"]] == ["ElasticNet"]


class TestModelArgument:
    """``models`` is variadic and every name goes through ``construct_model``."""

    def test_each_name_is_constructed(self, worker: Recorder, tmp_path: UPath, constructed: list[str]) -> None:
        out_dir = tmp_path / "results"
        runner.invoke(
            app,
            ["run", "ElasticNet", "RandomForest", "--dataset", "TOYv1", "--output-dir", str(out_dir)],
        )

        assert constructed == ["ElasticNet", "RandomForest"]

    def test_missing_models_is_a_usage_error(self, worker: Recorder) -> None:
        result = runner.invoke(app, ["run", "--dataset", "TOYv1"], env=HELP_ENV)

        assert result.exit_code == 2

    def test_missing_dataset_is_a_usage_error(self, worker: Recorder) -> None:
        result = runner.invoke(app, ["run", "ElasticNet"], env=HELP_ENV)

        assert result.exit_code == 2

    def test_nothing_runs_on_a_usage_error(self, worker: Recorder) -> None:
        runner.invoke(app, ["run", "ElasticNet"], env=HELP_ENV)

        assert worker.call_count == 0


class TestOptionForwarding:
    """Every option maps onto a keyword of :func:`drevalpy.run`."""

    def test_no_hpo_disables_tuning(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--no-hpo")

        assert worker.kwargs["hyperparameter_tuning"] is False

    def test_split_mode_short_option(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "-s", "LSO")

        assert worker.kwargs["split_mode"] == "LSO"

    def test_repeated_randomization_modes_become_a_list(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "-r", "SVRC", "-r", "SVCD")

        assert worker.kwargs["randomization_modes"] == ["SVRC", "SVCD"]

    def test_randomization_type(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--randomization-type", "invariant")

        assert worker.kwargs["randomization_type"] == "invariant"

    def test_randomization_root_seed(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--randomization-random-state", "17")

        assert worker.kwargs["randomization_random_state"] == 17

    def test_hpo_tuning_options(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--hpo-metric", "Pearson", "--hpo-trials", "3", "--hpo-random-state", "7")

        assert worker.kwargs["hpo_metric"] == "Pearson"
        assert worker.kwargs["hpo_num_trials"] == 3
        assert worker.kwargs["hpo_random_state"] == 7

    def test_robustness_trials(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--robustness-trials", "4")

        assert worker.kwargs["robustness_trials"] == 4

    def test_seed(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--seed", "5")

        assert worker.kwargs["seed"] == 5
        assert worker.kwargs["hpo_random_state"] is None
        assert worker.kwargs["randomization_random_state"] is None

    def test_removed_precompute_flag_is_rejected(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "--precomputed-only")

        assert result.exit_code == 2

    def test_non_integer_hpo_trials_is_a_usage_error(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "--hpo-trials", "many")

        assert result.exit_code == 2


class TestResponseTransformation:
    """``--response-transformation`` resolves to the sklearn transformer prototype."""

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            pytest.param("standard", StandardScaler, id="standard"),
            pytest.param("minmax", MinMaxScaler, id="minmax"),
            pytest.param("robust", RobustScaler, id="robust"),
        ],
    )
    def test_each_option_selects_its_transformer(
        self, worker: Recorder, tmp_path: UPath, option: str, expected: type
    ) -> None:
        _invoke(tmp_path, "--response-transformation", option)

        assert isinstance(worker.kwargs["response_transformation"], expected)

    def test_the_prototype_is_handed_over_unfitted(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--response-transformation", "standard")

        assert not hasattr(worker.kwargs["response_transformation"], "mean_")

    @pytest.mark.parametrize("option", ["none", "None"])
    def test_none_disables_the_transformation(self, worker: Recorder, tmp_path: UPath, option: str) -> None:
        _invoke(tmp_path, "--response-transformation", option)

        assert worker.kwargs["response_transformation"] is None

    def test_an_unknown_option_is_rejected(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "--response-transformation", "logarithmic")

        assert result.exit_code == 2
        assert worker.call_count == 0


class TestGroupBy:
    """``--group-by`` prints the grouped summary after the repr; the run itself is unchanged."""

    def test_prints_the_grouped_summary(
        self, worker: Recorder, tmp_path: UPath, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []

        def fake_format(experiment: Any, group_by: Any) -> str:
            seen.append(group_by)
            return "GROUPED-SUMMARY"

        monkeypatch.setattr("drevalpy.cli.run.format_grouped_summary", fake_format)

        result = _invoke(tmp_path, "--group-by", "drug")

        assert "GROUPED-SUMMARY" in plain(result.output)
        assert [member.value for member in seen] == ["drug"]

    def test_is_off_by_default(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert plain(result.output).strip().endswith("STUB-EXPERIMENT-REPR")

    def test_an_unknown_grouping_is_rejected(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "--group-by", "tissue")

        assert result.exit_code == 2
        assert worker.call_count == 0


class TestOutput:
    """The command owns creating the output directory and reporting where it wrote."""

    def test_creates_the_output_directory_including_parents(self, worker: Recorder, tmp_path: UPath) -> None:
        out_dir = tmp_path / "nested" / "results"
        runner.invoke(app, ["run", "ElasticNet", "--dataset", "TOYv1", "--output-dir", str(out_dir)])

        assert out_dir.is_dir()

    def test_tolerates_a_pre_existing_output_directory(self, worker: Recorder, tmp_path: UPath) -> None:
        out_dir = tmp_path / "results"
        out_dir.mkdir()

        result = _invoke(tmp_path)

        assert result.exit_code == 0, result.output

    def test_saves_the_result_into_the_output_directory(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        experiment = worker.return_value
        assert experiment.saved == [str(tmp_path / "results")]

    def test_echoes_the_output_directory(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert f"Wrote experiment results to {tmp_path / 'results'}" in plain(result.output)

    def test_echoes_the_result_repr(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert "STUB-EXPERIMENT-REPR" in plain(result.output)


class TestRepeatableDataset:
    """``--dataset`` is repeatable, and every value reaches ``run`` in order."""

    def test_several_datasets_become_a_list(self, worker: Recorder, tmp_path: UPath) -> None:
        out_dir = tmp_path / "results"
        runner.invoke(
            app, ["run", "ElasticNet", "-d", "GDSC2", "-d", "CTRPv2", "-s", "LStO", "--output-dir", str(out_dir)]
        )

        assert worker.kwargs["dataset"] == ["GDSC2", "CTRPv2"]
        assert worker.kwargs["split_mode"] == "LStO"


class TestRecipeModels:
    """MODEL arguments are resolved by ``ModelSpec.from_name``, so recipes pass through."""

    def test_a_recipe_string_is_handed_to_from_name(
        self, worker: Recorder, tmp_path: UPath, constructed: list[str]
    ) -> None:
        recipe = "pca[gene_expression]:fingerprints:elasticNet"
        out_dir = tmp_path / "results"
        runner.invoke(app, ["run", recipe, "-d", "TOYv1", "--output-dir", str(out_dir)])

        assert constructed == [recipe]
        assert [spec.name for spec in worker.kwargs["models"]] == [recipe]


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

    def test_the_flags_reach_combinations(self, worker: Recorder, grid: Recorder, tmp_path: UPath) -> None:
        result = self._invoke_grid(
            tmp_path, "-S", "raw[gene_expression]", "-S", "pca[gene_expression]", "-D", "fingerprints", "-P", "ridge"
        )

        assert result.exit_code == 0, result.output
        assert grid.kwargs == {
            "sample": ["raw[gene_expression]", "pca[gene_expression]"],
            "drug": ["fingerprints"],
            "predictors": ["ridge"],
        }

    def test_the_long_option_names_work(self, worker: Recorder, grid: Recorder, tmp_path: UPath) -> None:
        self._invoke_grid(
            tmp_path, "--sample-featurizer", "raw[gene_expression]", "--drug-featurizer", "fingerprints",
            "--predictor", "ridge",
        )  # fmt: skip

        assert grid.kwargs["predictors"] == ["ridge"]

    def test_skipped_combinations_are_printed_with_their_reason(
        self, worker: Recorder, grid: Recorder, tmp_path: UPath
    ) -> None:
        result = self._invoke_grid(tmp_path, "-P", "ridge")

        assert "raw[gene_expression]:fingerprints:naive: naive takes no features" in plain(result.output)

    def test_grid_specs_are_run_without_any_model_argument(
        self, worker: Recorder, grid: Recorder, tmp_path: UPath
    ) -> None:
        self._invoke_grid(tmp_path, "-P", "ridge")

        assert [spec.name for spec in worker.kwargs["models"]] == [
            "raw[gene_expression]:fingerprints:ridge",
            "ElasticNet",
        ]

    def test_grid_specs_are_added_to_the_model_arguments_and_deduplicated_by_name(
        self, worker: Recorder, grid: Recorder, tmp_path: UPath
    ) -> None:
        out_dir = tmp_path / "results"
        runner.invoke(app, ["run", "ElasticNet", "RandomForest", "-d", "TOYv1", "-P", "ridge", "-o", str(out_dir)])

        assert [spec.name for spec in worker.kwargs["models"]] == [
            "ElasticNet",
            "RandomForest",
            "raw[gene_expression]:fingerprints:ridge",
        ]

    def test_an_invalid_grid_is_a_usage_error(
        self, worker: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        def nothing_valid(**kwargs: Any) -> Combinations:
            raise ValueError("no valid combination")

        monkeypatch.setattr("drevalpy.models.combinations", nothing_valid)

        result = self._invoke_grid(tmp_path, "-P", "ridge")

        assert result.exit_code == 2
        assert worker.call_count == 0

    def test_featurizer_flags_without_a_predictor_are_a_usage_error(self, worker: Recorder, tmp_path: UPath) -> None:
        result = self._invoke_grid(tmp_path, "-S", "pca[gene_expression]")

        assert result.exit_code == 2
        assert worker.call_count == 0

    def test_no_grid_flags_do_not_call_combinations(
        self, worker: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        monkeypatch.setattr("drevalpy.models.combinations", lambda **kwargs: pytest.fail("no grid was asked for"))

        assert _invoke(tmp_path).exit_code == 0


class TestQualityOptions:
    """The curve-quality options become one ``CurveQuality`` passed to ``run``."""

    def test_the_default_rule_is_the_default_quality(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs["quality"] == CurveQuality.default()

    def test_no_quality_filter_turns_the_rule_off(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--no-quality-filter")

        assert worker.kwargs["quality"] == CurveQuality.off()

    def test_thresholds_replace_the_defaults(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--min-relevance-score", "2.5", "--min-abs-fold-change", "0.7")

        quality = worker.kwargs["quality"]
        assert (quality.min_relevance_score, quality.min_abs_fold_change) == (2.5, 0.7)

    def test_a_threshold_with_no_quality_filter_is_a_usage_error(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "--no-quality-filter", "--min-abs-fold-change", "0.7")

        assert result.exit_code == 2
        assert worker.call_count == 0


class TestWandbProject:
    def test_defaults_to_no_tracking(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs["wandb_project"] is None

    def test_the_project_is_forwarded(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--wandb-project", "drevalpy")

        assert worker.kwargs["wandb_project"] == "drevalpy"


class TestFailures:
    """Recorded failures are printed after the results are saved, and set exit code 1."""

    FAILURE = RunFailure("ElasticNet", 1, None, None, "RuntimeError", "boom")

    def test_failures_set_exit_code_1_and_are_listed(self, worker: Recorder, tmp_path: UPath) -> None:
        worker.return_value = StubExperiment(failures=(self.FAILURE,))

        result = _invoke(tmp_path)

        assert result.exit_code == 1
        assert "1 run(s) failed" in plain(result.output)
        assert "ElasticNet fold 1: RuntimeError: boom" in plain(result.output)

    def test_the_results_are_saved_before_exiting(self, worker: Recorder, tmp_path: UPath) -> None:
        experiment = worker.return_value = StubExperiment(failures=(self.FAILURE,))

        _invoke(tmp_path)

        assert experiment.saved == [str(tmp_path / "results")]

    def test_no_failures_exit_cleanly(self, worker: Recorder, tmp_path: UPath) -> None:
        assert _invoke(tmp_path).exit_code == 0

    def test_every_run_failing_lists_the_failures_and_exits_1(
        self, worker: Recorder, tmp_path: UPath, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def all_failed(**kwargs: Any) -> None:
            raise ExperimentFailedError([self.FAILURE])

        patch_worker(monkeypatch, "drevalpy._run", "run", all_failed)

        result = _invoke(tmp_path)

        assert result.exit_code == 1
        assert "ElasticNet fold 1: RuntimeError: boom" in plain(result.output)
