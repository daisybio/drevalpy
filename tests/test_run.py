"""Tests for the top-level ``run`` orchestrator.

``_fit_fold`` is replaced with a recorder, so these assert the fan-out arithmetic and the kwargs
forwarded to it without training a model. ``robustness`` is not stubbed. The recorder's run on an
original fold reports ``{"alpha": <fold index>}`` as its best hyperparameters, so a test can tell
which run a stress-test run took its values from.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable
from typing import Any

import anndata as ad
import mudata as md
import numpy as np
import pytest

from drevalpy import run, single, split, synthetic
from drevalpy.errors import InvalidInputError, OutputDirectoryConflictError
from drevalpy.models import ModelSpec
from drevalpy.models.response_transformation import get_response_transformation
from drevalpy.types import Fold, PairMask
from drevalpy.types.results import ExperimentResult
from drevalpy.types.results.failure import ExperimentFailedError
from tests.synthetic.results import DEFAULT_DATASET_NAME, make_run_result

# the orchestrator lives in the private module; ``drevalpy.run`` is the re-exported function
RUN_MODULE = importlib.import_module("drevalpy._run")
RESOLVE_MODULE = importlib.import_module("drevalpy.data._resolve")


class _FakeDataset:
    def __init__(self, name: str = DEFAULT_DATASET_NAME, tag: str = "original") -> None:
        self.name = name
        self.tag = tag
        self.randomization: tuple[str, str] | None = None
        self.measure = "pEC50"
        with md.set_options(pull_on_update=False):
            self.mdata = md.MuData({"response": ad.AnnData(np.arange(12, dtype=np.float32).reshape(4, 3))})


def _stub_model(name: str) -> type:
    return ModelSpec(name, ModelSpec.from_name("NaivePredictor").config)


def _make_fold(*, fold_index: int, shape: tuple[int, int] = (4, 3)) -> Fold:
    train = np.zeros(shape, dtype=bool)
    test = np.zeros(shape, dtype=bool)
    val = np.zeros(shape, dtype=bool)
    train[:2, :] = True
    test[2, :] = True
    val[3, :] = True
    return Fold(
        train=PairMask(train),
        test=PairMask(test),
        val=PairMask(val),
        metadata={"fold_index": fold_index, "split_mode": "LSO", "fold_id": f"fold_{fold_index}"},
    )


class _SingleRecorder:
    """Records every ``single`` call and returns a matching ``RunResult``."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.fail_when: Callable[[Any, Any, Fold], bool] = lambda model, ds, fold: False
        self.error: BaseException = RuntimeError("boom")

    def __call__(self, model_class: type, dataset: Any, fold: Fold, **kwargs: Any):
        self.calls.append(((model_class, dataset, fold), kwargs))
        if self.fail_when(model_class, dataset, fold):
            raise self.error
        hyperparameters = kwargs.get("hyperparameters")
        return make_run_result(
            model_name=model_class.name,
            dataset_name=dataset.name,
            fold_index=fold.metadata["fold_index"],
            split_mode=fold.metadata["split_mode"],
            fold_metadata=dict(fold.metadata),
            randomization=dataset.randomization,
            best_hyperparameters=(
                {"alpha": fold.metadata["fold_index"]} if hyperparameters is None else dict(hyperparameters)
            ),
        )

    @property
    def datasets(self) -> list[Any]:
        return [args[1] for args, _ in self.calls]

    @property
    def folds(self) -> list[Fold]:
        return [args[2] for args, _ in self.calls]

    @property
    def model_names(self) -> list[str]:
        return [args[0].name for args, _ in self.calls]


@pytest.fixture
def dataset() -> _FakeDataset:
    return _FakeDataset()


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _SingleRecorder:
    stub = _SingleRecorder()
    monkeypatch.setattr(RUN_MODULE, "_fit_fold", stub)
    monkeypatch.setattr(RESOLVE_MODULE, "load", lambda name: _FakeDataset(name=name))
    _set_folds(monkeypatch, 2)
    return stub


def _set_folds(monkeypatch: pytest.MonkeyPatch, n_folds: int, seen: list[Any] | None = None) -> None:
    def fake_split(
        ds: Any,
        mode: str,
        n_splits: int = 5,
        validation_ratio: float = 0.1,
        random_state: int = 42,
        quality: Any = None,
    ) -> list[Fold]:
        if seen is not None:
            seen.append((ds, mode, random_state, quality, n_splits, validation_ratio))
        return [_make_fold(fold_index=i) for i in range(n_folds)]

    monkeypatch.setattr(RUN_MODULE, "split", fake_split)


def _set_randomizations(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    def fake_randomization(model_class, ds, modes, randomization_type="permutation", random_state=42):
        copies = []
        for index in range(n):
            copy = _FakeDataset(name=ds.name, tag=f"random_{index}")
            copy.randomization = (modes[0], f"view_{index}")
            copies.append(copy)
        return copies

    monkeypatch.setattr(RUN_MODULE, "randomization", fake_randomization)


def _spy_randomization(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def spy(model, ds, modes, **kwargs):
        seen.append({"model": model.name, "modes": modes, **kwargs})
        return []

    monkeypatch.setattr(RUN_MODULE, "randomization", spy)
    return seen


def _broken_randomization(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args, **kwargs):
        raise KeyError("no such view")

    monkeypatch.setattr(RUN_MODULE, "randomization", broken)


def _run(models: Any, data: Any, **kwargs: Any) -> ExperimentResult:
    kwargs.setdefault("hyperparameter_tuning", False)
    return run(models if isinstance(models, list) else [_stub_model(models)], data, kwargs.pop("mode", "LSO"), **kwargs)


class TestDatasetResolution:
    def test_a_dataset_name_is_loaded_and_an_object_is_used_as_is(self, recorder, dataset, monkeypatch) -> None:
        sentinel = _FakeDataset(name="CTRPv1", tag="loaded")
        loaded: list[str] = []
        monkeypatch.setattr(RESOLVE_MODULE, "load", lambda name: loaded.append(name) or sentinel)

        _run("A", "CTRPv1")
        assert loaded == ["CTRPv1"]
        assert recorder.datasets == [sentinel, sentinel]

        monkeypatch.setattr(RESOLVE_MODULE, "load", lambda name: pytest.fail("load must not be called"))
        recorder.calls.clear()
        _run("A", dataset)
        assert recorder.datasets == [dataset, dataset]

    def test_several_datasets_are_loaded_and_concatenated_in_order(self, recorder, dataset, monkeypatch) -> None:
        joined = _FakeDataset(name="A+B", tag="joined")
        seen: list[list[Any]] = []
        monkeypatch.setattr(RESOLVE_MODULE, "concat", lambda datasets: seen.append(list(datasets)) or joined)
        split_calls: list[Any] = []
        _set_folds(monkeypatch, 1, split_calls)

        _run("M", [dataset, "B"], mode="LStO")

        assert seen[0][0] is dataset
        assert seen[0][1].name == "B"
        assert recorder.datasets == [joined]
        assert [entry[0] for entry in split_calls] == [joined]

    def test_a_sequence_of_one_is_used_without_concatenation(self, recorder, dataset, monkeypatch) -> None:
        monkeypatch.setattr(RESOLVE_MODULE, "concat", lambda datasets: pytest.fail("concat must not be called"))

        _run("M", [dataset])

        assert recorder.datasets == [dataset, dataset]

    def test_an_empty_sequence_is_rejected(self, recorder) -> None:
        with pytest.raises(InvalidInputError, match="empty sequence"):
            _run("M", [])


class TestFanOut:
    @pytest.mark.parametrize(("n_models", "n_folds"), [(1, 1), (2, 3), (3, 2)])
    def test_one_run_per_model_and_fold(self, recorder, dataset, monkeypatch, n_models, n_folds) -> None:
        _set_folds(monkeypatch, n_folds)

        result = _run([_stub_model(f"Model{i}") for i in range(n_models)], dataset)

        assert len(recorder.calls) == n_models * n_folds
        assert sum(m.n_folds for m in result.models) == n_models * n_folds

    @pytest.mark.parametrize("n_randomizations", [1, 3])
    def test_randomization_adds_one_run_per_randomized_dataset(
        self, recorder, dataset, monkeypatch, n_randomizations
    ) -> None:
        _set_folds(monkeypatch, 3)
        _set_randomizations(monkeypatch, n_randomizations)

        result = _run([_stub_model("A"), _stub_model("B")], dataset, randomization_modes=["SVRC"])

        expected = 2 * 3 * (1 + n_randomizations)
        assert len(recorder.calls) == expected
        assert sum(len(m.runs) for m in result.models) == expected
        assert sum(m.n_folds for m in result.models) == 2 * 3

    def test_the_original_dataset_runs_before_its_randomized_copies(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 1)
        _set_randomizations(monkeypatch, 2)

        _run("A", dataset, randomization_modes=["SVRC"])

        assert [ds.tag for ds in recorder.datasets] == ["original", "random_0", "random_1"]

    @pytest.mark.parametrize("modes", [[], None])
    def test_no_randomization_mode_adds_nothing(self, recorder, dataset, modes) -> None:
        _run("A", dataset, randomization_modes=modes)

        assert len(recorder.calls) == 2

    def test_no_models_is_rejected(self, recorder, dataset) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _run([], dataset)

    def test_results_are_grouped_by_model_name_and_models_iterate_before_folds(
        self, recorder, dataset, monkeypatch
    ) -> None:
        _set_folds(monkeypatch, 2)
        _set_randomizations(monkeypatch, 1)

        result = _run([_stub_model("Alpha"), _stub_model("Beta")], dataset, randomization_modes=["SVRC"])

        assert sorted(result.model_names) == ["Alpha", "Beta"]
        assert result.has_randomization
        assert recorder.model_names == ["Alpha"] * 4 + ["Beta"] * 4

    def test_randomization_runs_once_per_model_and_every_fold_reuses_its_datasets(
        self, recorder, dataset, monkeypatch
    ) -> None:
        _set_folds(monkeypatch, 3)
        _set_randomizations(monkeypatch, 1)
        calls: list[str] = []
        inner = RUN_MODULE.randomization
        monkeypatch.setattr(
            RUN_MODULE, "randomization", lambda model, *a, **k: calls.append(model.name) or inner(model, *a, **k)
        )

        _run([_stub_model("A"), _stub_model("B")], dataset, randomization_modes=["SVRC"])

        assert calls == ["A", "B"]
        first_model = [ds for ds in recorder.datasets[:6] if ds.randomization is not None]
        assert len(first_model) == 3
        assert first_model[0] is first_model[1] is first_model[2]


class TestRobustness:
    @pytest.mark.parametrize("trials", [0, 2, 4])
    def test_trials_multiply_the_fold_count(self, recorder, dataset, monkeypatch, trials) -> None:
        _set_folds(monkeypatch, 3)

        _run("A", dataset, robustness_repeats=trials)

        assert len(recorder.calls) == 3 * (1 + trials)
        if trials == 0:
            assert all("robustness_repeat" not in fold.metadata for fold in recorder.folds)

    def test_trial_and_fold_indices_are_recorded(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 2)

        result = _run("A", dataset, robustness_repeats=2)

        assert [fold.metadata.get("robustness_repeat") for fold in recorder.folds] == [None, 0, 1, None, 0, 1]
        assert [fold.metadata["fold_index"] for fold in recorder.folds] == [0, 0, 0, 1, 1, 1]
        assert result.has_robustness
        assert [r.robustness_repeat for r in result.models[0].runs][:3] == [None, 0, 1]


class TestNormalizationReference:
    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["drug_mean", "drug_tissue_mean", "standard"])
    def test_end_to_end_the_reference_ignores_the_transformation_while_others_move(
        self, synthetic_dataset, name
    ) -> None:
        def predictions(transformation: Any) -> dict[str, np.ndarray]:
            result = run(
                [ModelSpec.from_name("NaiveMeanEffectsPredictor"), ModelSpec.from_name("NaivePredictor")],
                synthetic_dataset,
                "LSO",
                hyperparameter_tuning=False,
                response_transformation=transformation,
            )
            return {m.model_name: np.concatenate([r.predictions for r in m.runs]) for m in result.models}

        reference = predictions(None)
        transformed = predictions(get_response_transformation(name))

        np.testing.assert_array_equal(transformed["NaiveMeanEffectsPredictor"], reference["NaiveMeanEffectsPredictor"])
        if name == "drug_mean":
            assert not np.allclose(transformed["NaivePredictor"], reference["NaivePredictor"])

    @pytest.mark.slow
    def test_run_and_split_single_aggregate_give_the_same_normalized_metrics(self, synthetic_dataset) -> None:
        models = [ModelSpec.from_name("NaiveMeanEffectsPredictor"), ModelSpec.from_name("NaivePredictor")]
        settings = {"hyperparameter_tuning": False, "response_transformation": get_response_transformation("drug_mean")}

        whole = run(models, synthetic_dataset, "LSO", n_splits=3, **settings)
        folds = split(synthetic_dataset, "LSO", n_splits=3, random_state=42)
        parts = ExperimentResult(
            [single(model, synthetic_dataset, fold, **settings) for model in models for fold in folds]
        )

        def normalized(result: ExperimentResult) -> list[dict[str, float]]:
            return [run.metrics for model in result.normalize().models for run in model.runs]

        assert normalized(parts) == normalized(whole)


class TestForwardedArguments:
    def test_hpo_settings_reach_single(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 1)

        _run("A", dataset, hyperparameter_tuning=True, hpo_metric="MSE", hpo_num_trials=4, hpo_random_state=7)

        _, kwargs = recorder.calls[0]
        assert kwargs == {
            "hyperparameter_tuning": True,
            "hpo_metric": "MSE",
            "hpo_num_trials": 4,
            "hpo_random_state": 7,
            "seed": 42,
            "response_transformation": None,
            "wandb_project": None,
            "n_jobs": 1,
            "hyperparameters": None,
        }

    def test_wandb_project_and_n_jobs_reach_every_single_call(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 3)

        _run("M", dataset, wandb_project="proj", n_jobs=4)

        assert [kwargs["wandb_project"] for _, kwargs in recorder.calls] == ["proj"] * 3
        assert [kwargs["n_jobs"] for _, kwargs in recorder.calls] == [4] * 3

    def test_the_split_mode_and_quality_rule_reach_split(self, recorder, dataset, monkeypatch) -> None:
        rule = object()
        split_calls: list[Any] = []
        _set_folds(monkeypatch, 1, split_calls)

        _run("M", dataset, mode="LDO", quality=rule)

        assert [(entry[1], entry[3]) for entry in split_calls] == [("LDO", rule)]
        assert inspect.signature(run).parameters["quality"].default is None

    def test_n_splits_reaches_split_and_defaults_to_five(self, recorder, dataset, monkeypatch) -> None:
        split_calls: list[Any] = []
        _set_folds(monkeypatch, 1, split_calls)

        _run("M", dataset)
        _run("M", dataset, n_splits=3)

        assert [entry[4] for entry in split_calls] == [5, 3]

    def test_the_validation_ratio_reaches_split_and_keeps_its_default(self, recorder, dataset, monkeypatch) -> None:
        split_calls: list[Any] = []
        _set_folds(monkeypatch, 1, split_calls)

        _run("M", dataset)
        _run("M", dataset, validation_ratio=0.3)

        assert [entry[5] for entry in split_calls] == [0.1, 0.3]
        assert inspect.signature(split).parameters["validation_ratio"].default == 0.1

    @pytest.mark.slow
    def test_end_to_end_n_splits_sets_the_runs_per_model(self) -> None:
        result = run(
            [ModelSpec.from_name("NaiveMeanEffectsPredictor")],
            synthetic(),
            "LPO",
            n_splits=2,
            hyperparameter_tuning=False,
        )

        assert [m.n_folds for m in result.models] == [2]

    @pytest.mark.slow
    def test_fewer_than_two_splits_fail_in_the_splitter(self) -> None:
        with pytest.raises(ValueError, match="n_splits=2 or more"):
            run([ModelSpec.from_name("NaiveMeanEffectsPredictor")], synthetic(), "LPO", n_splits=1)

    def test_the_randomization_type_and_seed_reach_randomization(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 1)
        seen = _spy_randomization(monkeypatch)

        _run("A", dataset, randomization_modes=["SVRC", "SVRD"], randomization_type="invariant")
        _run("A", dataset, randomization_modes=["SVRC"], randomization_random_state=17)

        assert seen[0] == {
            "model": "A",
            "modes": ["SVRC", "SVRD"],
            "randomization_type": "invariant",
            "random_state": 42,
        }
        assert seen[1]["random_state"] == 17

    def test_the_seed_drives_split_and_defaults_randomization(self, recorder, dataset, monkeypatch) -> None:
        split_calls: list[Any] = []
        _set_folds(monkeypatch, 1, split_calls)
        seen = _spy_randomization(monkeypatch)

        _run("A", dataset, randomization_modes=["SVRC"], seed=5)
        _run("A", dataset, randomization_modes=["SVRC"], seed=5, randomization_random_state=0)

        assert [entry[2] for entry in split_calls] == [5, 5]
        assert [entry["random_state"] for entry in seen] == [5, 0]
        _, kwargs = recorder.calls[0]
        assert kwargs["seed"] == 5
        assert kwargs["hpo_random_state"] == 5

    @pytest.mark.parametrize("n_jobs", [0, -3])
    def test_an_unusable_n_jobs_is_rejected_before_anything_runs(self, recorder, dataset, n_jobs) -> None:
        with pytest.raises(InvalidInputError, match="n_jobs"):
            _run("A", dataset, n_jobs=n_jobs)

        assert recorder.calls == []


class TestModeSupport:
    def test_a_single_drug_model_under_ldo_is_rejected_before_the_dataset_is_split(
        self, recorder, dataset, monkeypatch
    ) -> None:
        monkeypatch.setattr(RUN_MODULE, "split", lambda *args, **kwargs: pytest.fail("split must not be called"))
        models = [_stub_model("A"), ModelSpec.from_name("SingleDrugElasticNet")]

        with pytest.raises(InvalidInputError, match="single-drug model"):
            _run(models, dataset, mode="LDO", on_error="record")

        assert recorder.calls == []


class TestStressTestHyperparameters:
    def test_original_runs_tune_and_every_stress_test_run_reuses_its_folds_values(
        self, recorder, dataset, monkeypatch
    ) -> None:
        _set_folds(monkeypatch, 2)
        _set_randomizations(monkeypatch, 1)

        result = _run("A", dataset, hyperparameter_tuning=True, randomization_modes=["SVRC"], robustness_repeats=1)

        seen = [
            (fold.metadata["fold_index"], fold.metadata.get("robustness_repeat"), ds.tag, kwargs["hyperparameters"])
            for (_, ds, fold), kwargs in recorder.calls
        ]
        assert seen == [
            (0, None, "original", None),
            (0, None, "random_0", {"alpha": 0}),
            (0, 0, "original", {"alpha": 0}),
            (0, 0, "random_0", {"alpha": 0}),
            (1, None, "original", None),
            (1, None, "random_0", {"alpha": 1}),
            (1, 0, "original", {"alpha": 1}),
            (1, 0, "random_0", {"alpha": 1}),
        ]
        assert [run.best_hyperparameters for run in result.models[0].runs] == [
            {"alpha": index} for index in [0] * 4 + [1] * 4
        ]

    def test_the_stress_test_runs_of_a_failed_original_run_are_failures(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 2)
        _set_randomizations(monkeypatch, 1)
        recorder.fail_when = lambda model, ds, fold: fold.metadata["fold_index"] == 0

        result = _run("A", dataset, randomization_modes=["SVRC"], robustness_repeats=1)

        assert [fold.metadata["fold_index"] for fold in recorder.folds] == [0, 1, 1, 1, 1]
        failures = [
            (f.identity.fold_index, f.identity.randomization, f.identity.robustness_repeat, f.error_type)
            for f in result.failures
        ]
        view = ("SVRC", "view_0")
        assert failures == [
            (0, None, None, "RuntimeError"),
            (0, view, None, "OriginalRunFailedError"),
            (0, None, 0, "OriginalRunFailedError"),
            (0, view, 0, "OriginalRunFailedError"),
        ]
        assert "original run of fold 0 failed" in result.failures[1].message
        assert len(result.models[0].runs) == 4

    @pytest.mark.slow
    def test_end_to_end_only_the_original_runs_tune(self, synthetic_dataset, monkeypatch) -> None:
        fit_module = importlib.import_module("drevalpy.experiment._fit")
        real_tune = fit_module.hpam_tune
        tuned: list[tuple[int, Any, Any]] = []

        def counting_tune(execution, dataset, **kwargs):
            if execution.hyperparameter_tuning:
                tuned.append(
                    (
                        execution.identity.fold_index,
                        execution.identity.randomization,
                        execution.identity.robustness_repeat,
                    )
                )
            return real_tune(execution, dataset, **kwargs)

        monkeypatch.setattr(fit_module, "hpam_tune", counting_tune)

        result = run(
            [ModelSpec.from_name("Ridge")],
            synthetic_dataset,
            "LSO",
            n_splits=2,
            hpo_num_trials=2,
            randomization_modes=["SVRC"],
            robustness_repeats=1,
        )

        assert tuned == [(0, None, None), (1, None, None)]
        runs = result.models[0].runs
        assert len(runs) == 8
        for run_result in runs:
            original = next(
                r
                for r in runs
                if r.fold_index == run_result.fold_index and r.randomization is None and r.robustness_repeat is None
            )
            assert run_result.best_hyperparameters == original.best_hyperparameters
            assert (run_result.trials is None) == (run_result is not original)


class TestFailures:
    def test_one_failing_fold_is_recorded_and_the_other_runs_survive(
        self, recorder, dataset, monkeypatch, caplog
    ) -> None:
        _set_folds(monkeypatch, 3)
        recorder.fail_when = lambda model, ds, fold: fold.metadata["fold_index"] == 1

        with caplog.at_level("WARNING"):
            result = _run("A", dataset)

        assert [r.fold_index for r in result.models[0].runs] == [0, 2]
        assert [(f.identity.model_name, f.identity.fold_index, f.error_type, f.message) for f in result.failures] == [
            ("A", 1, "RuntimeError", "boom")
        ]
        assert any("A fold 1" in message and "boom" in message for message in caplog.messages)

    def test_a_failing_model_does_not_stop_the_others(self, recorder, dataset) -> None:
        recorder.fail_when = lambda model, ds, fold: model.name == "Bad"

        result = _run([_stub_model("Bad"), _stub_model("Good")], dataset)

        assert result.model_names == ["Good"]
        assert [(f.identity.model_name, f.identity.fold_index) for f in result.failures] == [("Bad", 0), ("Bad", 1)]

    def test_failed_randomized_and_robustness_runs_record_their_identity(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 1)
        _set_randomizations(monkeypatch, 1)
        recorder.fail_when = lambda model, ds, fold: (
            ds.randomization is not None or fold.metadata.get("robustness_repeat") == 1
        )

        result = _run("A", dataset, randomization_modes=["SVRC"], robustness_repeats=2)

        view = ("SVRC", "view_0")
        assert {(f.identity.randomization, f.identity.robustness_repeat) for f in result.failures} == {
            (view, None),
            (view, 0),
            (view, 1),
            (None, 1),
        }

    def test_a_randomization_that_fails_is_recorded_per_fold_and_the_original_runs_stay(
        self, recorder, dataset, monkeypatch
    ) -> None:
        _broken_randomization(monkeypatch)

        result = _run("A", dataset, randomization_modes=["SVRC"])

        assert [(f.identity.fold_index, f.error_type) for f in result.failures] == [(0, "KeyError"), (1, "KeyError")]
        assert len(result.models[0].runs) == 2

    def test_every_run_failing_raises_with_the_failures(self, recorder, dataset) -> None:
        recorder.fail_when = lambda model, ds, fold: True

        with pytest.raises(ExperimentFailedError) as caught:
            _run("A", dataset)

        assert [f.identity.fold_index for f in caught.value.failures] == [0, 1]

    def test_on_error_raise_reraises_the_first_error(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 3)
        recorder.fail_when = lambda model, ds, fold: fold.metadata["fold_index"] == 1
        recorder.error = ValueError("first")

        with pytest.raises(ValueError, match="first"):
            _run("A", dataset, on_error="raise")

        assert len(recorder.calls) == 2

    def test_on_error_raise_also_reraises_a_randomization_error(self, recorder, dataset, monkeypatch) -> None:
        _broken_randomization(monkeypatch)

        with pytest.raises(KeyError):
            _run("A", dataset, randomization_modes=["SVRC"], on_error="raise")

    def test_a_keyboard_interrupt_propagates(self, recorder, dataset) -> None:
        recorder.fail_when = lambda model, ds, fold: True
        recorder.error = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            _run("A", dataset)

        assert len(recorder.calls) == 1

    def test_an_unknown_on_error_is_rejected(self, recorder, dataset) -> None:
        with pytest.raises(InvalidInputError, match="on_error"):
            _run("A", dataset, on_error="ignore")

    def test_failures_round_trip_through_save_and_load(self, recorder, dataset, tmp_path) -> None:
        assert _run("A", dataset).failures == ()
        recorder.fail_when = lambda model, ds, fold: fold.metadata["fold_index"] == 1
        result = _run("A", dataset)

        result.save(tmp_path)

        assert (tmp_path / "failures.json").exists()
        assert ExperimentResult.load(tmp_path).failures == result.failures


class TestResume:
    """With ``output_dir`` every run is saved when it finishes, and a rerun loads it instead of fitting it."""

    @staticmethod
    def _stress_run(data: Any, directory: Any, **kwargs: Any) -> ExperimentResult:
        return _run(
            "A",
            data,
            hyperparameter_tuning=True,
            randomization_modes=["SVRC"],
            robustness_repeats=1,
            output_dir=str(directory),
            **kwargs,
        )

    def test_a_second_run_with_the_same_settings_fits_nothing(self, recorder, dataset, monkeypatch, tmp_path) -> None:
        _set_randomizations(monkeypatch, 1)
        first = self._stress_run(dataset, tmp_path)
        recorder.calls.clear()

        second = self._stress_run(dataset, tmp_path, n_jobs=4, wandb_project="other", on_error="raise")

        assert recorder.calls == []
        assert [r.run_key for r in second.models[0].runs] == [r.run_key for r in first.models[0].runs]
        assert [r.best_hyperparameters for r in second.models[0].runs] == [
            r.best_hyperparameters for r in first.models[0].runs
        ]
        assert ExperimentResult.load(tmp_path).models[0].n_folds == 2

    @pytest.mark.parametrize("error", [KeyboardInterrupt(), ValueError("killed")], ids=["interrupt", "raise"])
    def test_after_a_kill_a_rerun_fits_only_the_missing_runs(
        self, recorder, dataset, monkeypatch, tmp_path, error: BaseException
    ) -> None:
        _set_randomizations(monkeypatch, 1)
        recorder.fail_when = lambda model, ds, fold: len(recorder.calls) == 3
        recorder.error = error
        with pytest.raises(type(error)):
            self._stress_run(dataset, tmp_path, on_error="raise")
        recorder.calls.clear()
        recorder.fail_when = lambda model, ds, fold: False

        result = self._stress_run(dataset, tmp_path)

        seen = [
            (fold.metadata["fold_index"], fold.metadata.get("robustness_repeat"), ds.tag, kwargs["hyperparameters"])
            for (_, ds, fold), kwargs in recorder.calls
        ]
        assert seen == [
            (0, 0, "original", {"alpha": 0}),
            (0, 0, "random_0", {"alpha": 0}),
            (1, None, "original", None),
            (1, None, "random_0", {"alpha": 1}),
            (1, 0, "original", {"alpha": 1}),
            (1, 0, "random_0", {"alpha": 1}),
        ]
        assert len(result.models[0].runs) == 8

    def test_failures_are_not_saved_so_a_rerun_retries_them(self, recorder, dataset, tmp_path) -> None:
        recorder.fail_when = lambda model, ds, fold: fold.metadata["fold_index"] == 1
        _run("A", dataset, output_dir=str(tmp_path))
        assert (tmp_path / "failures.json").exists()
        recorder.calls.clear()
        recorder.fail_when = lambda model, ds, fold: False

        result = _run("A", dataset, output_dir=str(tmp_path))

        assert [fold.metadata["fold_index"] for fold in recorder.folds] == [1]
        assert result.failures == ()
        assert not (tmp_path / "failures.json").exists()

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"seed": 7}, id="seed"),
            pytest.param({"mode": "LPO"}, id="split-mode"),
            pytest.param({"hyperparameter_tuning": True}, id="tuning"),
            pytest.param({"robustness_repeats": 1}, id="robustness"),
            pytest.param({"validation_ratio": 0.3}, id="validation-ratio"),
            pytest.param({"response_transformation": get_response_transformation("standard")}, id="transformation"),
        ],
    )
    def test_other_settings_raise_and_overwrite_deletes_and_reruns(
        self, recorder, dataset, tmp_path, change: dict[str, Any]
    ) -> None:
        _run("A", dataset, output_dir=str(tmp_path))
        recorder.calls.clear()

        with pytest.raises(OutputDirectoryConflictError, match="--overwrite"):
            _run("A", dataset, output_dir=str(tmp_path), **change)
        assert recorder.calls == []

        result = _run("A", dataset, output_dir=str(tmp_path), overwrite=True, **change)

        assert len(recorder.calls) == len(result.models[0].runs)
        assert len(list(tmp_path.rglob("*.npz"))) == len(result.models[0].runs)

    @pytest.mark.parametrize("change", ["responses", "features"])
    def test_other_data_under_the_same_name_or_another_model_raise(
        self, recorder, dataset, tmp_path, change: str
    ) -> None:
        _run("A", dataset, output_dir=str(tmp_path))
        other = _FakeDataset()
        if change == "responses":
            other.mdata.mod["response"].X = np.ones((4, 3), dtype=np.float32)
        else:
            other.mdata.mod["response"].obsm["pathways"] = np.ones((4, 2), dtype=np.float32)

        with pytest.raises(OutputDirectoryConflictError):
            _run("A", other, output_dir=str(tmp_path))
        with pytest.raises(OutputDirectoryConflictError):
            _run("B", dataset, output_dir=str(tmp_path))

    def test_a_run_that_crashed_before_any_run_finished_is_started_again(
        self, recorder, dataset, monkeypatch, tmp_path
    ) -> None:
        def broken_split(*args: Any, **kwargs: Any) -> list[Fold]:
            raise ValueError("n_splits=2 or more")

        real_split = RUN_MODULE.split
        monkeypatch.setattr(RUN_MODULE, "split", broken_split)
        with pytest.raises(ValueError, match="n_splits"):
            _run("A", dataset, n_splits=1, output_dir=str(tmp_path))
        monkeypatch.setattr(RUN_MODULE, "split", real_split)

        result = _run("A", dataset, n_splits=2, output_dir=str(tmp_path))

        assert len(result.models[0].runs) == 2
        assert result.settings is not None
        assert result.settings["n_splits"] == 2

    def test_a_run_whose_every_run_failed_saves_the_failures_and_is_started_again(
        self, recorder, dataset, tmp_path
    ) -> None:
        recorder.fail_when = lambda model, ds, fold: True
        with pytest.raises(ExperimentFailedError):
            _run("A", dataset, output_dir=str(tmp_path))
        assert len(json.loads((tmp_path / "failures.json").read_text(encoding="utf-8"))) == 2
        recorder.fail_when = lambda model, ds, fold: False

        result = _run("A", dataset, seed=7, output_dir=str(tmp_path))

        assert len(result.models[0].runs) == 2
        assert not (tmp_path / "failures.json").exists()

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_a_directory_of_other_files_is_refused_and_kept(self, recorder, dataset, tmp_path, overwrite) -> None:
        (tmp_path / "notes.txt").write_text("mine", encoding="utf-8")

        with pytest.raises(OutputDirectoryConflictError, match="empty or new directory"):
            _run("A", dataset, output_dir=str(tmp_path), overwrite=overwrite)

        assert recorder.calls == []
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["notes.txt"]

    def test_saving_the_result_into_its_directory_keeps_it_resumable(self, recorder, dataset, tmp_path) -> None:
        result = _run("A", dataset, output_dir=str(tmp_path / "run"))
        result.save(tmp_path / "run")
        result.save(tmp_path / "copy")
        recorder.calls.clear()

        again = _run("A", dataset, output_dir=str(tmp_path / "run"))

        assert recorder.calls == []
        assert ExperimentResult.load(tmp_path / "copy").settings == result.settings == again.settings

    def test_settings_that_cannot_be_recorded_are_rejected(self, recorder, dataset, tmp_path) -> None:
        class Opaque:
            def get_params(self, deep: bool = False) -> dict[str, Any]:
                return {"state": object()}

        with pytest.raises(InvalidInputError, match="cannot be recorded"):
            _run("A", dataset, response_transformation=Opaque(), output_dir=str(tmp_path))
        assert recorder.calls == []

    @pytest.mark.slow
    def test_end_to_end_a_rerun_loads_every_run(self, synthetic_dataset, monkeypatch, tmp_path) -> None:
        fits: list[Any] = []
        real_fit = RUN_MODULE._fit_fold
        monkeypatch.setattr(
            RUN_MODULE, "_fit_fold", lambda *args, **kwargs: fits.append(args) or real_fit(*args, **kwargs)
        )

        def experiment() -> ExperimentResult:
            return run(
                [ModelSpec.from_name("Ridge")],
                synthetic_dataset,
                "LSO",
                n_splits=2,
                hpo_num_trials=2,
                randomization_modes=["SVRC"],
                robustness_repeats=1,
                output_dir=str(tmp_path),
            )

        first = experiment()
        n_fits = len(fits)
        second = experiment()

        assert n_fits == len(first.models[0].runs) == 8
        assert len(fits) == n_fits
        assert [r.metrics for r in second.models[0].runs] == [r.metrics for r in first.models[0].runs]
