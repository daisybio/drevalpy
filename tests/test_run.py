"""Tests for the top-level ``run`` orchestrator.

``single`` is replaced with a recorder, so these assert the fan-out arithmetic and the kwargs
forwarded to it without training a model. ``robustness`` is not stubbed.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from drevalpy import run
from drevalpy.errors import InvalidInputError
from drevalpy.models import ModelSpec
from drevalpy.models.response_transformation import GroupMeanCenterer, get_response_transformation
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.results import ExperimentResult
from drevalpy.types.results.failure import ExperimentFailedError
from tests.synthetic.results import DEFAULT_DATASET_NAME, make_run_result

# the orchestrator lives in the private module; ``drevalpy.run`` is the re-exported function
RUN_MODULE = importlib.import_module("drevalpy._run")


class _FakeDataset:
    def __init__(self, name: str = DEFAULT_DATASET_NAME, tag: str = "original") -> None:
        self.name = name
        self.tag = tag
        self.randomization: tuple[str, str] | None = None


def _stub_model(name: str) -> type:
    return ModelSpec(name, ModelSpec.from_name("NaivePredictor").config)


def _make_masks(*, fold_index: int, shape: tuple[int, int] = (4, 3)) -> SplitMasks:
    train = np.zeros(shape, dtype=bool)
    test = np.zeros(shape, dtype=bool)
    val = np.zeros(shape, dtype=bool)
    train[:2, :] = True
    test[2, :] = True
    val[3, :] = True
    return SplitMasks(
        train=SplitMask(train),
        test=SplitMask(test),
        val=SplitMask(val),
        metadata={"fold_index": fold_index, "split_mode": "LSO", "fold_id": f"fold_{fold_index}"},
    )


class _SingleRecorder:
    """Records every ``single`` call and returns a matching ``RunResult``."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.fail_when: Callable[[Any, Any, SplitMasks], bool] = lambda model, ds, masks: False
        self.error: BaseException = RuntimeError("boom")

    def __call__(self, model_class: type, dataset: Any, split_masks: SplitMasks, **kwargs: Any):
        self.calls.append(((model_class, dataset, split_masks), kwargs))
        if self.fail_when(model_class, dataset, split_masks):
            raise self.error
        return make_run_result(
            model_name=model_class.name,
            dataset_name=dataset.name,
            fold_index=split_masks.metadata["fold_index"],
            fold_metadata=dict(split_masks.metadata),
            randomization=dataset.randomization,
        )

    @property
    def datasets(self) -> list[Any]:
        return [args[1] for args, _ in self.calls]

    @property
    def masks(self) -> list[SplitMasks]:
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
    monkeypatch.setattr(RUN_MODULE, "single", stub)
    monkeypatch.setattr(RUN_MODULE, "load", lambda name: _FakeDataset(name=name))
    _set_folds(monkeypatch, 2)
    return stub


def _set_folds(monkeypatch: pytest.MonkeyPatch, n_folds: int, seen: list[Any] | None = None) -> None:
    def fake_split(ds: Any, mode: str, random_state: int = 42, quality: Any = None) -> list[SplitMasks]:
        if seen is not None:
            seen.append((ds, mode, random_state, quality))
        return [_make_masks(fold_index=i) for i in range(n_folds)]

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
        monkeypatch.setattr(RUN_MODULE, "load", lambda name: loaded.append(name) or sentinel)

        _run("A", "CTRPv1")
        assert loaded == ["CTRPv1"]
        assert recorder.datasets == [sentinel, sentinel]

        monkeypatch.setattr(RUN_MODULE, "load", lambda name: pytest.fail("load must not be called"))
        recorder.calls.clear()
        _run("A", dataset)
        assert recorder.datasets == [dataset, dataset]

    def test_several_datasets_are_loaded_and_concatenated_in_order(self, recorder, dataset, monkeypatch) -> None:
        joined = _FakeDataset(name="A+B", tag="joined")
        seen: list[list[Any]] = []
        monkeypatch.setattr(RUN_MODULE, "concat", lambda datasets: seen.append(list(datasets)) or joined)
        splits: list[Any] = []
        _set_folds(monkeypatch, 1, splits)

        _run("M", [dataset, "B"], mode="LStO")

        assert seen[0][0] is dataset
        assert seen[0][1].name == "B"
        assert recorder.datasets == [joined]
        assert [entry[0] for entry in splits] == [joined]

    def test_a_sequence_of_one_is_used_without_concatenation(self, recorder, dataset, monkeypatch) -> None:
        monkeypatch.setattr(RUN_MODULE, "concat", lambda datasets: pytest.fail("concat must not be called"))

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
        assert sum(m.n_runs for m in result.models) == expected
        assert sum(m.n_folds for m in result.models) == 2 * 3

    def test_the_original_dataset_runs_before_its_randomized_copies(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 1)
        _set_randomizations(monkeypatch, 2)

        _run("A", dataset, randomization_modes=["SVRC"])

        assert [ds.tag for ds in recorder.datasets] == ["original", "random_0", "random_1"]

    def test_an_empty_randomization_mode_list_adds_nothing(self, recorder, dataset, monkeypatch) -> None:
        monkeypatch.setattr(RUN_MODULE, "randomization", lambda *a, **k: pytest.fail("must not randomize"))

        _run("A", dataset, randomization_modes=[])

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
            assert all("robustness_repeat" not in masks.metadata for masks in recorder.masks)

    def test_trial_and_fold_indices_are_recorded(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 2)

        result = _run("A", dataset, robustness_repeats=2)

        assert [masks.metadata.get("robustness_repeat") for masks in recorder.masks] == [None, 0, 1, None, 0, 1]
        assert [masks.metadata["fold_index"] for masks in recorder.masks] == [0, 0, 0, 1, 1, 1]
        assert result.has_robustness
        assert [r.robustness_repeat for r in result.models[0].runs][:3] == [None, 0, 1]


class TestNormalizationReference:
    def test_the_reference_is_exempt_and_the_other_models_are_not(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 3)
        transformer = GroupMeanCenterer()

        _run([_stub_model("NaiveMeanEffectsPredictor"), _stub_model("A")], dataset, response_transformation=transformer)

        by_model = {name: [] for name in recorder.model_names}
        for args, kwargs in recorder.calls:
            by_model[args[0].name].append(kwargs["response_transformation"])
        assert by_model["NaiveMeanEffectsPredictor"] == [None] * 3
        assert all(seen is transformer for seen in by_model["A"])

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
        }

    def test_wandb_project_and_n_jobs_reach_every_single_call(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 3)

        _run("M", dataset, wandb_project="proj", n_jobs=4)

        assert [kwargs["wandb_project"] for _, kwargs in recorder.calls] == ["proj"] * 3
        assert [kwargs["n_jobs"] for _, kwargs in recorder.calls] == [4] * 3

    def test_the_split_mode_and_quality_rule_reach_split(self, recorder, dataset, monkeypatch) -> None:
        rule = object()
        splits: list[Any] = []
        _set_folds(monkeypatch, 1, splits)

        _run("M", dataset, mode="LDO", quality=rule)

        assert [(entry[1], entry[3]) for entry in splits] == [("LDO", rule)]
        assert inspect.signature(run).parameters["quality"].default is None

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
        splits: list[Any] = []
        _set_folds(monkeypatch, 1, splits)
        seen = _spy_randomization(monkeypatch)

        _run("A", dataset, randomization_modes=["SVRC"], seed=5)
        _run("A", dataset, randomization_modes=["SVRC"], seed=5, randomization_random_state=0)

        assert [entry[2] for entry in splits] == [5, 5]
        assert [entry["random_state"] for entry in seen] == [5, 0]
        _, kwargs = recorder.calls[0]
        assert kwargs["seed"] == 5
        assert kwargs["hpo_random_state"] is None

    @pytest.mark.parametrize("n_jobs", [0, -3])
    def test_an_unusable_n_jobs_is_rejected_before_anything_runs(self, recorder, dataset, n_jobs) -> None:
        with pytest.raises(InvalidInputError, match="n_jobs"):
            _run("A", dataset, n_jobs=n_jobs)

        assert recorder.calls == []


class TestFailures:
    def test_one_failing_fold_is_recorded_and_the_other_runs_survive(
        self, recorder, dataset, monkeypatch, caplog
    ) -> None:
        _set_folds(monkeypatch, 3)
        recorder.fail_when = lambda model, ds, masks: masks.metadata["fold_index"] == 1

        with caplog.at_level("WARNING"):
            result = _run("A", dataset)

        assert [r.fold_index for r in result.models[0].runs] == [0, 2]
        assert [(f.identity.model_name, f.identity.fold_index, f.error_type, f.message) for f in result.failures] == [
            ("A", 1, "RuntimeError", "boom")
        ]
        assert any("A fold 1" in message and "boom" in message for message in caplog.messages)

    def test_a_failing_model_does_not_stop_the_others(self, recorder, dataset) -> None:
        recorder.fail_when = lambda model, ds, masks: model.name == "Bad"

        result = _run([_stub_model("Bad"), _stub_model("Good")], dataset)

        assert result.model_names == ["Good"]
        assert [(f.identity.model_name, f.identity.fold_index) for f in result.failures] == [("Bad", 0), ("Bad", 1)]

    def test_failed_randomized_and_robustness_runs_record_their_identity(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 1)
        _set_randomizations(monkeypatch, 1)
        recorder.fail_when = lambda model, ds, masks: (
            ds.randomization is not None or masks.metadata.get("robustness_repeat") == 1
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
        recorder.fail_when = lambda model, ds, masks: True

        with pytest.raises(ExperimentFailedError) as caught:
            _run("A", dataset)

        assert [f.identity.fold_index for f in caught.value.failures] == [0, 1]

    def test_on_error_raise_reraises_the_first_error(self, recorder, dataset, monkeypatch) -> None:
        _set_folds(monkeypatch, 3)
        recorder.fail_when = lambda model, ds, masks: masks.metadata["fold_index"] == 1
        recorder.error = ValueError("first")

        with pytest.raises(ValueError, match="first"):
            _run("A", dataset, on_error="raise")

        assert len(recorder.calls) == 2

    def test_on_error_raise_also_reraises_a_randomization_error(self, recorder, dataset, monkeypatch) -> None:
        _broken_randomization(monkeypatch)

        with pytest.raises(KeyError):
            _run("A", dataset, randomization_modes=["SVRC"], on_error="raise")

    def test_a_keyboard_interrupt_propagates(self, recorder, dataset) -> None:
        recorder.fail_when = lambda model, ds, masks: True
        recorder.error = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            _run("A", dataset)

        assert len(recorder.calls) == 1

    def test_an_unknown_on_error_is_rejected(self, recorder, dataset) -> None:
        with pytest.raises(InvalidInputError, match="on_error"):
            _run("A", dataset, on_error="ignore")

    def test_failures_round_trip_through_save_and_load(self, recorder, dataset, tmp_path) -> None:
        assert _run("A", dataset).failures == ()
        recorder.fail_when = lambda model, ds, masks: masks.metadata["fold_index"] == 1
        result = _run("A", dataset)

        result.save(tmp_path)

        assert (tmp_path / "failures.json").exists()
        assert ExperimentResult.load(tmp_path).failures == result.failures
