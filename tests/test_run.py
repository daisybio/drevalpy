"""Tests for the top-level ``run`` orchestrator.

``drevalpy/run.py`` contains no numerical logic of its own: it loads or accepts a
dataset, expands it into folds, optionally multiplies those folds by robustness
trials, optionally adds randomized copies of the dataset, and hands every
combination to :func:`drevalpy.single`. These tests therefore replace
``single`` with a recorder and assert the fan-out arithmetic and the kwargs it
forwards, so nothing here trains a model.

``robustness`` is *not* stubbed: it is four statements over real ``SplitMask``
objects, so exercising it for real is cheaper than faking it and lets the
``fold_metadata["robustness_trial"]`` assertion cover the whole path.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from drevalpy import run
from drevalpy.errors import InvalidInputError
from drevalpy.models import ModelSpec
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.results import ExperimentResult
from drevalpy.types.results.failure import ExperimentFailedError
from tests.synthetic.results import DEFAULT_DATASET_NAME, make_run_result

#: The orchestrator lives in the private ``drevalpy._run`` module so that the
#: public ``drevalpy.run`` name can be the re-exported function. Patching the
#: private module is what ``run``'s own top-level imports actually read.
RUN_MODULE = importlib.import_module("drevalpy._run")


class _FakeDataset:
    """Stand-in for ``Dataset`` exposing only what ``run`` and the stubs read."""

    def __init__(self, name: str = DEFAULT_DATASET_NAME, tag: str = "original") -> None:
        self.name = name
        self.tag = tag
        self.randomization: tuple[str, str] | None = None


def _stub_model(name: str) -> type:
    """Build a minimal ``DRPModel`` stand-in identified by ``name``."""
    return ModelSpec(name, ModelSpec.from_name("NaivePredictor").config)


def _make_masks(*, fold_index: int, shape: tuple[int, int] = (4, 3)) -> SplitMasks:
    """Build one fold of real masks with a disjoint train/test/val partition."""
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
        #: Raise `error` for every call this predicate accepts (model, dataset, masks).
        self.fail_when: Callable[[Any, Any, SplitMasks], bool] = lambda model, ds, masks: False
        self.error: BaseException = RuntimeError("boom")

    def __call__(self, model_class: type, mudataset: Any, split_masks: SplitMasks, **kwargs: Any):
        self.calls.append(((model_class, mudataset, split_masks), kwargs))
        if self.fail_when(model_class, mudataset, split_masks):
            raise self.error
        return make_run_result(
            model_name=model_class.name,
            dataset_name=mudataset.name,
            fold_index=split_masks.metadata["fold_index"],
            fold_metadata=dict(split_masks.metadata),
            randomization=mudataset.randomization,
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
    """Replace ``single`` with a recorder and neutralise ``split``/``load``.

    ``split`` returns two folds by default; individual tests override it.
    """
    stub = _SingleRecorder()
    monkeypatch.setattr(RUN_MODULE, "single", stub)
    monkeypatch.setattr(
        RUN_MODULE,
        "split",
        lambda ds, mode, random_state=42, quality=None: [_make_masks(fold_index=i) for i in range(2)],
    )
    monkeypatch.setattr(RUN_MODULE, "load", lambda name: _FakeDataset(name=name))
    return stub


def _set_folds(monkeypatch: pytest.MonkeyPatch, n_folds: int) -> None:
    monkeypatch.setattr(
        RUN_MODULE,
        "split",
        lambda ds, mode, random_state=42, quality=None: [_make_masks(fold_index=i) for i in range(n_folds)],
    )


def _set_randomizations(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    def fake_randomization(model_class, ds, modes, randomization_type="permutation", random_state=42):
        copies = []
        for index in range(n):
            copy = _FakeDataset(name=ds.name, tag=f"random_{index}")
            copy.randomization = (modes[0], f"view_{index}")
            copies.append(copy)
        return copies

    monkeypatch.setattr(RUN_MODULE, "randomization", fake_randomization)


class TestDatasetResolution:
    def test_a_dataset_name_is_loaded(self, recorder: _SingleRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
        loaded = []
        monkeypatch.setattr(RUN_MODULE, "load", lambda name: loaded.append(name) or _FakeDataset(name=name))

        run([_stub_model("A")], "CTRPv1", "LSO", hyperparameter_tuning=False)

        assert loaded == ["CTRPv1"]

    def test_the_loaded_dataset_is_the_one_handed_to_single(
        self, recorder: _SingleRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel = _FakeDataset(name="CTRPv1", tag="loaded")
        monkeypatch.setattr(RUN_MODULE, "load", lambda name: sentinel)

        run([_stub_model("A")], "CTRPv1", "LSO", hyperparameter_tuning=False)

        assert recorder.datasets == [sentinel, sentinel]

    def test_a_dataset_object_is_used_without_loading(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(RUN_MODULE, "load", lambda name: pytest.fail("load must not be called"))

        run([_stub_model("A")], dataset, "LSO", hyperparameter_tuning=False)

        assert recorder.datasets == [dataset, dataset]


class TestFanOut:
    @pytest.mark.parametrize(
        ("n_models", "n_folds"),
        [
            pytest.param(1, 1, id="1x1"),
            pytest.param(2, 3, id="2x3"),
            pytest.param(3, 2, id="3x2"),
        ],
    )
    def test_one_run_per_model_and_fold(
        self,
        recorder: _SingleRecorder,
        dataset: _FakeDataset,
        monkeypatch: pytest.MonkeyPatch,
        n_models: int,
        n_folds: int,
    ) -> None:
        _set_folds(monkeypatch, n_folds)
        models = [_stub_model(f"Model{i}") for i in range(n_models)]

        result = run(models, dataset, "LSO", hyperparameter_tuning=False)

        assert len(recorder.calls) == n_models * n_folds
        assert sum(m.n_folds for m in result.models) == n_models * n_folds

    @pytest.mark.parametrize("n_randomizations", [1, 3])
    def test_randomization_adds_one_run_per_randomized_dataset(
        self,
        recorder: _SingleRecorder,
        dataset: _FakeDataset,
        monkeypatch: pytest.MonkeyPatch,
        n_randomizations: int,
    ) -> None:
        n_models, n_folds = 2, 3
        _set_folds(monkeypatch, n_folds)
        _set_randomizations(monkeypatch, n_randomizations)
        models = [_stub_model(f"Model{i}") for i in range(n_models)]

        result = run(models, dataset, "LSO", randomization_modes=["SVRC"], hyperparameter_tuning=False)

        expected = n_models * n_folds * (1 + n_randomizations)
        assert len(recorder.calls) == expected
        assert sum(m.n_runs for m in result.models) == expected
        assert sum(m.n_folds for m in result.models) == n_models * n_folds

    def test_the_original_dataset_runs_before_its_randomized_copies(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)
        _set_randomizations(monkeypatch, 2)

        run([_stub_model("A")], dataset, "LSO", randomization_modes=["SVRC"], hyperparameter_tuning=False)

        assert [ds.tag for ds in recorder.datasets] == ["original", "random_0", "random_1"]

    def test_an_empty_randomization_mode_list_adds_nothing(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)
        monkeypatch.setattr(RUN_MODULE, "randomization", lambda *a, **k: pytest.fail("must not randomize"))

        run([_stub_model("A")], dataset, "LSO", randomization_modes=[], hyperparameter_tuning=False)

        assert len(recorder.calls) == 2

    def test_no_models_yields_no_runs_and_an_empty_experiment_is_rejected(
        self, recorder: _SingleRecorder, dataset: _FakeDataset
    ) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            run([], dataset, "LSO", hyperparameter_tuning=False)


class TestRobustness:
    @pytest.mark.parametrize("trials", [1, 2, 4])
    def test_trials_multiply_the_fold_count(
        self,
        recorder: _SingleRecorder,
        dataset: _FakeDataset,
        monkeypatch: pytest.MonkeyPatch,
        trials: int,
    ) -> None:
        n_folds = 3
        _set_folds(monkeypatch, n_folds)

        run([_stub_model("A")], dataset, "LSO", robustness_trials=trials, hyperparameter_tuning=False)

        assert len(recorder.calls) == n_folds * (1 + trials)

    def test_zero_trials_leaves_the_folds_untouched(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 3)

        run([_stub_model("A")], dataset, "LSO", robustness_trials=0, hyperparameter_tuning=False)

        assert len(recorder.calls) == 3
        assert all("robustness_trial" not in masks.metadata for masks in recorder.masks)

    def test_each_fold_records_its_trial_index(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)

        run([_stub_model("A")], dataset, "LSO", robustness_trials=3, hyperparameter_tuning=False)

        assert [masks.metadata.get("robustness_trial") for masks in recorder.masks] == [None, 0, 1, 2, None, 0, 1, 2]

    def test_the_trial_index_reaches_the_result_metadata(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)

        result = run([_stub_model("A")], dataset, "LSO", robustness_trials=2, hyperparameter_tuning=False)

        assert result.has_robustness
        assert [r.robustness_trial for r in result.models[0].runs] == [None, 0, 1]

    def test_the_original_fold_index_survives_the_multiplication(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)

        run([_stub_model("A")], dataset, "LSO", robustness_trials=2, hyperparameter_tuning=False)

        assert [masks.metadata["fold_index"] for masks in recorder.masks] == [0, 0, 0, 1, 1, 1]


class TestForwardedArguments:
    def test_hpo_settings_reach_single(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)

        run(
            [_stub_model("A")],
            dataset,
            "LSO",
            hyperparameter_tuning=True,
            hpo_metric="MSE",
            hpo_num_trials=4,
            hpo_random_state=7,
        )

        _, kwargs = recorder.calls[0]
        assert kwargs == {
            "hyperparameter_tuning": True,
            "hpo_metric": "MSE",
            "hpo_num_trials": 4,
            "hpo_random_state": 7,
            "seed": 42,
            "response_transformation": None,
            "wandb_project": None,
        }

    def test_the_split_mode_reaches_split(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        def fake_split(ds: Any, mode: str, random_state: int = 42, quality: Any = None) -> list[SplitMasks]:
            seen.append(mode)
            return [_make_masks(fold_index=0)]

        monkeypatch.setattr(RUN_MODULE, "split", fake_split)

        run([_stub_model("A")], dataset, "LDO", hyperparameter_tuning=False)

        assert seen == ["LDO"]

    def test_the_randomization_type_reaches_randomization(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)
        seen: list[dict[str, Any]] = []

        def fake_randomization(model_class, ds, modes, **kwargs):
            seen.append({"modes": modes, **kwargs})
            return []

        monkeypatch.setattr(RUN_MODULE, "randomization", fake_randomization)

        run(
            [_stub_model("A")],
            dataset,
            "LSO",
            randomization_modes=["SVRC", "SVRD"],
            randomization_type="invariant",
            hyperparameter_tuning=False,
        )

        assert seen == [{"modes": ["SVRC", "SVRD"], "randomization_type": "invariant", "random_state": 42}]

    def test_the_randomization_root_seed_reaches_randomization(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)
        seen: list[int] = []

        def fake_randomization(*args, **kwargs):
            seen.append(kwargs["random_state"])
            return []

        monkeypatch.setattr(RUN_MODULE, "randomization", fake_randomization)

        run([_stub_model("A")], dataset, "LSO", randomization_modes=["SVRC"], randomization_random_state=17)

        assert seen == [17]

    def test_the_seed_drives_split_and_defaults_randomization(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        split_seeds: list[int] = []
        randomization_seeds: list[int] = []

        def fake_split(ds: Any, mode: str, random_state: int = 42, quality: Any = None) -> list[SplitMasks]:
            split_seeds.append(random_state)
            return [_make_masks(fold_index=0)]

        def fake_randomization(*args, **kwargs):
            randomization_seeds.append(kwargs["random_state"])
            return []

        monkeypatch.setattr(RUN_MODULE, "split", fake_split)
        monkeypatch.setattr(RUN_MODULE, "randomization", fake_randomization)

        run([_stub_model("A")], dataset, "LSO", randomization_modes=["SVRC"], seed=5)

        assert split_seeds == [5]
        assert randomization_seeds == [5]
        _, kwargs = recorder.calls[0]
        assert kwargs["seed"] == 5
        assert kwargs["hpo_random_state"] is None

    def test_an_explicit_randomization_state_overrides_the_seed(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)
        seen: list[int] = []

        def fake_randomization(*args, **kwargs):
            seen.append(kwargs["random_state"])
            return []

        monkeypatch.setattr(RUN_MODULE, "randomization", fake_randomization)

        run([_stub_model("A")], dataset, "LSO", randomization_modes=["SVRC"], seed=5, randomization_random_state=0)

        assert seen == [0]


class TestGrouping:
    def test_results_are_grouped_by_model_name(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 3)
        models = [_stub_model("Alpha"), _stub_model("Beta")]

        result = run(models, dataset, "LSO", hyperparameter_tuning=False)

        assert isinstance(result, ExperimentResult)
        assert sorted(result.model_names) == ["Alpha", "Beta"]
        assert [m.n_folds for m in result.models] == [3, 3]

    def test_models_are_iterated_before_folds(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)
        models = [_stub_model("Alpha"), _stub_model("Beta")]

        run(models, dataset, "LSO", hyperparameter_tuning=False)

        assert recorder.model_names == ["Alpha", "Alpha", "Beta", "Beta"]

    def test_randomized_runs_are_grouped_with_their_model(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)
        _set_randomizations(monkeypatch, 2)

        result = run([_stub_model("Alpha")], dataset, "LSO", randomization_modes=["SVRC"], hyperparameter_tuning=False)

        assert result.model_names == ["Alpha"]
        assert result.has_randomization


class TestDatasetSequences:
    def test_several_datasets_are_loaded_and_concatenated_in_order(
        self, recorder: _SingleRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        joined = _FakeDataset(name="A+B", tag="joined")
        seen: list[list[Any]] = []
        monkeypatch.setattr(RUN_MODULE, "concat", lambda datasets: seen.append(list(datasets)) or joined)

        run([_stub_model("M")], ["A", "B"], "LSO", hyperparameter_tuning=False)

        assert [[d.name for d in datasets] for datasets in seen] == [["A", "B"]]
        assert recorder.datasets == [joined, joined]

    def test_names_and_dataset_objects_can_be_mixed(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[list[Any]] = []
        monkeypatch.setattr(RUN_MODULE, "concat", lambda datasets: seen.append(list(datasets)) or datasets[0])

        run([_stub_model("M")], [dataset, "B"], "LSO", hyperparameter_tuning=False)

        assert seen[0][0] is dataset
        assert seen[0][1].name == "B"

    def test_a_sequence_of_one_is_used_without_concatenation(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(RUN_MODULE, "concat", lambda datasets: pytest.fail("concat must not be called"))

        run([_stub_model("M")], [dataset], "LSO", hyperparameter_tuning=False)

        assert recorder.datasets == [dataset, dataset]

    def test_an_empty_sequence_is_rejected(self, recorder: _SingleRecorder) -> None:
        with pytest.raises(InvalidInputError, match="empty sequence"):
            run([_stub_model("M")], [], "LSO")

    def test_the_concatenated_dataset_is_the_one_split(
        self, recorder: _SingleRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        joined = _FakeDataset(name="A+B")
        monkeypatch.setattr(RUN_MODULE, "concat", lambda datasets: joined)
        split_datasets: list[Any] = []

        def fake_split(ds: Any, mode: str, random_state: int = 42, quality: Any = None) -> list[SplitMasks]:
            split_datasets.append(ds)
            return [_make_masks(fold_index=0)]

        monkeypatch.setattr(RUN_MODULE, "split", fake_split)

        run([_stub_model("M")], ["A", "B"], "LStO", hyperparameter_tuning=False)

        assert split_datasets == [joined]


class TestQualityAndTracking:
    def test_the_quality_rule_reaches_split(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rule = object()
        seen: list[Any] = []

        def fake_split(ds: Any, mode: str, random_state: int = 42, quality: Any = None) -> list[SplitMasks]:
            seen.append(quality)
            return [_make_masks(fold_index=0)]

        monkeypatch.setattr(RUN_MODULE, "split", fake_split)

        run([_stub_model("M")], dataset, "LSO", hyperparameter_tuning=False, quality=rule)

        assert seen == [rule]

    def test_the_default_quality_is_none(self, recorder: _SingleRecorder, dataset: _FakeDataset) -> None:
        import inspect

        assert inspect.signature(run).parameters["quality"].default is None

    def test_the_wandb_project_reaches_every_single_call(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 3)

        run([_stub_model("M")], dataset, "LSO", hyperparameter_tuning=False, wandb_project="proj")

        assert [kwargs["wandb_project"] for _, kwargs in recorder.calls] == ["proj"] * 3


class TestRandomizationHoisting:
    def test_randomization_is_called_once_per_model_not_per_fold(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 3)
        calls: list[str] = []

        def spy(model, ds, modes, **kwargs):
            calls.append(model.name)
            return []

        monkeypatch.setattr(RUN_MODULE, "randomization", spy)

        run([_stub_model("A"), _stub_model("B")], dataset, "LSO", randomization_modes=["SVRC"])

        assert calls == ["A", "B"]

    def test_every_fold_reuses_the_same_randomized_datasets(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)
        _set_randomizations(monkeypatch, 1)

        run([_stub_model("A")], dataset, "LSO", randomization_modes=["SVRC"], hyperparameter_tuning=False)

        randomized = [ds for ds in recorder.datasets if ds.randomization is not None]
        assert len(randomized) == 2
        assert randomized[0] is randomized[1]


class TestFailures:
    def test_one_failing_fold_is_recorded_and_the_other_runs_survive(
        self,
        recorder: _SingleRecorder,
        dataset: _FakeDataset,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _set_folds(monkeypatch, 3)
        recorder.fail_when = lambda model, ds, masks: masks.metadata["fold_index"] == 1

        with caplog.at_level("WARNING"):
            result = run([_stub_model("A")], dataset, "LSO", hyperparameter_tuning=False)

        assert [r.fold_index for r in result.models[0].runs] == [0, 2]
        assert [(f.model_name, f.fold_index, f.error_type, f.message) for f in result.failures] == [
            ("A", 1, "RuntimeError", "boom")
        ]
        assert any("A fold 1" in message and "boom" in message for message in caplog.messages)

    def test_a_failing_model_does_not_stop_the_others(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)
        recorder.fail_when = lambda model, ds, masks: model.name == "Bad"

        result = run([_stub_model("Bad"), _stub_model("Good")], dataset, "LSO", hyperparameter_tuning=False)

        assert result.model_names == ["Good"]
        assert [(f.model_name, f.fold_index) for f in result.failures] == [("Bad", 0), ("Bad", 1)]

    def test_a_failed_randomized_run_records_its_randomization(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)
        _set_randomizations(monkeypatch, 1)
        recorder.fail_when = lambda model, ds, masks: ds.randomization is not None

        result = run([_stub_model("A")], dataset, "LSO", randomization_modes=["SVRC"], hyperparameter_tuning=False)

        assert [f.randomization for f in result.failures] == [("SVRC", "view_0")]
        assert len(result.models[0].runs) == 1

    def test_a_failed_robustness_trial_records_its_index(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)
        recorder.fail_when = lambda model, ds, masks: masks.metadata.get("robustness_trial") == 1

        result = run([_stub_model("A")], dataset, "LSO", robustness_trials=2, hyperparameter_tuning=False)

        assert [(f.fold_index, f.robustness_trial) for f in result.failures] == [(0, 1)]

    def test_a_randomization_that_fails_is_recorded_per_fold_and_the_original_runs_stay(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)

        def broken(*args, **kwargs):
            raise KeyError("no such view")

        monkeypatch.setattr(RUN_MODULE, "randomization", broken)

        result = run([_stub_model("A")], dataset, "LSO", randomization_modes=["SVRC"], hyperparameter_tuning=False)

        assert [(f.fold_index, f.error_type) for f in result.failures] == [(0, "KeyError"), (1, "KeyError")]
        assert len(result.models[0].runs) == 2

    def test_every_run_failing_raises_with_the_failures(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)
        recorder.fail_when = lambda model, ds, masks: True

        with pytest.raises(ExperimentFailedError) as caught:
            run([_stub_model("A")], dataset, "LSO", hyperparameter_tuning=False)

        assert [f.fold_index for f in caught.value.failures] == [0, 1]

    def test_on_error_raise_reraises_the_first_error(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 3)
        recorder.fail_when = lambda model, ds, masks: masks.metadata["fold_index"] == 1
        recorder.error = ValueError("first")

        with pytest.raises(ValueError, match="first"):
            run([_stub_model("A")], dataset, "LSO", hyperparameter_tuning=False, on_error="raise")

        assert len(recorder.calls) == 2

    def test_on_error_raise_also_reraises_a_randomization_error(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*args, **kwargs):
            raise KeyError("no such view")

        monkeypatch.setattr(RUN_MODULE, "randomization", broken)

        with pytest.raises(KeyError):
            run([_stub_model("A")], dataset, "LSO", randomization_modes=["SVRC"], on_error="raise")

    def test_a_keyboard_interrupt_propagates(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)
        recorder.fail_when = lambda model, ds, masks: True
        recorder.error = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            run([_stub_model("A")], dataset, "LSO", hyperparameter_tuning=False)

        assert len(recorder.calls) == 1

    def test_an_unknown_on_error_is_rejected(self, recorder: _SingleRecorder, dataset: _FakeDataset) -> None:
        with pytest.raises(InvalidInputError, match="on_error"):
            run([_stub_model("A")], dataset, "LSO", on_error="ignore")  # type: ignore[arg-type]

    def test_a_clean_run_has_no_failures(self, recorder: _SingleRecorder, dataset: _FakeDataset) -> None:
        result = run([_stub_model("A")], dataset, "LSO", hyperparameter_tuning=False)

        assert result.failures == ()

    def test_failures_round_trip_through_save_and_load(
        self,
        recorder: _SingleRecorder,
        dataset: _FakeDataset,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        _set_folds(monkeypatch, 2)
        recorder.fail_when = lambda model, ds, masks: masks.metadata["fold_index"] == 1
        result = run([_stub_model("A")], dataset, "LSO", hyperparameter_tuning=False)

        result.save(tmp_path)
        loaded = ExperimentResult.load(tmp_path)

        assert (tmp_path / "failures.json").exists()
        assert loaded.failures == result.failures
