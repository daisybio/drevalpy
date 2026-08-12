"""Tests for the top-level ``run`` orchestrator.

``drevalpy/run.py`` contains no numerical logic of its own: it loads or accepts a
dataset, expands it into folds, optionally multiplies those folds by robustness
trials, optionally adds randomized copies of the dataset, and hands every
combination to :func:`drevalpy.single.single`. These tests therefore replace
``single`` with a recorder and assert the fan-out arithmetic and the kwargs it
forwards, so nothing here trains a model.

``robustness`` is *not* stubbed: it is four statements over real ``SplitMask``
objects, so exercising it for real is cheaper than faking it and lets the
``fold_metadata["robustness_trial"]`` assertion cover the whole path.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from drevalpy.run import run
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.results import ExperimentResult
from tests.synthetic.results import DEFAULT_DATASET_NAME, make_run_result

#: ``drevalpy/__init__.py`` re-exports ``run`` as a *function*, which shadows the
#: same-named submodule on the package object, so ``monkeypatch.setattr`` with a
#: dotted string walks into the function and raises ``AttributeError``. Resolving
#: the module through ``importlib`` sidesteps the shadowing; the same trick is
#: used by ``tests/cli/_helpers.py::patch_worker``.
RUN_MODULE = importlib.import_module("drevalpy.run")


class _FakeDataset:
    """Stand-in for ``Dataset`` exposing only what ``run`` and the stubs read."""

    def __init__(self, name: str = DEFAULT_DATASET_NAME, tag: str = "original") -> None:
        self.name = name
        self.tag = tag
        self.randomization: tuple[str, str] | None = None


def _stub_model(name: str) -> type:
    """Build a minimal ``DRPModel`` stand-in identified by ``name``."""

    class _StubModel:
        @classmethod
        def get_model_name(cls) -> str:
            return name

    _StubModel.__name__ = name
    return _StubModel


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
        metadata={"fold_index": fold_index, "split_mode": "LCO", "fold_id": f"fold_{fold_index}"},
    )


class _SingleRecorder:
    """Records every ``single`` call and returns a matching ``RunResult``."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, model_class: type, mudataset: Any, split_masks: SplitMasks, **kwargs: Any):
        self.calls.append(((model_class, mudataset, split_masks), kwargs))
        return make_run_result(
            model_name=model_class.get_model_name(),
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
        return [args[0].get_model_name() for args, _ in self.calls]


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
    monkeypatch.setattr(RUN_MODULE, "split", lambda ds, mode: [_make_masks(fold_index=i) for i in range(2)])
    monkeypatch.setattr(RUN_MODULE, "load", lambda name: _FakeDataset(name=name))
    return stub


def _set_folds(monkeypatch: pytest.MonkeyPatch, n_folds: int) -> None:
    monkeypatch.setattr(RUN_MODULE, "split", lambda ds, mode: [_make_masks(fold_index=i) for i in range(n_folds)])


def _set_randomizations(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    def fake_randomization(model_class, ds, modes, randomization_type="permutation"):
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

        run([_stub_model("A")], "CTRPv1", "LCO", hyperparameter_tuning=False)

        assert loaded == ["CTRPv1"]

    def test_the_loaded_dataset_is_the_one_handed_to_single(
        self, recorder: _SingleRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel = _FakeDataset(name="CTRPv1", tag="loaded")
        monkeypatch.setattr(RUN_MODULE, "load", lambda name: sentinel)

        run([_stub_model("A")], "CTRPv1", "LCO", hyperparameter_tuning=False)

        assert recorder.datasets == [sentinel, sentinel]

    def test_a_dataset_object_is_used_without_loading(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(RUN_MODULE, "load", lambda name: pytest.fail("load must not be called"))

        run([_stub_model("A")], dataset, "LCO", hyperparameter_tuning=False)

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

        result = run(models, dataset, "LCO", hyperparameter_tuning=False)

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

        result = run(models, dataset, "LCO", randomization_modes=["SVRC"], hyperparameter_tuning=False)

        expected = n_models * n_folds * (1 + n_randomizations)
        assert len(recorder.calls) == expected
        assert sum(m.n_folds for m in result.models) == expected

    def test_the_original_dataset_runs_before_its_randomized_copies(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)
        _set_randomizations(monkeypatch, 2)

        run([_stub_model("A")], dataset, "LCO", randomization_modes=["SVRC"], hyperparameter_tuning=False)

        assert [ds.tag for ds in recorder.datasets] == ["original", "random_0", "random_1"]

    def test_an_empty_randomization_mode_list_adds_nothing(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)
        monkeypatch.setattr(RUN_MODULE, "randomization", lambda *a, **k: pytest.fail("must not randomize"))

        run([_stub_model("A")], dataset, "LCO", randomization_modes=[], hyperparameter_tuning=False)

        assert len(recorder.calls) == 2

    def test_no_models_yields_no_runs_and_an_empty_experiment_is_rejected(
        self, recorder: _SingleRecorder, dataset: _FakeDataset
    ) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            run([], dataset, "LCO", hyperparameter_tuning=False)


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

        run([_stub_model("A")], dataset, "LCO", robustness_trials=trials, hyperparameter_tuning=False)

        assert len(recorder.calls) == n_folds * trials

    def test_zero_trials_leaves_the_folds_untouched(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 3)

        run([_stub_model("A")], dataset, "LCO", robustness_trials=0, hyperparameter_tuning=False)

        assert len(recorder.calls) == 3
        assert all("robustness_trial" not in masks.metadata for masks in recorder.masks)

    def test_each_fold_records_its_trial_index(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)

        run([_stub_model("A")], dataset, "LCO", robustness_trials=3, hyperparameter_tuning=False)

        assert [masks.metadata["robustness_trial"] for masks in recorder.masks] == [0, 1, 2, 0, 1, 2]

    def test_the_trial_index_reaches_the_result_metadata(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)

        result = run([_stub_model("A")], dataset, "LCO", robustness_trials=2, hyperparameter_tuning=False)

        assert result.has_robustness
        assert [r.fold_metadata["robustness_trial"] for r in result.models[0].runs] == [0, 1]

    def test_the_original_fold_index_survives_the_multiplication(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)

        run([_stub_model("A")], dataset, "LCO", robustness_trials=2, hyperparameter_tuning=False)

        assert [masks.metadata["fold_index"] for masks in recorder.masks] == [0, 0, 1, 1]


class TestForwardedArguments:
    def test_hpo_settings_reach_single(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)

        run(
            [_stub_model("A")],
            dataset,
            "LCO",
            hyperparameter_tuning=True,
            hpo_metric="MSE",
            hpo_num_samples=4,
            hpo_random_state=7,
            precomputed_only=True,
        )

        _, kwargs = recorder.calls[0]
        assert kwargs == {
            "hyperparameter_tuning": True,
            "hpo_metric": "MSE",
            "hpo_num_samples": 4,
            "hpo_random_state": 7,
            "precomputed_only": True,
        }

    def test_the_split_mode_reaches_split(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        def fake_split(ds: Any, mode: str) -> list[SplitMasks]:
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
            "LCO",
            randomization_modes=["SVRC", "SVRD"],
            randomization_type="invariant",
            hyperparameter_tuning=False,
        )

        assert seen == [{"modes": ["SVRC", "SVRD"], "randomization_type": "invariant"}]


class TestGrouping:
    def test_results_are_grouped_by_model_name(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 3)
        models = [_stub_model("Alpha"), _stub_model("Beta")]

        result = run(models, dataset, "LCO", hyperparameter_tuning=False)

        assert isinstance(result, ExperimentResult)
        assert sorted(result.model_names) == ["Alpha", "Beta"]
        assert [m.n_folds for m in result.models] == [3, 3]

    def test_models_are_iterated_before_folds(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 2)
        models = [_stub_model("Alpha"), _stub_model("Beta")]

        run(models, dataset, "LCO", hyperparameter_tuning=False)

        assert recorder.model_names == ["Alpha", "Alpha", "Beta", "Beta"]

    def test_randomized_runs_are_grouped_with_their_model(
        self, recorder: _SingleRecorder, dataset: _FakeDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_folds(monkeypatch, 1)
        _set_randomizations(monkeypatch, 2)

        result = run([_stub_model("Alpha")], dataset, "LCO", randomization_modes=["SVRC"], hyperparameter_tuning=False)

        assert result.model_names == ["Alpha"]
        assert result.has_randomization
