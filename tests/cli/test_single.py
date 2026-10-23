"""Tests for :mod:`drevalpy.cli.single`, the ``drevalpy single`` command."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.types import SplitMask, SplitMasks
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain

runner = make_runner()


class StubRunResult:
    """Stand-in for the ``RunResult`` that :func:`drevalpy.single` returns."""

    model_name = "StubElasticNet"
    dataset_name = "StubDataset"
    fold_index = 3

    def __init__(self) -> None:
        """Start with an empty save log."""
        self.saved: list[str] = []

    def save(self, path: str) -> None:
        """Record the destination the command derived.

        :param path: Output file path chosen by the command.
        """
        self.saved.append(path)


def _write_split(path: UPath, *, metadata: dict[str, object] | None = None) -> None:
    """Write a real 2x2 ``SplitMasks`` .npz so ``SplitMasks.load`` stays unpatched."""
    train = np.zeros((2, 2), dtype=bool)
    train[0, 0] = True
    test = np.zeros((2, 2), dtype=bool)
    test[1, 1] = True
    val = np.zeros((2, 2), dtype=bool)
    val[0, 1] = True
    SplitMasks(
        train=SplitMask(train),
        test=SplitMask(test),
        val=SplitMask(val),
        metadata=dict(metadata or {"fold_index": 3}),
    ).save(path)


@pytest.fixture()
def split_file(tmp_path: UPath) -> UPath:
    """Write a fold .npz whose metadata deliberately lacks ``split_mode``."""
    path = tmp_path / "fold_0.npz"
    _write_split(path)
    return path


@pytest.fixture()
def dataset() -> FakeDataset:
    """Provide the dataset stub ``Dataset.load`` is made to return."""
    return FakeDataset()


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch, dataset: FakeDataset) -> Recorder:
    """Patch every lazy import of ``single_cmd`` except ``SplitMasks``.

    ``SplitMasks.load`` is left real so the ``split_mode`` injection is asserted
    against the genuine round-trip rather than a stub's attribute.

    :param monkeypatch: Fixture used to replace the source-module workers.
    :param dataset: Stub returned by the patched ``Dataset.load``.
    :returns: Recorder standing in for :func:`drevalpy.single`.
    """
    recorder = Recorder(return_value=StubRunResult())
    monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", lambda name: type(f"Stub{name}", (), {}))
    monkeypatch.setattr("drevalpy.types.data.dataset.Dataset.load", classmethod(lambda cls, path: dataset))
    patch_worker(monkeypatch, "drevalpy._single", "single", recorder)
    return recorder


def _invoke(split_file: UPath, tmp_path: UPath, *extra: str):
    out = tmp_path / "out" / "result.npz"
    return runner.invoke(
        app,
        ["single", "ElasticNet", str(tmp_path / "ds.h5mu"), str(split_file), str(out), *extra],
    )


class TestArguments:
    """All four positional arguments are required."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["single"], id="none"),
            pytest.param(["single", "ElasticNet"], id="model-only"),
            pytest.param(["single", "ElasticNet", "ds.h5mu"], id="missing-split-and-output"),
            pytest.param(["single", "ElasticNet", "ds.h5mu", "fold.npz"], id="missing-output"),
        ],
    )
    def test_missing_positional_arguments_are_usage_errors(self, worker: Recorder, argv: list[str]) -> None:
        result = runner.invoke(app, argv, env=HELP_ENV)

        assert result.exit_code == 2


class TestForwarding:
    """Options map onto :func:`drevalpy.single` keywords."""

    def test_exits_cleanly(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        result = _invoke(split_file, tmp_path)

        assert result.exit_code == 0, result.output

    def test_passes_model_class_dataset_and_masks_positionally(
        self, worker: Recorder, split_file: UPath, tmp_path: UPath, dataset: FakeDataset
    ) -> None:
        _invoke(split_file, tmp_path)

        model_class, passed_dataset, split_masks = worker.args
        assert model_class.__name__ == "StubElasticNet"
        assert passed_dataset is dataset
        assert isinstance(split_masks, SplitMasks)

    def test_forwards_hpo_defaults(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        """Exhaustive on the keyword set, so a newly forwarded option cannot slip in unnoticed.

        ``response_transformation`` is popped rather than compared because sklearn
        transformers have no value equality; :class:`TestResponseTransformation`
        pins the instance it must be.
        """
        _invoke(split_file, tmp_path)

        forwarded = dict(worker.kwargs)
        forwarded.pop("response_transformation")
        assert forwarded == {
            "hyperparameter_tuning": True,
            "hpo_metric": "RMSE",
            "hpo_num_trials": 16,
            "hpo_random_state": None,
            "seed": 42,
            "wandb_project": None,
            "n_jobs": 1,
        }

    def test_forwards_n_jobs(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path, "--n-jobs", "2")

        assert worker.kwargs["n_jobs"] == 2

    def test_no_hpo_disables_tuning(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path, "--no-hpo")

        assert worker.kwargs["hyperparameter_tuning"] is False

    def test_forwards_overridden_hpo_options(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path, "--hpo-metric", "Pearson", "--hpo-trials", "2", "--hpo-random-state", "9")

        assert worker.kwargs["hpo_metric"] == "Pearson"
        assert worker.kwargs["hpo_num_trials"] == 2
        assert worker.kwargs["hpo_random_state"] == 9

    def test_seed_is_forwarded(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path, "--seed", "3")

        assert worker.kwargs["seed"] == 3
        assert worker.kwargs["hpo_random_state"] is None

    def test_wandb_project_is_forwarded(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path, "--wandb-project", "my-project")

        assert worker.kwargs["wandb_project"] == "my-project"

    def test_split_mode_is_not_a_worker_keyword(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        """``--split-mode`` only ever reaches the worker through the masks' metadata."""
        _invoke(split_file, tmp_path, "--split-mode", "LSO")

        assert "split_mode" not in worker.kwargs


class TestResponseTransformation:
    """``--response-transformation`` is forwarded as the unfitted prototype; the choices are in ``test_options``."""

    def test_the_default_applies_no_transformation(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path)

        assert worker.kwargs["response_transformation"] is None

    def test_a_choice_is_forwarded_as_its_unfitted_transformer(
        self, worker: Recorder, split_file: UPath, tmp_path: UPath
    ) -> None:
        """``single`` fits a clone per scope, so the CLI must not fit anything itself."""
        _invoke(split_file, tmp_path, "--response-transformation", "standard")

        transformer = worker.kwargs["response_transformation"]
        assert isinstance(transformer, StandardScaler)
        assert not hasattr(transformer, "mean_")

    def test_an_unknown_option_is_rejected(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        result = _invoke(split_file, tmp_path, "--response-transformation", "logarithmic")

        assert result.exit_code == 2
        assert worker.call_count == 0


class TestSplitModeInjection:
    """``--split-mode`` is a fallback: it fills the gap, it does not override."""

    def test_injected_when_absent_from_metadata(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path, "--split-mode", "LSO")

        assert worker.args[2].metadata["split_mode"] == "LSO"

    def test_the_lco_alias_is_injected_as_lso(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path, "--split-mode", "LCO")

        assert worker.args[2].metadata["split_mode"] == "LSO"

    def test_default_is_lpo(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path)

        assert worker.args[2].metadata["split_mode"] == "LPO"

    def test_existing_metadata_wins(self, worker: Recorder, tmp_path: UPath) -> None:
        split_file = tmp_path / "fold_with_mode.npz"
        _write_split(split_file, metadata={"fold_index": 3, "split_mode": "LDO"})

        _invoke(split_file, tmp_path, "--split-mode", "LSO")

        assert worker.args[2].metadata["split_mode"] == "LDO"

    def test_a_stored_lco_mode_is_read_as_lso(self, worker: Recorder, tmp_path: UPath) -> None:
        split_file = tmp_path / "fold_with_old_mode.npz"
        _write_split(split_file, metadata={"fold_index": 3, "split_mode": "LCO"})

        _invoke(split_file, tmp_path)

        assert worker.args[2].metadata["split_mode"] == "LSO"

    def test_other_metadata_is_preserved(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path)

        assert worker.args[2].metadata["fold_index"] == 3


class TestOutput:
    """The command creates the output *parent* and echoes a one-line summary."""

    def test_creates_the_parent_directory(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path)

        assert (tmp_path / "out").is_dir()

    def test_saves_to_the_requested_path(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path)

        assert worker.return_value.saved == [str(tmp_path / "out" / "result.npz")]

    def test_echoes_model_dataset_fold_and_destination(
        self, worker: Recorder, split_file: UPath, tmp_path: UPath
    ) -> None:
        result = _invoke(split_file, tmp_path)

        expected = f"Result: StubElasticNet on StubDataset (fold 3) -> {tmp_path / 'out' / 'result.npz'}"
        assert expected in plain(result.output)


class TestMissingSplitFile:
    """A missing fold file is reported as an error before anything is created or run."""

    def test_nonexistent_split_is_reported_without_a_traceback(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path / "absent.npz", tmp_path)

        assert result.exit_code == 1
        assert plain(result.output).startswith("Error: ")
        assert "Traceback" not in result.output

    def test_worker_is_not_called_and_no_output_directory_is_made(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path / "absent.npz", tmp_path)

        assert worker.call_count == 0
        assert not (tmp_path / "out").exists()
