"""Tests for the ``drevalpy data`` commands ``load``, ``synthetic`` and ``concat``, plus the group surface."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli.data import data_app
from drevalpy.cli.data.concat import concat_datasets
from drevalpy.cli.data.load import load_dataset
from drevalpy.cli.data.split import split_dataset
from drevalpy.cli.data.synthetic import synthetic_dataset
from drevalpy.cli.main import app
from drevalpy.data._synthetic import MAX_SYNTHETIC_DRUGS, N_DRUGS, N_SAMPLES
from drevalpy.errors import UnknownNameError
from tests.cli._helpers import HELP_ENV, FakeDataset, GroupSurface, Recorder, make_runner, patch_worker, plain

runner = make_runner()


def _patch_loader(monkeypatch: pytest.MonkeyPatch, worker) -> None:
    patch_worker(monkeypatch, "drevalpy.data.datasets._load", "load", worker)
    patch_worker(monkeypatch, "drevalpy.data", "load", worker)


@pytest.fixture()
def dataset() -> FakeDataset:
    return FakeDataset(name="TOYv1")


@pytest.fixture()
def loader(monkeypatch: pytest.MonkeyPatch, dataset: FakeDataset) -> Recorder:
    recorder = Recorder(return_value=dataset)
    _patch_loader(monkeypatch, recorder)
    return recorder


def test_loads_by_name_and_writes_to_the_requested_path(
    loader: Recorder, dataset: FakeDataset, tmp_path: UPath
) -> None:
    out = tmp_path / "nested" / "deeper" / "out.h5mu"

    result = runner.invoke(app, ["data", "load", "TOYv1", str(out)])

    assert result.exit_code == 0, result.output
    assert loader.args == ("TOYv1",)
    assert out.parent.is_dir()
    assert dataset.mdata.written == [str(out)]
    assert f"Wrote TOYv1 to {out}" in plain(result.output)


def test_a_path_is_forwarded_unchanged(loader: Recorder, tmp_path: UPath) -> None:
    source = tmp_path / "local.h5mu"

    runner.invoke(app, ["data", "load", str(source), str(tmp_path / "out.h5mu")])

    assert loader.args == (str(source),)


def test_a_loader_error_is_reported_without_a_traceback_and_writes_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
) -> None:
    def boom(name: str) -> FakeDataset:
        raise UnknownNameError(f"Unknown dataset {name!r}")

    _patch_loader(monkeypatch, boom)
    out = tmp_path / "out.h5mu"

    result = runner.invoke(app, ["data", "load", "NoSuchDataset", str(out)])

    assert result.exit_code == 1
    assert plain(result.output).strip() == "Error: Unknown dataset 'NoSuchDataset'"
    assert not out.exists()


def test_output_is_required_and_help_documents_the_drug_limit() -> None:
    assert runner.invoke(app, ["data", "synthetic"], env=HELP_ENV).exit_code == 2

    helped = runner.invoke(app, ["data", "synthetic", "--help"], env=HELP_ENV)

    assert f"at most {MAX_SYNTHETIC_DRUGS}" in plain(helped.output)


def test_too_many_drugs_is_an_error_naming_the_limit_and_writes_no_file(tmp_path: UPath) -> None:
    out = tmp_path / "x.h5mu"

    result = runner.invoke(app, ["data", "synthetic", "--output", str(out), "--n-drugs", str(MAX_SYNTHETIC_DRUGS + 1)])

    assert result.exit_code == 1
    assert plain(result.output).startswith("Error: ")
    assert str(MAX_SYNTHETIC_DRUGS) in plain(result.output)
    assert not out.exists()


@pytest.mark.slow
class TestSyntheticWriting:
    def test_writes_a_loadable_dataset_with_the_default_shape(self, tmp_path: UPath) -> None:
        import drevalpy

        out = tmp_path / "nested" / "toy.h5mu"
        result = runner.invoke(app, ["data", "synthetic", "--output", str(out)])

        assert result.exit_code == 0, result.output
        assert str(out) in plain(result.output)
        loaded = drevalpy.load(str(out))
        assert loaded.response_matrix.shape == (N_SAMPLES, N_DRUGS)
        assert "gene_expression" in loaded.mdata.mod

    def test_options_set_the_loaded_dataset(self, tmp_path: UPath) -> None:
        import drevalpy
        from drevalpy.data.utils import TISSUE_IDENTIFIER

        args = ["--n-samples", "40", "--n-drugs", "5", "--n-tissues", "4", "--name", "STUDY_A"]
        result = runner.invoke(app, ["data", "synthetic", "--output", str(tmp_path / "big.h5mu"), *args])
        loaded = drevalpy.load(str(tmp_path / "big.h5mu"))

        assert result.exit_code == 0, result.output
        assert loaded.name == "STUDY_A"
        assert loaded.response_matrix.shape == (40, 5)
        assert loaded.response.obs[TISSUE_IDENTIFIER].nunique() == 4

    def test_the_seed_changes_the_responses_and_the_drug_limit_itself_is_accepted(self, tmp_path: UPath) -> None:
        import drevalpy

        for seed in (1, 2):
            runner.invoke(app, ["data", "synthetic", "--output", str(tmp_path / f"{seed}.h5mu"), "--seed", str(seed)])
        first, second = (drevalpy.load(str(tmp_path / f"{seed}.h5mu")) for seed in (1, 2))
        limit = runner.invoke(
            app, ["data", "synthetic", "--output", str(tmp_path / "x.h5mu"), "--n-drugs", str(MAX_SYNTHETIC_DRUGS)]
        )

        assert not np.array_equal(first.response_matrix, second.response_matrix, equal_nan=True)
        assert limit.exit_code == 0, limit.output


FEATURES = [f"FEATURE{i:04d}" for i in range(12)]


def test_help_lists_the_options_and_output_is_required() -> None:
    helped = runner.invoke(app, ["data", "concat", "--help"], env=HELP_ENV)

    assert helped.exit_code == 0
    assert "--join" in plain(helped.output)
    assert "--name" in plain(helped.output)
    assert runner.invoke(app, ["data", "concat"], env=HELP_ENV).exit_code == 2


@pytest.mark.parametrize("value", ["exact", "inner", "INNER"])
def test_join_choices_are_accepted_case_insensitively(
    monkeypatch: pytest.MonkeyPatch, tmp_path: UPath, value: str
) -> None:
    recorder = Recorder(return_value=FakeDataset(name="BOTH"))
    monkeypatch.setattr("drevalpy.data.load", lambda item: item)
    monkeypatch.setattr("drevalpy.data.concat", recorder)

    result = runner.invoke(app, ["data", "concat", "a", "b", str(tmp_path / "out.h5mu"), "--join", value])

    assert result.exit_code == 0, result.output
    assert recorder.kwargs["join"] == value.lower()


def test_an_unknown_join_is_a_usage_error() -> None:
    result = runner.invoke(app, ["data", "concat", "a", "b", "o.h5mu", "--join", "outer"], env=HELP_ENV)

    assert result.exit_code == 2


@pytest.mark.slow
class TestConcatWriting:
    @pytest.fixture
    def inputs(self, tmp_path: UPath) -> tuple[str, str]:
        import drevalpy

        first = tmp_path / "first.h5mu"
        second = tmp_path / "second.h5mu"
        drevalpy.synthetic(name="FIRST", n_samples=12, n_drugs=4, feature_names=FEATURES).save(str(first))
        drevalpy.synthetic(name="SECOND", n_samples=10, n_drugs=6, seed=7, feature_names=FEATURES).save(str(second))
        return str(first), str(second)

    def test_writes_the_concatenated_dataset_and_name_option_names_it(
        self, tmp_path: UPath, inputs: tuple[str, str]
    ) -> None:
        import drevalpy

        out = tmp_path / "nested" / "both.h5mu"
        named = tmp_path / "named.h5mu"
        result = runner.invoke(app, ["data", "concat", *inputs, str(out)])
        renamed = runner.invoke(app, ["data", "concat", *inputs, str(named), "--name", "BOTH"])

        assert result.exit_code == 0, result.output
        assert renamed.exit_code == 0, renamed.output
        combined = drevalpy.load(str(out))
        assert combined.name == "FIRST+SECOND"
        assert combined.response_matrix.shape == (22, 6)
        assert set(combined.sample_column("study", combined.sample_ids)) == {"FIRST", "SECOND"}
        assert drevalpy.load(str(named)).name == "BOTH"

    def test_join_inner_is_forwarded(self, tmp_path: UPath, inputs: tuple[str, str]) -> None:
        import drevalpy

        first, second = inputs
        narrow = tmp_path / "narrow.h5mu"
        drevalpy.synthetic(name="NARROW", n_samples=8, feature_names=FEATURES[:8]).save(str(narrow))
        out = tmp_path / "both.h5mu"

        exact = runner.invoke(app, ["data", "concat", first, str(narrow), str(out)])
        inner = runner.invoke(app, ["data", "concat", first, str(narrow), str(out), "--join", "inner"])

        assert exact.exit_code == 1
        assert plain(exact.output).startswith("Error: ")
        assert inner.exit_code == 0, inner.output
        assert drevalpy.load(str(out)).mdata.mod["gene_expression"].n_vars == 8

    def test_bad_inputs_are_reported_and_write_nothing(self, tmp_path: UPath, inputs: tuple[str, str]) -> None:
        out = tmp_path / "both.h5mu"
        single = runner.invoke(app, ["data", "concat", inputs[0], str(out)])
        unknown = runner.invoke(app, ["data", "concat", inputs[0], str(tmp_path / "nope.h5mu"), str(out)])

        assert single.exit_code == 1
        assert "at least two" in single.output
        assert unknown.exit_code == 1
        assert plain(unknown.output).startswith("Error: ")
        assert "Traceback" not in unknown.output
        assert not out.exists()


class TestGroup(GroupSurface):
    group = data_app
    path = "data"
    callbacks = {
        "concat": concat_datasets,
        "load": load_dataset,
        "split": split_dataset,
        "synthetic": synthetic_dataset,
    }
