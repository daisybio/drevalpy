"""Tests for :mod:`drevalpy.cli.data.concat`, the ``drevalpy data concat`` command."""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, plain

runner = make_runner()

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
class TestWriting:
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
