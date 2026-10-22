"""Tests for :mod:`drevalpy.cli.data.concat`, the ``drevalpy data concat`` command."""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()

FEATURES = [f"FEATURE{i:04d}" for i in range(12)]


class TestArguments:
    def test_help_lists_the_options(self) -> None:
        result = runner.invoke(app, ["data", "concat", "--help"], env=HELP_ENV)

        text = plain(result.output)
        assert result.exit_code == 0
        assert "--join" in text
        assert "--name" in text

    def test_output_is_required(self) -> None:
        result = runner.invoke(app, ["data", "concat"], env=HELP_ENV)

        assert result.exit_code == 2


class TestWriting:
    """Each case writes and reads ``.h5mu`` files, which costs over 0.2s."""

    pytestmark = pytest.mark.slow

    @pytest.fixture
    def inputs(self, tmp_path: UPath) -> tuple[str, str]:
        import drevalpy

        first = tmp_path / "first.h5mu"
        second = tmp_path / "second.h5mu"
        drevalpy.synthetic(name="FIRST", n_samples=12, n_drugs=4, feature_names=FEATURES).save(str(first))
        drevalpy.synthetic(name="SECOND", n_samples=10, n_drugs=6, seed=7, feature_names=FEATURES).save(str(second))
        return str(first), str(second)

    def test_writes_the_concatenated_dataset(self, tmp_path: UPath, inputs: tuple[str, str]) -> None:
        import drevalpy

        out = tmp_path / "nested" / "both.h5mu"
        result = runner.invoke(app, ["data", "concat", *inputs, str(out)])

        assert result.exit_code == 0, result.output
        combined = drevalpy.load(str(out))
        assert combined.name == "FIRST+SECOND"
        assert combined.response_matrix.shape == (22, 6)
        assert set(combined.sample_column("study", combined.sample_ids)) == {"FIRST", "SECOND"}

    def test_name_option_names_the_result(self, tmp_path: UPath, inputs: tuple[str, str]) -> None:
        import drevalpy

        out = tmp_path / "both.h5mu"
        result = runner.invoke(app, ["data", "concat", *inputs, str(out), "--name", "BOTH"])

        assert result.exit_code == 0, result.output
        assert drevalpy.load(str(out)).name == "BOTH"

    def test_join_inner_is_forwarded(self, tmp_path: UPath, inputs: tuple[str, str]) -> None:
        import drevalpy

        first, second = inputs
        narrow = tmp_path / "narrow.h5mu"
        drevalpy.synthetic(name="NARROW", n_samples=8, feature_names=FEATURES[:8]).save(str(narrow))
        out = tmp_path / "both.h5mu"

        exact = runner.invoke(app, ["data", "concat", first, str(narrow), str(out)])
        inner = runner.invoke(app, ["data", "concat", first, str(narrow), str(out), "--join", "inner"])

        assert exact.exit_code == 1
        assert inner.exit_code == 0, inner.output
        assert drevalpy.load(str(out)).mdata.mod["gene_expression"].n_vars == 8

    def test_a_single_dataset_is_an_error_and_writes_nothing(self, tmp_path: UPath, inputs: tuple[str, str]) -> None:
        out = tmp_path / "both.h5mu"
        result = runner.invoke(app, ["data", "concat", inputs[0], str(out)])

        assert result.exit_code == 1
        assert "at least two" in result.output
        assert not out.exists()

    def test_an_unknown_dataset_is_reported(self, tmp_path: UPath, inputs: tuple[str, str]) -> None:
        result = runner.invoke(
            app, ["data", "concat", inputs[0], str(tmp_path / "nope.h5mu"), str(tmp_path / "o.h5mu")]
        )

        assert result.exit_code != 0
