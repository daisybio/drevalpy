"""Tests for :mod:`drevalpy.visualization.report`."""

from __future__ import annotations

import gc
import logging
import re
import weakref
from collections.abc import Iterator
from typing import Any, NamedTuple

import multiqc
import pytest
from matplotlib import pyplot as plt
from upath import UPath

from drevalpy.registry.visualization import visualization_registry
from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.base import PlotlyVisualization, Section, Visualization
from drevalpy.visualization.report import (
    _add_module,
    _applicable_results,
    _ensure_experiment,
    create_report,
    save_all_png,
)
from tests.synthetic import (
    REFERENCE_MODEL,
    make_experiment_result,
    make_model_result,
    make_run_failure,
    make_run_result,
)


class _StubViz(Visualization):
    """Records what it was asked to compute and emits one trivial section."""

    registry_name = "stub_viz"
    sections: list[Section] = [Section(name="Stub", anchor="stub_section", content="<p>stub</p>")]
    last: Any = None

    def __init__(self) -> None:
        self.computed: list[Any] = []
        type(self).last = self

    def compute(self, result) -> None:
        self.computed.append(result)

    def to_png(self, path) -> None:
        UPath(path).write_bytes(b"stub-png")

    def to_multiqc(self) -> list[Section]:
        return list(self.sections)

    def show(self) -> None:  # pragma: no cover - not part of the report path
        raise AssertionError("show() is not used by the report")


@pytest.fixture(scope="module", autouse=True)
def _isolate_multiqc_config() -> Iterator[None]:
    """Reset MultiQC's global config once per module; per-test resets clear only report state."""
    multiqc.reset()
    yield
    multiqc.reset()


@pytest.fixture(autouse=True)
def _reset_multiqc() -> Iterator[None]:
    """Isolate MultiQC report state and the leaderboard's rcParams edits."""
    multiqc.report.reset()
    with plt.rc_context():
        yield
    plt.close("all")
    multiqc.report.reset()


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result()


@pytest.fixture
def only_stub_is_applicable(monkeypatch) -> type[_StubViz]:
    """Restrict the report to a single cheap, experiment-level visualization."""
    _make_applicable(monkeypatch, _StubViz, "ExperimentResult")
    return _StubViz


def _make_applicable(monkeypatch, viz_cls: type[Visualization], result_type: str) -> None:
    """Make *viz_cls* the only applicable visualization, consuming *result_type*."""
    monkeypatch.setattr(visualization_registry, "applicable", lambda experiment: [viz_cls])
    monkeypatch.setattr(visualization_registry, "result_type", lambda name: result_type)


class _BuiltReport(NamedTuple):
    """A shared build: ``modules`` is snapshotted while the build was current."""

    out: UPath
    modules: list[Any]
    records: list[logging.LogRecord]


def _build_report(out: UPath, result, **kwargs) -> _BuiltReport:
    """Run ``create_report`` once, capturing MultiQC modules and the module logger's records."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger("drevalpy.visualization.report")
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    multiqc.reset()
    try:
        with plt.rc_context():
            create_report(result, out, **kwargs)
            modules = list(multiqc.report.modules)
    finally:
        logger.setLevel(previous_level)
        logger.removeHandler(handler)
        plt.close("all")
    return _BuiltReport(out, modules, records)


@pytest.fixture(scope="module")
def normalized_real_report(tmp_path_factory) -> _BuiltReport:
    """A normalized report from the real plots (regression: an all-NaN PCC column broke the leaderboard)."""
    return _build_report(
        tmp_path_factory.mktemp("normalized_report"),
        make_experiment_result(n_models=4, n_folds=3),
        title="Normalized",
        reference_model=REFERENCE_MODEL,
    )


@pytest.fixture(scope="module")
def logged_stub_report(experiment: ExperimentResult, tmp_path_factory) -> _BuiltReport:
    """One stub-plot build, shared by the tests that only read its log."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        _make_applicable(monkeypatch, _StubViz, "ExperimentResult")
        return _build_report(tmp_path_factory.mktemp("logged"), experiment)


class TestEnsureExperiment:
    def test_wraps_runs_and_models_and_passes_experiments_through(self, experiment):
        wrapped_run = _ensure_experiment(make_run_result(model_name="ElasticNet"))
        wrapped_model = _ensure_experiment(make_model_result(n_folds=3))

        assert wrapped_run.model_names == ["ElasticNet"]
        assert wrapped_model.max_folds == 3
        assert _ensure_experiment(experiment) is experiment


class TestAddModule:
    def test_appends_one_module_and_empty_content_becomes_a_string(self):
        _add_module([Section(name="S", anchor="s", content="x"), Section(name="T", anchor="t")], "My module", "my")

        assert [m.name for m in multiqc.report.modules] == ["My module"]
        assert [s.name for s in multiqc.report.modules[0].sections] == ["S", "T"]
        assert multiqc.report.modules[0].sections[1].content == ""


class TestApplicableResults:
    def test_experiment_plots_get_the_experiment_and_model_plots_get_each_model(self, experiment, monkeypatch):
        _make_applicable(monkeypatch, _StubViz, "ExperimentResult")
        assert list(_applicable_results(experiment)) == [(_StubViz, experiment)]

        _make_applicable(monkeypatch, _StubViz, "ModelResult")
        pairs = list(_applicable_results(experiment))
        assert [cls for cls, _ in pairs] == [_StubViz] * len(experiment.models)
        assert [model.model_name for _, model in pairs] == experiment.model_names

        monkeypatch.setattr(visualization_registry, "applicable", lambda experiment: [])
        assert list(_applicable_results(experiment)) == []

    def test_real_plots_resolve_their_result_type_through_the_registry(self, experiment):
        pairs = list(_applicable_results(experiment))

        assert "leaderboard" in {cls.registry_name for cls, result in pairs if result is experiment}
        assert all(
            (result is experiment) == (visualization_registry.result_type(cls.registry_name) == "ExperimentResult")
            for cls, result in pairs
        )


class TestCreateReport:
    pytestmark = pytest.mark.slow

    def test_writes_a_multiqc_html_report_for_every_applicable_plot(self, experiment, tmp_path):
        create_report(experiment, tmp_path / "report", title="W4 Report")

        assert [p.name for p in (tmp_path / "report").glob("*.html")] == ["W4-Report_multiqc_report.html"]

    def test_creates_directories_and_accepts_a_string_path(self, experiment, tmp_path, only_stub_is_applicable):
        out = tmp_path / "deep" / "nested"

        create_report(experiment, str(out))

        assert any(out.glob("*.html"))

    def test_wraps_a_bare_run_result(self, tmp_path, only_stub_is_applicable):
        create_report(make_run_result(model_name="ElasticNet"), tmp_path / "run_report")

        assert only_stub_is_applicable.last.computed[0].model_names == ["ElasticNet"]

    def test_normalizes_against_the_reference_model_when_asked(self, experiment, tmp_path, only_stub_is_applicable):
        create_report(experiment, tmp_path / "normalized", reference_model=REFERENCE_MODEL)

        normalized = only_stub_is_applicable.last.computed[0]
        assert normalized.normalized_by == REFERENCE_MODEL
        assert REFERENCE_MODEL not in normalized.model_names

    def test_releases_the_unnormalized_experiment(self, tmp_path, only_stub_is_applicable):
        sentinel: list[weakref.ref] = []

        def built_inline() -> ExperimentResult:
            experiment = make_experiment_result(n_models=2, n_folds=1)
            sentinel.append(weakref.ref(experiment))
            return experiment

        create_report(built_inline(), tmp_path / "released", reference_model=REFERENCE_MODEL)
        gc.collect()

        assert sentinel[0]() is None

    def test_every_real_plot_survives_normalization_and_the_leaderboard_has_values(
        self, normalized_real_report: _BuiltReport
    ) -> None:
        assert [p.name for p in normalized_real_report.out.glob("*.html")] == ["Normalized_multiqc_report.html"]
        leaderboard = next(m for m in normalized_real_report.modules if m.name == "leaderboard")
        assert "No data available" not in leaderboard.sections[0].content
        assert "data:image/png;base64," in leaderboard.sections[0].content


class TestLogging:
    pytestmark = pytest.mark.slow

    def test_logs_counts_before_writing_and_confirms_the_write(self, logged_stub_report: _BuiltReport) -> None:
        messages = [r.getMessage() for r in logged_stub_report.records]
        assert any("3 models (3 model pairs)" in m for m in messages)
        assert any("report: writing 1 modules / 1 sections" in m for m in messages)
        assert messages[-1].startswith("report: written | rss=")

    def test_logs_the_reference_model_before_normalizing(self, experiment, tmp_path, only_stub_is_applicable, caplog):
        with caplog.at_level(logging.INFO, logger="drevalpy.visualization.report"):
            create_report(experiment, tmp_path / "logged_ref", reference_model=REFERENCE_MODEL)

        assert any(f"Normalizing against reference model '{REFERENCE_MODEL}'" in r.getMessage() for r in caplog.records)

    def test_experiment_plot_logs_one_summary_at_info(self, experiment, tmp_path, only_stub_is_applicable, caplog):
        with caplog.at_level(logging.INFO, logger="drevalpy.visualization.report"):
            create_report(experiment, tmp_path / "info")

        messages = [r.getMessage() for r in caplog.records]
        assert sum(m.startswith("plot stub_viz: done in") for m in messages) == 1
        assert not any("computing" in m for m in messages)

    def test_experiment_plot_logs_computing_at_debug(self, experiment, tmp_path, only_stub_is_applicable, caplog):
        with caplog.at_level(logging.DEBUG, logger="drevalpy.visualization.report"):
            create_report(experiment, tmp_path / "debug")

        assert any(r.getMessage().startswith("plot stub_viz: computing | rss=") for r in caplog.records)

    @pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
    def test_model_plot_logs_one_summary_at_info_and_per_model_timing_at_debug(
        self, experiment, tmp_path, monkeypatch, caplog, level
    ):
        _make_applicable(monkeypatch, _StubViz, "ModelResult")
        with caplog.at_level(level, logger="drevalpy.visualization.report"):
            create_report(experiment, tmp_path / "models")

        messages = [r.getMessage() for r in caplog.records]
        per_model = [m for m in messages if m.startswith("plot stub_viz (")]
        if level == logging.INFO:
            assert sum(m.startswith(f"plot stub_viz: {experiment.n_models} models done in") for m in messages) == 1
            assert not per_model
        else:
            for name in experiment.model_names:
                assert any(m.startswith(f"plot stub_viz ({name}): computing | rss=") for m in per_model)
                assert any(
                    m.startswith(f"plot stub_viz ({name}): done in") and re.search(r"rss [+-]\d", m) for m in per_model
                )


class TestFailedRuns:
    pytestmark = pytest.mark.slow

    def test_a_failed_runs_module_lists_every_failure_with_escaped_messages(
        self, experiment, tmp_path, only_stub_is_applicable
    ):
        failures = (
            make_run_failure("ElasticNet", 1, ("SVRC", "gene_expression"), None, "ValueError", "bad <b>input</b>"),
            make_run_failure("Ridge", 0, None, 2, "RuntimeError", "boom"),
        )

        built = _build_report(tmp_path / "failed", ExperimentResult(list(experiment.models[0].runs), failures))

        content = next(m for m in built.modules if m.name == "Failed runs").sections[0].content
        assert content.count("<tr>") == 3
        assert "SVRC (gene_expression)" in content
        assert "RuntimeError" in content
        assert "bad &lt;b&gt;input&lt;/b&gt;" in content
        assert "<b>input</b>" not in content

    def test_no_module_without_failures(self, experiment, tmp_path, only_stub_is_applicable):
        built = _build_report(tmp_path / "clean", experiment)

        assert [m.name for m in built.modules] == ["stub_viz"]


class TestSaveAllPng:
    def test_writes_one_png_per_experiment_level_plot_creating_directories(
        self, experiment, tmp_path, only_stub_is_applicable
    ):
        out = tmp_path / "deep" / "pngs"

        save_all_png(experiment, out)

        assert [p.name for p in out.iterdir()] == ["stub_viz.png"]

    def test_writes_one_png_per_model_for_model_level_plots(self, experiment, tmp_path, monkeypatch):
        _make_applicable(monkeypatch, _StubViz, "ModelResult")

        save_all_png(experiment, tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            f"stub_viz_{name}.png" for name in experiment.model_names
        )

    def test_normalizes_against_the_reference_model_when_asked(self, experiment, tmp_path, only_stub_is_applicable):
        save_all_png(experiment, tmp_path / "norm_pngs", reference_model=REFERENCE_MODEL)

        assert only_stub_is_applicable.last.computed[0].normalized_by == REFERENCE_MODEL

    def test_plotly_plots_are_skipped_with_one_warning_where_png_export_is_unsupported(
        self, experiment, tmp_path, monkeypatch, caplog
    ):
        class _PlotlyStub(_StubViz, PlotlyVisualization):
            registry_name = "plotly_stub"

        _make_applicable(monkeypatch, _PlotlyStub, "ExperimentResult")
        monkeypatch.setattr("drevalpy.visualization.report.plotly_png_supported", lambda: False)

        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.report"):
            save_all_png(experiment, tmp_path)

        assert list(tmp_path.iterdir()) == []
        assert "Skipped the Plotly plots plotly_stub" in caplog.text
