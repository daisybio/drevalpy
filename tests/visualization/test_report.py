"""Tests for :mod:`drevalpy.visualization.report`.

One end-to-end ``create_report`` run is enough to cover the MultiQC wiring; the
remaining orchestration branches are driven with a stub visualization so the
suite does not pay for seven real plots per assertion.
"""

from __future__ import annotations

import gc
import logging
import weakref
from typing import Any

import multiqc
import pytest
from matplotlib import pyplot as plt
from upath import UPath

from drevalpy.registry.visualization import visualization_registry
from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.report import (
    _add_module,
    _ensure_experiment,
    _run_visualization,
    create_report,
    save_all_png,
)
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_model_result, make_run_result


class _StubViz(Visualization):
    """Records what it was asked to compute and emits one trivial section."""

    registry_name = "stub_viz"
    sections: list[Section] = [Section(name="Stub", anchor="stub_section", content="<p>stub</p>")]
    #: The most recently constructed instance, so tests can inspect what the
    #: report built internally.
    last: Any = None

    def __init__(self) -> None:
        self.computed: list[tuple[Any, Any]] = []
        type(self).last = self

    def compute(self, result, dataset=None) -> None:
        self.computed.append((result, dataset))

    def to_png(self, path) -> None:
        UPath(path).write_bytes(b"stub-png")

    def to_multiqc(self) -> list[Section]:
        return list(self.sections)

    def show(self) -> None:  # pragma: no cover - not part of the report path
        raise AssertionError("show() is not used by the report")


class _SilentViz(_StubViz):
    """A visualization that has nothing to contribute."""

    registry_name = "silent_viz"
    sections: list[Section] = []


@pytest.fixture(autouse=True)
def _reset_multiqc():
    """Isolate MultiQC report state and contain the leaderboard's rcParams edits."""
    multiqc.reset()
    with plt.rc_context():
        yield
    plt.close("all")
    multiqc.reset()


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result()


@pytest.fixture
def only_stub_is_applicable(monkeypatch) -> type[_StubViz]:
    """Restrict the report to a single cheap visualization."""
    monkeypatch.setattr(visualization_registry, "applicable", lambda experiment: [_StubViz])
    return _StubViz


class TestEnsureExperiment:
    def test_wraps_a_single_run(self):
        run = make_run_result(model_name="ElasticNet")

        experiment = _ensure_experiment(run)

        assert isinstance(experiment, ExperimentResult)
        assert experiment.model_names == ["ElasticNet"]

    def test_wraps_a_model_result_keeping_every_fold(self):
        model = make_model_result(n_folds=3)

        experiment = _ensure_experiment(model)

        assert isinstance(experiment, ExperimentResult)
        assert experiment.max_folds == 3

    def test_passes_an_experiment_through_unchanged(self, experiment):
        assert _ensure_experiment(experiment) is experiment


class TestAddModule:
    def test_appends_one_module_carrying_the_sections(self):
        _add_module([Section(name="S", anchor="s", content="x")], "My module", "my_module")

        assert [m.name for m in multiqc.report.modules] == ["My module"]
        assert [s.name for s in multiqc.report.modules[0].sections] == ["S"]

    def test_sections_without_content_become_empty_strings(self):
        _add_module([Section(name="S", anchor="s")], "My module", "my_module")

        assert multiqc.report.modules[0].sections[0].content == ""


class TestRunVisualization:
    def test_experiment_plots_are_computed_once(self, experiment):
        viz = _StubViz()

        _run_visualization(viz, experiment, "ExperimentResult")

        assert [result for result, _ in viz.computed] == [experiment]

    def test_experiment_plots_add_one_module_named_after_the_registry(self, experiment):
        _run_visualization(_StubViz(), experiment, "ExperimentResult")

        assert [m.name for m in multiqc.report.modules] == ["stub_viz"]

    def test_model_plots_are_computed_once_per_model(self, experiment):
        viz = _StubViz()

        _run_visualization(viz, experiment, "ModelResult")

        assert [result.model_name for result, _ in viz.computed] == experiment.model_names

    def test_model_plots_add_one_module_per_model(self, experiment):
        _run_visualization(_StubViz(), experiment, "ModelResult")

        assert [m.name for m in multiqc.report.modules] == [f"stub_viz ({name})" for name in experiment.model_names]

    def test_the_dataset_is_forwarded_to_compute(self, experiment):
        dataset = object()
        viz = _StubViz()

        _run_visualization(viz, experiment, "ExperimentResult", dataset=dataset)

        assert viz.computed[0][1] is dataset

    def test_visualizations_without_sections_add_no_module(self, experiment):
        _run_visualization(_SilentViz(), experiment, "ExperimentResult")

        assert multiqc.report.modules == []

    def test_silent_model_plots_add_no_module(self, experiment):
        _run_visualization(_SilentViz(), experiment, "ModelResult")

        assert multiqc.report.modules == []


class TestCreateReport:
    def test_writes_a_multiqc_html_report_for_every_applicable_plot(self, experiment, tmp_path):
        out = tmp_path / "report"

        create_report(experiment, out, title="W4 Report")

        assert [p.name for p in out.glob("*.html")] == ["W4-Report_multiqc_report.html"]

    def test_creates_missing_output_directories(self, experiment, tmp_path, only_stub_is_applicable):
        out = tmp_path / "deep" / "nested"

        create_report(experiment, out)

        assert out.is_dir()

    def test_accepts_a_string_output_directory(self, experiment, tmp_path, only_stub_is_applicable):
        out = tmp_path / "as_string"

        create_report(experiment, str(out))

        assert any(out.glob("*.html"))

    def test_wraps_a_bare_run_result(self, tmp_path, only_stub_is_applicable):
        run = make_run_result(model_name="ElasticNet")

        create_report(run, tmp_path / "run_report")

        assert only_stub_is_applicable.last.computed[0][0].model_names == ["ElasticNet"]

    def test_normalizes_against_the_reference_model_when_asked(self, experiment, tmp_path, only_stub_is_applicable):
        create_report(experiment, tmp_path / "normalized", reference_model=REFERENCE_MODEL)

        normalized = only_stub_is_applicable.last.computed[0][0]
        assert normalized.normalized_by == REFERENCE_MODEL
        assert REFERENCE_MODEL not in normalized.model_names

    def test_forwards_the_dataset_to_the_plots(self, experiment, tmp_path, only_stub_is_applicable):
        dataset = object()

        create_report(experiment, tmp_path / "with_dataset", dataset=dataset)

        assert only_stub_is_applicable.last.computed[0][1] is dataset

    def test_releases_the_unnormalized_experiment(self, tmp_path, only_stub_is_applicable):
        """The pre-normalization copy must not stay reachable through the argument."""
        sentinel: list[weakref.ref] = []

        def built_inline() -> ExperimentResult:
            experiment = make_experiment_result(n_models=2, n_folds=1)
            sentinel.append(weakref.ref(experiment))
            return experiment

        create_report(built_inline(), tmp_path / "released", reference_model=REFERENCE_MODEL)
        gc.collect()

        assert sentinel[0]() is None

    def test_every_real_plot_survives_normalization(self, tmp_path):
        """The pipeline always passes ``--reference-model``, and this path used to crash.

        ``normalize()`` recomputes metrics under their plain names, which the
        leaderboard did not read; its PCC column came out all-NaN and matplotlib
        raised ``Axis limits cannot be NaN or Inf`` before MultiQC ever ran.
        """
        out = tmp_path / "normalized_report"

        create_report(
            make_experiment_result(n_models=4, n_folds=3),
            out,
            title="Normalized",
            reference_model=REFERENCE_MODEL,
        )

        assert [p.name for p in out.glob("*.html")] == ["Normalized_multiqc_report.html"]

    def test_the_normalized_leaderboard_renders_with_real_values(self, tmp_path):
        """A section is worthless if it is present but blank."""
        create_report(
            make_experiment_result(n_models=4, n_folds=3),
            tmp_path / "leaderboard_values",
            reference_model=REFERENCE_MODEL,
        )

        leaderboard = next(m for m in multiqc.report.modules if m.name == "leaderboard")
        assert "No data available" not in leaderboard.sections[0].content
        assert "data:image/png;base64," in leaderboard.sections[0].content


class TestLogging:
    """The report used to die with an empty ``Command output``; these lines are the fix."""

    def test_logs_the_model_count_and_the_pair_count(self, experiment, tmp_path, only_stub_is_applicable, caplog):
        with caplog.at_level(logging.INFO, logger="drevalpy.visualization.report"):
            create_report(experiment, tmp_path / "logged")

        messages = [r.getMessage() for r in caplog.records]
        assert any("3 models (3 model pairs)" in m for m in messages)

    def test_logs_the_reference_model_before_normalizing(self, experiment, tmp_path, only_stub_is_applicable, caplog):
        with caplog.at_level(logging.INFO, logger="drevalpy.visualization.report"):
            create_report(experiment, tmp_path / "logged_ref", reference_model=REFERENCE_MODEL)

        messages = [r.getMessage() for r in caplog.records]
        assert any(f"Normalizing against reference model '{REFERENCE_MODEL}'" in m for m in messages)

    def test_names_each_plot_before_computing_it(self, experiment, only_stub_is_applicable, caplog):
        with caplog.at_level(logging.INFO, logger="drevalpy.visualization.report"):
            _run_visualization(_StubViz(), experiment, "ExperimentResult")

        assert caplog.records[0].getMessage().startswith("plot stub_viz: computing | rss=")

    def test_reports_per_model_progress(self, experiment, caplog):
        with caplog.at_level(logging.INFO, logger="drevalpy.visualization.report"):
            _run_visualization(_StubViz(), experiment, "ModelResult")

        messages = [r.getMessage() for r in caplog.records]
        assert any("model 1/3" in m for m in messages)
        assert any("model 3/3" in m for m in messages)

    def test_reports_elapsed_time_and_memory_delta_after_each_plot(self, experiment, caplog):
        with caplog.at_level(logging.INFO, logger="drevalpy.visualization.report"):
            _run_visualization(_StubViz(), experiment, "ExperimentResult")

        assert "plot stub_viz: done in" in caplog.records[-1].getMessage()

    def test_logs_the_module_and_section_counts_before_writing(
        self, experiment, tmp_path, only_stub_is_applicable, caplog
    ):
        with caplog.at_level(logging.INFO, logger="drevalpy.visualization.report"):
            create_report(experiment, tmp_path / "counted")

        messages = [r.getMessage() for r in caplog.records]
        assert any("report: writing 1 modules / 1 sections" in m for m in messages)

    def test_confirms_the_report_was_written(self, experiment, tmp_path, only_stub_is_applicable):
        # MultiQC re-initialises root logging inside write_report, so caplog's root handler
        # misses anything logged afterwards; listen on the module logger directly.
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger = logging.getLogger("drevalpy.visualization.report")
        logger.addHandler(handler)
        try:
            create_report(experiment, tmp_path / "confirmed")
        finally:
            logger.removeHandler(handler)

        assert records[-1].getMessage().startswith("report: written | rss=")


class TestSaveAllPng:
    def test_writes_one_png_per_experiment_level_plot(self, experiment, tmp_path, only_stub_is_applicable):
        out = tmp_path / "pngs"

        save_all_png(experiment, out)

        assert [p.name for p in out.iterdir()] == ["stub_viz.png"]

    def test_writes_one_png_per_model_for_model_level_plots(self, experiment, tmp_path, monkeypatch):
        monkeypatch.setattr(visualization_registry, "applicable", lambda exp: [_StubViz])
        monkeypatch.setitem(visualization_registry._result_types, "stub_viz", "ModelResult")
        out = tmp_path / "per_model"

        save_all_png(experiment, out)

        assert sorted(p.name for p in out.iterdir()) == sorted(
            f"stub_viz_{name}.png" for name in experiment.model_names
        )

    def test_creates_missing_output_directories(self, experiment, tmp_path, only_stub_is_applicable):
        out = tmp_path / "deep" / "pngs"

        save_all_png(experiment, out)

        assert out.is_dir()

    def test_normalizes_against_the_reference_model_when_asked(self, experiment, tmp_path, only_stub_is_applicable):
        save_all_png(experiment, tmp_path / "norm_pngs", reference_model=REFERENCE_MODEL)

        assert only_stub_is_applicable.last.computed[0][0].normalized_by == REFERENCE_MODEL

    def test_forwards_the_dataset_to_the_plots(self, experiment, tmp_path, only_stub_is_applicable):
        dataset = object()

        save_all_png(experiment, tmp_path / "ds_pngs", dataset=dataset)

        assert only_stub_is_applicable.last.computed[0][1] is dataset
