"""Report orchestrator using MultiQC Python API."""

from __future__ import annotations

import html
import logging
import time
from itertools import groupby
from operator import itemgetter
from typing import TYPE_CHECKING

from upath import UPath

from drevalpy._utils._progress import log_stage, rss_gb
from drevalpy._utils.filenames import safe_filename
from drevalpy.log import get_logger
from drevalpy.types.enums import randomization_label
from drevalpy.types.results import ExperimentResult, ModelResult, RunResult
from drevalpy.visualization.base import Plot, Section, is_matplotlib_figure, plotly_png_supported, write_png

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from drevalpy.types.results.failure import RunFailure

logger = get_logger(__name__)


def _add_module(sections: Sequence[Section], name: str, anchor: str) -> None:
    """Create a MultiQC module from sections and append to the report.

    :param sections: Sections the module shows, in order.
    :param name: Module title.
    :param anchor: Module anchor, unique within the report.
    """
    import multiqc

    module = multiqc.BaseMultiqcModule(name=name, anchor=anchor)
    for section in sections:
        module.add_section(
            plot=section.plot,
            content=section.content or "",
            name=section.name,
            anchor=section.anchor,
            description=section.description,
        )
    multiqc.report.modules.append(module)


def _add_failures_module(failures: Sequence[RunFailure]) -> None:
    """Append a "Failed runs" module listing every recorded failure.

    :param failures: Failures recorded on the experiment.
    """
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(cell))}</td>"
            for cell in (
                failure.identity.model_name,
                failure.identity.fold_index,
                randomization_label(failure.identity.randomization) if failure.identity.randomization else "",
                "" if failure.identity.robustness_repeat is None else failure.identity.robustness_repeat,
                failure.error_type,
                failure.message,
            )
        )
        + "</tr>"
        for failure in failures
    )
    header = "".join(
        f"<th>{name}</th>" for name in ("Model", "Fold", "Randomization", "Robustness repeat", "Error", "Message")
    )
    section = Section(
        name="Failed runs",
        anchor="failed_runs",
        description=f"{len(failures)} run(s) raised and have no result.",
        content=f'<table class="table table-condensed"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>',
    )
    _add_module([section], "Failed runs", "failed_runs")


def _ensure_experiment(result: ExperimentResult | ModelResult | RunResult) -> ExperimentResult:
    """Wrap a ModelResult or RunResult into an ExperimentResult.

    :param result: Experiment, model, or run result.
    :returns: *result* itself if it already is an experiment, otherwise an experiment holding its runs.
    """
    if isinstance(result, RunResult):
        return ExperimentResult([result])
    if isinstance(result, ModelResult):
        return ExperimentResult(list(result.runs))
    return result


def _applicable_results(
    experiment: ExperimentResult,
) -> Iterator[tuple[type[Plot], ExperimentResult | ModelResult]]:
    """Pair every applicable plot with each result it consumes.

    :param experiment: Experiment to visualize.
    :yields: ``(plot class, result)``: the whole experiment for an experiment-level plot,
        each model for a plot whose ``result_type`` is ``ModelResult``.
    """
    import drevalpy.visualization.plots  # noqa: F401
    from drevalpy.registry.visualization import applicable

    for plot_cls in applicable(experiment):
        if plot_cls.result_type is ModelResult:
            for model in experiment.models:
                yield plot_cls, model
        else:
            yield plot_cls, experiment


def _add_plot(plot: Plot, target: ExperimentResult | ModelResult) -> None:
    """Draw one plot of one result and append its MultiQC module.

    The per-target timing and memory lines are logged at DEBUG; :func:`create_report` logs the
    INFO summary for the whole plot.

    :param plot: Plot to draw.
    :param target: The experiment, or one model of it.
    """
    if isinstance(target, ModelResult):
        label, anchor = f"{plot.name} ({target.model_name})", f"{plot.name}_{target.model_name}"
    else:
        label = anchor = plot.name
    started = time.monotonic()
    rss_before = rss_gb()
    log_stage(logger, f"plot {label}: computing", level=logging.DEBUG)
    sections = plot.sections(target)
    if sections:
        _add_module(sections, label, anchor)
    logger.debug("plot %s: done in %.1fs, rss %+.2f GB", label, time.monotonic() - started, rss_gb() - rss_before)


def create_report(
    result: ExperimentResult | ModelResult | RunResult,
    output_dir: str | UPath,
    *,
    title: str = "Drug Response Evaluation",
    reference_model: str | None = None,
) -> None:
    """Generate a MultiQC report for the given result.

    :param result: Experiment, model, or run result.
    :param output_dir: Output directory for the report.
    :param title: Report title.
    :param reference_model: If set, normalize metrics against this model.
    """
    import multiqc

    experiment = _ensure_experiment(result)
    n_models = experiment.n_models
    logger.info(
        "Building report %r from %d models (%d model pairs)",
        title,
        n_models,
        n_models * (n_models - 1) // 2,
    )
    if reference_model:
        logger.info("Normalizing against reference model %r", reference_model)
        experiment = experiment.normalize(reference_model)
        # Only the normalized copy is plotted; drop the caller's argument reference so the
        # pre-normalization arrays can be collected instead of being retained in parallel.
        del result
        logger.info("Normalized to %d models", experiment.n_models)
    log_stage(logger, "report: experiment ready")

    multiqc.reset()

    for plot_cls, pairs in groupby(_applicable_results(experiment), key=itemgetter(0)):
        plot_started = time.monotonic()
        plot = plot_cls()
        targets = [target for _, target in pairs]
        for target in targets:
            _add_plot(plot, target)
        n_models = len(targets)
        scope = f"{n_models} model{'s' * (n_models != 1)} " if plot_cls.result_type is ModelResult else ""
        logger.info("plot %s: %sdone in %.1fs", plot.name, scope, time.monotonic() - plot_started)

    if experiment.failures:
        _add_failures_module(experiment.failures)

    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_stage(
        logger,
        f"report: writing {len(multiqc.report.modules)} modules / "
        f"{sum(len(m.sections) for m in multiqc.report.modules)} sections to {out}",
    )
    multiqc.write_report(output_dir=str(out), title=title, force=True)
    log_stage(logger, "report: written")


def save_all_png(
    result: ExperimentResult | ModelResult | RunResult,
    output_dir: str | UPath,
    *,
    reference_model: str | None = None,
) -> None:
    """Save all applicable plots as PNG files.

    On Windows, Plotly-based plots are skipped with a warning; see
    :func:`~drevalpy.visualization.base.plotly_png_supported`.

    :param result: Experiment, model, or run result.
    :param output_dir: Output directory for the PNG files.
    :param reference_model: If set, normalize metrics against this model.
    """
    experiment = _ensure_experiment(result)
    if reference_model:
        experiment = experiment.normalize(reference_model)
        # As in create_report: only the normalized copy is plotted from here on.
        del result

    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    for plot_cls, target in _applicable_results(experiment):
        plot = plot_cls()
        figure = plot.figure(target)
        if not is_matplotlib_figure(figure) and not plotly_png_supported():
            skipped.append(plot.name)
            continue
        stem = f"{plot.name}_{safe_filename(target.model_name)}" if isinstance(target, ModelResult) else plot.name
        write_png(figure, out / f"{stem}.png")
    if skipped:
        logger.warning(
            "Skipped the Plotly plots %s: PNG export hangs on Windows with Kaleido 0.2.1.",
            ", ".join(sorted(set(skipped))),
        )
