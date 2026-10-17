"""Report orchestrator using MultiQC Python API."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from upath import UPath as Path

from drevalpy.log import get_logger
from drevalpy.visualization._progress import log_stage, rss_gb
from drevalpy.visualization.base import Visualization

if TYPE_CHECKING:
    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.results import ExperimentResult, ModelResult, RunResult

logger = get_logger(__name__)

#: Log every Nth model in a per-model plot loop, plus always the first and last.
_MODEL_LOG_EVERY = 10


def _add_module(sections, name: str, anchor: str) -> None:
    """Create a MultiQC module from sections and append to the report."""
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


def _ensure_experiment(result):
    """Wrap a ModelResult or RunResult into an ExperimentResult."""
    from drevalpy.types.results import ExperimentResult, ModelResult, RunResult

    if isinstance(result, RunResult):
        return ExperimentResult([result])
    if isinstance(result, ModelResult):
        return ExperimentResult(list(result.runs))
    return result


def _run_visualization(viz: Visualization, experiment, result_type: str, dataset=None) -> None:
    """Compute a visualization and add its sections to the report."""
    started = time.monotonic()
    rss_before = rss_gb()
    if result_type == "ModelResult":
        models = experiment.models
        log_stage(logger, f"plot {viz.registry_name}: computing for {len(models)} models")
        for i, model in enumerate(models, start=1):
            if i == 1 or i == len(models) or i % _MODEL_LOG_EVERY == 0:
                logger.info("  %s: model %d/%d (%s)", viz.registry_name, i, len(models), model.model_name)
            viz.compute(model, dataset=dataset)
            sections = viz.to_multiqc()
            if sections:
                name = f"{viz.registry_name} ({model.model_name})"
                anchor = f"{viz.registry_name}_{model.model_name}"
                _add_module(sections, name, anchor)
    else:
        log_stage(logger, f"plot {viz.registry_name}: computing")
        viz.compute(experiment, dataset=dataset)
        sections = viz.to_multiqc()
        if sections:
            _add_module(sections, viz.registry_name, viz.registry_name)
    logger.info(
        "plot %s: done in %.1fs, rss %+.2f GB",
        viz.registry_name,
        time.monotonic() - started,
        rss_gb() - rss_before,
    )


def create_report(
    result: ExperimentResult | ModelResult | RunResult,
    output_dir: str | Path,
    *,
    title: str = "Drug Response Evaluation",
    reference_model: str | None = None,
    dataset: Dataset | None = None,
) -> None:
    """Generate a MultiQC report for the given result.

    :param result: Experiment, model, or run result.
    :param output_dir: Output directory for the report.
    :param title: Report title.
    :param reference_model: If set, normalize metrics against this model.
    :param dataset: Optional dataset for drug/cell-line metadata in plots.
    """
    try:
        import multiqc
    except ImportError as e:
        raise ImportError(
            "multiqc is required for report generation. Install it with: pip install drevalpy[report]"
        ) from e

    import drevalpy.visualization.plots  # noqa: F401
    from drevalpy.registry.visualization import visualization_registry

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

    for viz_cls in visualization_registry.applicable(experiment):
        result_type = visualization_registry._result_types.get(viz_cls.registry_name, "ExperimentResult")
        viz = viz_cls()
        _run_visualization(viz, experiment, result_type, dataset=dataset)

    out = Path(output_dir)
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
    output_dir: str | Path,
    *,
    reference_model: str | None = None,
    dataset: Dataset | None = None,
) -> None:
    """Save all applicable plots as PNG files.

    :param result: Experiment, model, or run result.
    :param output_dir: Output directory for the PNG files.
    :param reference_model: If set, normalize metrics against this model.
    :param dataset: Optional dataset for drug/cell-line metadata in plots.
    """
    import drevalpy.visualization.plots  # noqa: F401
    from drevalpy.registry.visualization import visualization_registry

    experiment = _ensure_experiment(result)
    if reference_model:
        experiment = experiment.normalize(reference_model)
        # As in create_report: only the normalized copy is plotted from here on.
        del result

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for viz_cls in visualization_registry.applicable(experiment):
        result_type = visualization_registry._result_types.get(viz_cls.registry_name, "ExperimentResult")
        viz = viz_cls()
        if result_type == "ModelResult":
            for model in experiment.models:
                viz.compute(model, dataset=dataset)
                viz.to_png(out / f"{viz.registry_name}_{model.model_name}.png")
        else:
            viz.compute(experiment, dataset=dataset)
            viz.to_png(out / f"{viz.registry_name}.png")
