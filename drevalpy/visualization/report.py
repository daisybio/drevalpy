"""Report orchestrator using MultiQC Python API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from upath import UPath as Path

from drevalpy.visualization.base import Visualization

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult, ModelResult, RunResult


def _add_module(sections, name: str, anchor: str) -> None:
    """Create a MultiQC module from sections and append to the report."""
    import multiqc

    module = multiqc.BaseMultiqcModule(name=name, anchor=anchor)
    for section in sections:
        module.add_section(
            plot=section.plot,
            content=section.content,
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


def _run_visualization(viz: Visualization, experiment, result_type: str) -> None:
    """Compute a visualization and add its sections to the report."""
    if result_type == "ModelResult":
        for model in experiment.models:
            viz.compute(model)
            sections = viz.to_multiqc()
            if sections:
                name = f"{viz.registry_name} ({model.model_name})"
                anchor = f"{viz.registry_name}_{model.model_name}"
                _add_module(sections, name, anchor)
    else:
        viz.compute(experiment)
        sections = viz.to_multiqc()
        if sections:
            _add_module(sections, viz.registry_name, viz.registry_name)


def create_report(
    result: ExperimentResult | ModelResult | RunResult,
    output_dir: str | Path,
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
    try:
        import multiqc
    except ImportError as e:
        raise ImportError(
            "multiqc is required for report generation. Install it with: pip install drevalpy[report]"
        ) from e

    import drevalpy.visualization.plots  # noqa: F401
    from drevalpy.visualization.registry import visualization_registry

    experiment = _ensure_experiment(result)
    if reference_model:
        experiment = experiment.normalize(reference_model)

    multiqc.reset()

    for viz_cls in visualization_registry.applicable(experiment):
        result_type = visualization_registry._result_types.get(viz_cls.registry_name, "ExperimentResult")
        viz = viz_cls()
        _run_visualization(viz, experiment, result_type)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    multiqc.write_report(output_dir=str(out), title=title, force=True)


def save_all_png(
    result: ExperimentResult | ModelResult | RunResult,
    output_dir: str | Path,
    *,
    reference_model: str | None = None,
) -> None:
    """Save all applicable plots as PNG files.

    :param result: Experiment, model, or run result.
    :param output_dir: Output directory for the PNG files.
    :param reference_model: If set, normalize metrics against this model.
    """
    import drevalpy.visualization.plots  # noqa: F401
    from drevalpy.visualization.registry import visualization_registry

    experiment = _ensure_experiment(result)
    if reference_model:
        experiment = experiment.normalize(reference_model)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for viz_cls in visualization_registry.applicable(experiment):
        result_type = visualization_registry._result_types.get(viz_cls.registry_name, "ExperimentResult")
        viz = viz_cls()
        if result_type == "ModelResult":
            for model in experiment.models:
                viz.compute(model)
                viz.to_png(out / f"{viz.registry_name}_{model.model_name}.png")
        else:
            viz.compute(experiment)
            viz.to_png(out / f"{viz.registry_name}.png")
