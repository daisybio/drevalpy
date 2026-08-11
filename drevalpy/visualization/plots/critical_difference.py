"""Critical difference diagram visualization (MultiQC)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.registry import visualization_registry
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


@visualization_registry.register(
    "critical_difference",
    "Critical difference diagram with Friedman test and model rankings",
    requirements=frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS}),
)
class CriticalDifferenceVisualization(Visualization):
    """Critical difference rank table and optional diagram."""

    def generate(self, result: ExperimentResult) -> list[Section]:
        """Compute rankings and Friedman test, present as a table.

        :param result: Experiment result with multiple models and folds.
        :returns: List of Sections (rank table + optional SVG).
        """
        try:
            from multiqc.plots import table as mqc_table
        except ImportError as e:
            raise ImportError(
                "multiqc is required for critical difference plots. Install with: pip install drevalpy[report]"
            ) from e

        metric_names = sorted({m for model in result.models for m in model.aggregate_metrics})
        sections: list[Section] = []

        for metric in metric_names:
            section = _build_metric_section(result, metric, mqc_table)
            if section is not None:
                sections.append(section)

        return sections


def _build_metric_section(result: ExperimentResult, metric: str, mqc_table: Any) -> Section | None:
    """Build a single CD section for one metric."""
    import numpy as np

    fold_scores = _collect_fold_scores(result, metric)
    if not fold_scores or len(fold_scores) < 2:
        return None

    model_names = [m.model_name for m in result.models]
    valid_models = [m for m in model_names if m in fold_scores]
    n_folds = min(len(v) for v in fold_scores.values())
    score_matrix = np.array([fold_scores[m][:n_folds] for m in valid_models])

    mean_ranks = _compute_mean_ranks(score_matrix)
    stat, p_value = _friedman_test(score_matrix)

    table_data = _build_table_data(valid_models, mean_ranks, score_matrix)
    headers = _build_headers(metric)

    plot = mqc_table.plot(
        table_data,
        headers,
        pconfig={"id": f"dreval_cd_{metric}", "title": f"Rankings: {metric}"},
    )

    desc = f"Model rankings for {metric}. "
    if not np.isnan(p_value):
        desc += f"Friedman \u03c7\u00b2={stat:.2f}, p={p_value:.4f}."

    return Section(
        name=f"Critical Difference: {metric}",
        anchor=f"dreval_cd_{metric}",
        description=desc,
        plot=plot,
    )


def _compute_mean_ranks(score_matrix):
    """Rank models per fold and return mean ranks."""
    import numpy as np
    from scipy import stats

    ranks = np.zeros_like(score_matrix)
    for col in range(score_matrix.shape[1]):
        ranks[:, col] = stats.rankdata(-score_matrix[:, col])
    return ranks.mean(axis=1)


def _friedman_test(score_matrix):
    """Run Friedman test on score matrix."""
    from scipy import stats

    try:
        stat, p_value = stats.friedmanchisquare(*score_matrix.T)
    except ValueError:
        stat, p_value = float("nan"), float("nan")
    return stat, p_value


def _build_table_data(valid_models: list[str], mean_ranks, score_matrix) -> dict[str, dict[str, float]]:
    """Build MultiQC table data dict."""
    table_data: dict[str, dict[str, float]] = {}
    for i, model in enumerate(valid_models):
        table_data[model] = {
            "mean_rank": float(mean_ranks[i]),
            "mean_score": float(score_matrix[i].mean()),
        }
    return table_data


def _build_headers(metric: str) -> dict[str, dict[str, str | float]]:
    """Build MultiQC table headers."""
    return {
        "mean_rank": {
            "title": "Mean Rank",
            "description": "Average rank across folds (lower is better)",
            "min": 1.0,
        },
        "mean_score": {
            "title": f"Mean {metric}",
            "description": f"Mean {metric} across folds",
        },
    }


def _collect_fold_scores(result: ExperimentResult, metric: str) -> dict[str, list[float]]:
    """Gather per-fold metric values keyed by model name."""
    fold_scores: dict[str, list[float]] = {}
    for model in result.models:
        scores = [r.metrics[metric] for r in model.runs if metric in r.metrics]
        if scores:
            fold_scores[model.model_name] = scores
    return fold_scores
