"""Comparison scatter plot: per-group correlation of one model against another.

One point per drug (or per cell line), not per prediction. Both axes carry the
same quantity for two different models, so the identity line reads directly as
"these two models are equally good on this group"; systematic deviation means one
model dominates. Model selection is a pair of Plotly dropdowns, so the payload is
one correlation vector per model rather than one point cloud per model pair.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from drevalpy.registry.visualization import register
from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.plots._group_metrics import (
    GROUPING_LABELS,
    GROUPINGS,
    GroupCorrelationMatrix,
    model_group_correlations,
)
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    from pathlib import Path

    from drevalpy.types.results import ExperimentResult

#: Correlations live in [-1, 1]; fixing the range keeps the identity line at 45
#: degrees when a dropdown swaps in a model with a narrower spread.
_AXIS_RANGE = (-1.05, 1.05)

#: Index of the trace the dropdowns restyle. Trace 1 is the reference line and
#: must be left alone.
_POINTS_TRACE = 0


def _axis_title(grouping: str, model_name: str) -> str:
    return f"{model_name} (per-{GROUPING_LABELS[grouping]} Pearson)"


def _axis_layout(grouping: str, model_name: str) -> dict:
    """Return an axis layout dict.

    Plotly 3 silently discards a bare string under ``title`` in a relayout
    payload, leaving the axis unlabelled, so the nested form is used throughout.

    :param grouping: One of the supported groupings.
    :param model_name: Model shown on this axis.
    :returns: A layout fragment carrying the title and the fixed range.
    """
    return {"title": {"text": _axis_title(grouping, model_name)}, "range": list(_AXIS_RANGE)}


def _dropdown_buttons(matrix: GroupCorrelationMatrix, axis: str) -> list[dict]:
    """Build the update buttons for one axis of one grouping.

    Each button carries a single model's correlation vector, so the whole figure
    holds ``n_models x n_groups`` numbers twice over rather than anything
    quadratic in the number of models.

    The trace index is passed explicitly: without it Plotly applies the restyle
    cyclically across every trace and the dashed reference line is overwritten
    with the model's data.

    :param matrix: Correlations to build buttons from.
    :param axis: ``"x"`` or ``"y"``.
    :returns: One Plotly ``updatemenu`` button per model.
    """
    key = "xaxis" if axis == "x" else "yaxis"
    buttons = []
    for index, model_name in enumerate(matrix.model_names):
        values = np.nan_to_num(matrix.values[index], nan=0.0).astype(float).tolist()
        buttons.append(
            {
                "label": model_name,
                "method": "update",
                "args": [
                    {axis: [values]},
                    {key: _axis_layout(matrix.grouping, model_name)},
                    [_POINTS_TRACE],
                ],
            }
        )
    return buttons


def _build_figure(matrix: GroupCorrelationMatrix) -> go.Figure:
    """Build the two-dropdown comparison figure for one grouping.

    :param matrix: Correlations to plot. May be empty, in which case an empty
        figure is returned.
    :returns: A Plotly figure with one scatter trace and two ``updatemenus``.
    """
    fig = go.Figure()
    if matrix.is_empty:
        return fig

    first = np.nan_to_num(matrix.values[0], nan=0.0).astype(float)
    fig.add_trace(
        go.Scatter(
            x=first,
            y=first,
            mode="markers",
            marker={"size": 6, "opacity": 0.7},
            customdata=list(matrix.group_names),
            hovertemplate=(
                f"{GROUPING_LABELS[matrix.grouping].capitalize()}: %{{customdata}}<br>"
                "x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(_AXIS_RANGE),
            y=list(_AXIS_RANGE),
            mode="lines",
            line={"dash": "dash", "width": 1, "color": "#888888"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    label = GROUPING_LABELS[matrix.grouping]
    first_model = matrix.model_names[0]
    fig.update_layout(
        title={
            "text": f"Per-{label} Pearson correlation, model against model",
            "y": 0.99,
            "yanchor": "top",
        },
        showlegend=False,
        xaxis=_axis_layout(matrix.grouping, first_model),
        yaxis=_axis_layout(matrix.grouping, first_model),
        annotations=[
            {
                "text": "x-axis model:",
                "showarrow": False,
                "x": 0,
                "xref": "paper",
                "xanchor": "left",
                "y": 1.12,
                "yref": "paper",
                "yanchor": "bottom",
            },
            {
                "text": "y-axis model:",
                "showarrow": False,
                "x": 0.5,
                "xref": "paper",
                "xanchor": "left",
                "y": 1.12,
                "yref": "paper",
                "yanchor": "bottom",
            },
        ],
        updatemenus=[
            {
                "buttons": _dropdown_buttons(matrix, "x"),
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
            },
            {
                "buttons": _dropdown_buttons(matrix, "y"),
                "direction": "down",
                "showactive": True,
                "x": 0.5,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
            },
        ],
        margin={"t": 130},
        plot_bgcolor="#e5ecf6",
        xaxis_gridcolor="white",
        yaxis_gridcolor="white",
    )
    return fig


def _inline_plotly_html(fig: go.Figure, div_id: str) -> str:
    """Render a figure as HTML against MultiQC's already-bundled Plotly.

    MultiQC's default template includes ``plotly-3.1.2.custom.min.js`` in the
    document head and that bundle assigns ``window.Plotly``, so the report needs
    no second copy of the library - only the data and a ``newPlot`` call. The
    Plotly default styling template is stripped from the layout for the same
    reason: it is ~18 kB of defaults the browser already has.

    :param fig: Figure to render.
    :param div_id: DOM id for the container div; must be unique in the report.
    :returns: A self-contained HTML fragment.
    """
    spec = fig.to_plotly_json()
    layout = {key: value for key, value in spec["layout"].items() if key != "template"}
    payload = json.dumps({"data": spec["data"], "layout": layout}, cls=PlotlyJSONEncoder)
    return (
        f'<div id="{div_id}" style="width:100%;height:640px"></div>\n'
        "<script>(function(){\n"
        f"  var spec = {payload};\n"
        f'  var target = document.getElementById("{div_id}");\n'
        "  function draw(){ Plotly.newPlot(target, spec.data, spec.layout, "
        "{responsive: true, displayModeBar: true}); }\n"
        '  if (typeof Plotly === "undefined") {\n'
        '    document.addEventListener("DOMContentLoaded", draw);\n'
        "  } else { draw(); }\n"
        "})();</script>"
    )


@register(
    "comparison_scatter",
    "Per-drug and per-cell-line correlation compared between two selectable models",
    requirements=frozenset({PlotRequirement.MULTIPLE_MODELS}),
)
class ComparisonScatterVisualization(Visualization):
    """Model-against-model comparison of per-group correlation (Plotly)."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._fig: go.Figure | None = None
        self._matrices: dict[str, GroupCorrelationMatrix] = {}

    def compute(self, result: ExperimentResult, dataset=None) -> None:
        """Compute per-group correlations for every model and build the figure.

        Retains one ``float32`` matrix of shape ``n_models x n_groups`` per
        grouping, independent of the number of predictions.

        :param result: Experiment result with at least two models.
        :param dataset: Unused; accepted for interface compatibility.
        """
        self._matrices = {}
        for grouping in GROUPINGS:
            matrix = model_group_correlations(result, grouping).drop_all_nan_models()
            if matrix.n_models >= 2 and matrix.n_groups > 0:
                self._matrices[grouping] = matrix
        self._fig = _build_figure(self._primary_matrix()) if self._matrices else go.Figure()

    def _primary_matrix(self) -> GroupCorrelationMatrix:
        """Return the matrix backing ``_fig``, i.e. the first available grouping."""
        return self._matrices[next(iter(self._matrices))]

    def to_png(self, path: str | Path) -> None:
        """Render the primary grouping's figure to a static PNG.

        :param path: Output file path.
        :raises RuntimeError: If called before ``compute()``.
        """
        if self._fig is None:
            raise RuntimeError("Call compute() before to_png()")
        self._fig.write_image(str(path))

    def to_multiqc(self) -> list[Section]:
        """Return one Section per grouping, each holding a two-dropdown figure.

        :returns: Sections for the drug and cell-line groupings, in that order.
            Empty when fewer than two models have defined correlations.
        :raises RuntimeError: If called before ``compute()``.
        """
        if self._fig is None:
            raise RuntimeError("Call compute() before to_multiqc()")

        sections: list[Section] = []
        for grouping, matrix in self._matrices.items():
            label = GROUPING_LABELS[grouping]
            anchor = f"dreval_comp_scatter_{grouping}"
            figure = self._fig if matrix is self._primary_matrix() else _build_figure(matrix)
            sections.append(
                Section(
                    name=f"{label.capitalize()}-wise comparison",
                    anchor=anchor,
                    description=(
                        f"Pearson correlation of predictions against ground truth within each {label}, "
                        f"for {matrix.n_models} models over {matrix.n_groups} {label}s. "
                        "Pick the model on each axis with the dropdowns; points above the diagonal "
                        "favour the model on the y-axis."
                    ),
                    content=_inline_plotly_html(figure, f"{anchor}_div"),
                )
            )
        return sections

    def show(self) -> None:
        """Display the comparison scatter in a Jupyter notebook.

        :raises RuntimeError: If called before ``compute()``.
        """
        if self._fig is None:
            raise RuntimeError("Call compute() before show()")
        self._fig.show()
