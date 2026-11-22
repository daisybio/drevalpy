"""Comparison scatter plot: per-group correlation of one model against another.

One point per drug (or per sample or study), not per prediction. Both axes carry the
same quantity for two different models, so the identity line reads directly as
"these two models are equally good on this group"; systematic deviation means one
model dominates. Model selection is a pair of Plotly dropdowns, so the payload is
one correlation vector per model rather than one point cloud per model pair.

A model that predicts a constant within a group scores a correlation of 0 there and is drawn;
a group whose response is constant is undefined for every model and is drawn at 0 on both axes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from drevalpy.evaluation._correlation_matrix import model_group_correlations
from drevalpy.evaluation._grouping import GROUPINGS
from drevalpy.registry.visualization import register
from drevalpy.visualization.base import Plot, Section
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

    from drevalpy.types.results import ExperimentResult

#: Correlations live in [-1, 1]; fixing the range keeps the identity line at 45
#: degrees when a dropdown swaps in a model with a narrower spread.
_AXIS_RANGE = (-1.05, 1.05)

#: Index of the trace the dropdowns restyle. Trace 1 is the reference line and
#: must be left alone.
_POINTS_TRACE = 0


#: The two dropdowns: the axis each one drives and its horizontal position in paper coordinates.
_DROPDOWNS = (("x", 0.0), ("y", 0.5))


def _correlations(row: pd.Series) -> np.ndarray:
    """One model's correlations as plotted: ``float`` with the undefined ones drawn at 0."""
    return row.fillna(0.0).to_numpy(dtype=float)


def _axis_layout(grouping: str, model_name: str) -> dict:
    """Return an axis layout dict.

    Plotly 3 silently discards a bare string under ``title`` in a relayout
    payload, leaving the axis unlabelled, so the nested form is used throughout.

    :param grouping: One of the supported groupings.
    :param model_name: Model shown on this axis.
    :returns: A layout fragment carrying the title and the fixed range.
    """
    return {"title": {"text": f"{model_name} (per-{grouping} Pearson)"}, "range": list(_AXIS_RANGE)}


def _dropdown_buttons(grouping: str, matrix: pd.DataFrame, axis: str) -> list[dict]:
    """Build the update buttons for one axis of one grouping.

    Each button carries a single model's correlation vector, so the whole figure
    holds ``n_models x n_groups`` numbers twice over rather than anything
    quadratic in the number of models.

    The trace index is passed explicitly: without it Plotly applies the restyle
    cyclically across every trace and the dashed reference line is overwritten
    with the model's data.

    :param grouping: The grouping the correlations are for.
    :param matrix: Correlations to build buttons from, one row per model.
    :param axis: ``"x"`` or ``"y"``.
    :returns: One Plotly ``updatemenu`` button per model.
    """
    return [
        {
            "label": model_name,
            "method": "update",
            "args": [
                {axis: [_correlations(row).tolist()]},
                {f"{axis}axis": _axis_layout(grouping, model_name)},
                [_POINTS_TRACE],
            ],
        }
        for model_name, row in matrix.iterrows()
    ]


def _build_figure(grouping: str, matrix: pd.DataFrame) -> go.Figure:
    """Build the two-dropdown comparison figure for one grouping.

    :param grouping: The grouping the correlations are for.
    :param matrix: Correlations to plot, one row per model, with at least one model and one group.
    :returns: A Plotly figure with one scatter trace and two ``updatemenus``.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    first = _correlations(matrix.iloc[0])
    fig.add_trace(
        go.Scatter(
            x=first,
            y=first,
            mode="markers",
            marker={"size": 6, "opacity": 0.7},
            customdata=list(matrix.columns),
            hovertemplate=(
                f"{grouping.capitalize()}: %{{customdata}}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>"
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

    first_axis = _axis_layout(grouping, matrix.index[0])
    fig.update_layout(
        title={
            "text": f"Per-{grouping} Pearson correlation, model against model",
            "y": 0.99,
            "yanchor": "top",
        },
        showlegend=False,
        xaxis=first_axis,
        yaxis=first_axis,
        annotations=[
            {
                "text": f"{axis}-axis model:",
                "showarrow": False,
                "x": x,
                "xref": "paper",
                "xanchor": "left",
                "y": 1.12,
                "yref": "paper",
                "yanchor": "bottom",
            }
            for axis, x in _DROPDOWNS
        ],
        updatemenus=[
            {
                "buttons": _dropdown_buttons(grouping, matrix, axis),
                "direction": "down",
                "showactive": True,
                "x": x,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
            }
            for axis, x in _DROPDOWNS
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
    from plotly.utils import PlotlyJSONEncoder

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


def _matrices(result: ExperimentResult) -> dict[str, pd.DataFrame]:
    """Compute the per-group correlations of every model, for each grouping with something to compare.

    Each matrix is one ``float32`` frame of shape ``n_models x n_groups``, independent of
    the number of predictions.

    :param result: Experiment to read.
    :returns: Matrices keyed by grouping, in :data:`GROUPINGS` order, leaving out a grouping
        with fewer than two models that have a defined correlation or with no group.
    """
    matrices = {}
    for grouping in GROUPINGS:
        matrix = model_group_correlations(result, grouping)
        if matrix.shape[0] >= 2 and matrix.shape[1] > 0:
            matrices[grouping] = matrix
    return matrices


@register(
    "comparison_scatter",
    "Per-drug, per-sample and per-study correlation compared between two selectable models",
)
class ComparisonScatterPlot(Plot):
    """Model-against-model comparison of per-group correlation (Plotly)."""

    requirements = frozenset({PlotRequirement.MULTIPLE_MODELS})

    def figure(self, result: ExperimentResult) -> go.Figure:
        """Draw the two-dropdown figure of the first grouping with something to compare.

        :param result: Experiment result with at least two models.
        :returns: The Plotly figure; empty when fewer than two models have defined correlations.
        """
        import plotly.graph_objects as go

        matrices = _matrices(result)
        return _build_figure(*next(iter(matrices.items()))) if matrices else go.Figure()

    def sections(self, result: ExperimentResult) -> list[Section]:
        """Return one Section per grouping, each holding a two-dropdown figure.

        :param result: Experiment result with at least two models.
        :returns: Sections for the drug, sample and study groupings, in that order,
            leaving out a grouping with no defined group (such as ``"study"`` on a run
            that was not concatenated). Empty when fewer than two models have defined
            correlations.
        """
        sections: list[Section] = []
        for grouping, matrix in _matrices(result).items():
            label = grouping
            plural = "studies" if label == "study" else f"{label}s"
            anchor = f"dreval_comp_scatter_{grouping}"
            sections.append(
                Section(
                    name=f"{label.capitalize()}-wise comparison",
                    anchor=anchor,
                    description=(
                        f"Pearson correlation of predictions against ground truth within each {label}, "
                        f"for {matrix.shape[0]} models over {matrix.shape[1]} {plural}. "
                        "Pick the model on each axis with the dropdowns; points above the diagonal "
                        "favour the model on the y-axis."
                    ),
                    content=_inline_plotly_html(_build_figure(grouping, matrix), f"{anchor}_div"),
                )
            )
        return sections
