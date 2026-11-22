"""The ``Plot`` interface, the report ``Section`` and the two renderers every plot shares.

A plot is stateless: :meth:`Plot.figure` and :meth:`Plot.sections` each take the result and
build what they return from it, so one instance can draw any number of results and nothing
outlives a call. :func:`save_png` and :func:`show` render a plot's figure, whichever library
drew it: a Matplotlib figure is written with ``savefig``, anything else is a Plotly figure.
"""

from __future__ import annotations

import base64
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar

from drevalpy.errors import DrEvalError
from drevalpy.types.results import ExperimentResult, ModelResult

if TYPE_CHECKING:
    import matplotlib.figure
    from upath import UPath

    from drevalpy.visualization.requirements import PlotRequirement


@dataclass
class Section:
    """A single report section to be added to the MultiQC report."""

    name: str
    anchor: str
    description: str = ""
    plot: Any = None
    content: str | None = None


def embedded_png_html(figure: matplotlib.figure.Figure) -> str:
    """Render *figure* as a self-contained ``<img>`` tag for a report Section.

    :param figure: Figure to rasterize.
    :returns: An ``<img>`` element carrying the PNG as a base64 data URI.
    """
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    payload = base64.b64encode(buffer.getvalue()).decode()
    return f'<img src="data:image/png;base64,{payload}" style="max-width:100%" />'


class Plot(ABC):
    """A report plot: one figure, and the report sections, for one result.

    ``result_type`` is :class:`~drevalpy.types.results.ExperimentResult` for a plot of the
    whole experiment, or :class:`~drevalpy.types.results.ModelResult` for a plot the report
    draws once per model. The plot applies only to experiments that meet every one of its
    ``requirements``. ``name`` is set by
    :func:`drevalpy.registry.visualization.register`.
    """

    name: ClassVar[str] = ""
    result_type: ClassVar[type[ExperimentResult] | type[ModelResult]] = ExperimentResult
    requirements: ClassVar[frozenset[PlotRequirement]] = frozenset()

    @abstractmethod
    def figure(self, result: Any) -> Any:
        """Draw the plot of *result*.

        :param result: A result of type ``result_type``.
        :returns: A Matplotlib or Plotly figure.
        """

    def sections(self, result: Any) -> list[Section]:
        """Return the report sections for *result*.

        The default embeds :meth:`figure`, which must then be a Matplotlib figure, as one PNG
        section named and anchored after the plot.

        :param result: A result of type ``result_type``.
        :returns: The sections, in order; empty when there is nothing to show.
        """
        return [Section(name=self.name, anchor=self.name, content=embedded_png_html(self.figure(result)))]


def plotly_png_supported() -> bool:
    """Whether Plotly figures can be written as PNG on this platform.

    Kaleido 0.2.1, the version MultiQC pins, hangs on Windows: its bundled Chromium starts
    but never reports ready. MultiQC 1.35 requires exactly that version.

    :returns: ``False`` on Windows, ``True`` elsewhere.
    """
    return sys.platform != "win32"


def is_matplotlib_figure(figure: Any) -> bool:
    """Whether *figure* is a Matplotlib figure; the other kind a plot draws is a Plotly figure.

    :param figure: A figure returned by :meth:`Plot.figure`.
    :returns: ``True`` for a :class:`matplotlib.figure.Figure`.
    """
    from matplotlib.figure import Figure

    return isinstance(figure, Figure)


def write_png(figure: Any, path: str | UPath) -> None:
    """Write a Matplotlib or Plotly *figure* to a PNG file.

    :param figure: A figure returned by :meth:`Plot.figure`.
    :param path: Output file path.
    :raises DrEvalError: For a Plotly figure on Windows, where Kaleido 0.2.1 hangs instead of rendering.
    """
    if is_matplotlib_figure(figure):
        figure.savefig(str(path), dpi=150, bbox_inches="tight")
        return
    if not plotly_png_supported():
        raise DrEvalError(
            "Writing a Plotly figure as PNG is not supported on Windows: Kaleido 0.2.1, the version "
            "MultiQC requires, hangs there. Use show() in a notebook, or the HTML report."
        )
    figure.write_image(str(path))


def save_png(plot: Plot, result: Any, path: str | UPath) -> None:
    """Draw *plot* for *result* and write it to a PNG file.

    A Plotly plot raises :class:`~drevalpy.errors.DrEvalError` on Windows; see
    :func:`plotly_png_supported`.

    :param plot: The plot to draw.
    :param result: A result of the plot's ``result_type``.
    :param path: Output file path.
    """
    write_png(plot.figure(result), path)


def show(plot: Plot, result: Any) -> None:
    """Draw *plot* for *result* and display it in a Jupyter notebook.

    :param plot: The plot to draw.
    :param result: A result of the plot's ``result_type``.
    """
    figure = plot.figure(result)
    if is_matplotlib_figure(figure):
        from IPython.display import display

        display(figure)
    else:
        figure.show()
