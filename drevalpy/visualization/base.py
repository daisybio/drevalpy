"""Visualization base class and Section dataclass."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.figure
    import plotly.graph_objects as go
    from upath import UPath

    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.results import ExperimentResult, ModelResult


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


def require_figure(figure: Any, caller: str) -> Any:
    """Return *figure*, rejecting the not-yet-computed state.

    :param figure: The visualization's figure, ``None`` before ``compute()``.
    :param caller: Name of the method being guarded, for the error message.
    :returns: The figure.
    :raises RuntimeError: If *figure* is ``None``.
    """
    if figure is None:
        msg = f"Call compute() before {caller}()"
        raise RuntimeError(msg)
    return figure


class Visualization(ABC):
    """Base class for all visualizations producing MultiQC report sections."""

    registry_name: str = ""

    @abstractmethod
    def compute(self, result: ExperimentResult | ModelResult, dataset: Dataset | None = None) -> None:
        """Compute the visualization data from the result (store internally).

        :param result: An ExperimentResult or ModelResult to visualize.
        :param dataset: Optional dataset for looking up drug/cell-line metadata.
        """
        ...

    @abstractmethod
    def to_png(self, path: str | UPath) -> None:
        """Render to a static PNG file.

        :param path: File path for the output PNG.
        """
        ...

    @abstractmethod
    def to_multiqc(self) -> list[Section]:
        """Return MultiQC Section objects for report integration.

        Implementations may use native MultiQC plot objects (Path A)
        or embed a base64-encoded image via Section.content (Path B).
        """
        ...

    @abstractmethod
    def show(self) -> None:
        """Display interactively in a Jupyter notebook."""
        ...


class ImageVisualization(Visualization):
    """Base for plots that render as a static image (no native MultiQC type).

    Subclasses implement only compute() and _create_figure(). The base class
    handles to_png(), to_multiqc(), and show() automatically.
    """

    _fig: matplotlib.figure.Figure | None = None

    @abstractmethod
    def _create_figure(self) -> matplotlib.figure.Figure:
        """Create and return the matplotlib Figure."""
        ...

    def to_png(self, path: str | UPath) -> None:
        """Save the figure to a PNG file.

        :param path: Output file path.
        """
        require_figure(self._fig, "to_png").savefig(str(path), dpi=150, bbox_inches="tight")

    def to_multiqc(self) -> list[Section]:
        """Embed figure as a base64-encoded PNG in a report Section."""
        figure = require_figure(self._fig, "to_multiqc")
        return [
            Section(
                name=self.registry_name,
                anchor=self.registry_name,
                content=embedded_png_html(figure),
            )
        ]

    def show(self) -> None:
        """Display the figure in a Jupyter notebook."""
        figure = require_figure(self._fig, "show")
        from IPython.display import display

        display(figure)


class PlotlyVisualization(Visualization):
    """Base for plots whose ``compute()`` leaves a Plotly figure in ``_fig``.

    Subclasses implement compute() and to_multiqc(); rendering the figure to a
    PNG or into a notebook is the same call for every one of them.
    """

    _fig: go.Figure | None = None

    def to_png(self, path: str | UPath) -> None:
        """Render the figure to a static PNG.

        :param path: Output file path.
        """
        require_figure(self._fig, "to_png").write_image(str(path))

    def show(self) -> None:
        """Display the figure in a Jupyter notebook."""
        require_figure(self._fig, "show").show()
