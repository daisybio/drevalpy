"""Visualization base class and Section dataclass."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.figure

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
    def to_png(self, path: str | Path) -> None:
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

    def to_png(self, path: str | Path) -> None:
        """Save the figure to a PNG file.

        :param path: Output file path.
        """
        if self._fig is None:
            raise RuntimeError("Call compute() before to_png()")
        self._fig.savefig(str(path), dpi=150, bbox_inches="tight")

    def to_multiqc(self) -> list[Section]:
        """Embed figure as a base64-encoded PNG in a report Section."""
        if self._fig is None:
            raise RuntimeError("Call compute() before to_multiqc()")
        buf = BytesIO()
        self._fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return [
            Section(
                name=self.registry_name,
                anchor=self.registry_name,
                content=f'<img src="data:image/png;base64,{b64}" style="max-width:100%" />',
            )
        ]

    def show(self) -> None:
        """Display the figure in a Jupyter notebook."""
        if self._fig is None:
            raise RuntimeError("Call compute() before show()")
        from IPython.display import display

        display(self._fig)
