"""Visualization base class and Section dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    def generate(self, result: ExperimentResult | ModelResult) -> list[Section]:
        """Generate one or more report sections from the result.

        :param result: An ExperimentResult or ModelResult to visualize.
        :returns: List of Section objects for the MultiQC report.
        """
        ...
