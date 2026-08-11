"""Abstract wrapper class for all visualizations."""

from abc import ABC, abstractmethod
from io import TextIOWrapper

from upath import UPath as Path


class OutPlot(ABC):
    """Abstract base for report plot classes."""

    result_type: str = "ExperimentResult"
    requirements: frozenset = frozenset()

    @abstractmethod
    def draw_and_save(self, out_prefix: str | Path, out_suffix: str) -> None:
        """Draw the plot and write it to disk.

        :param out_prefix: Output directory path.
        :param out_suffix: Filename suffix for the saved artifact.
        """
        pass

    @abstractmethod
    def _draw(self) -> None:
        """Draw the plot."""
        pass

    @staticmethod
    @abstractmethod
    def write_to_html(test_mode: str, f: TextIOWrapper, *_unused_args, **_kwargs) -> TextIOWrapper:
        """Embed or link the plot in an HTML report.

        :param test_mode: Evaluation test mode (for example ``"LCO"``).
        :param f: Open HTML file handle to append content to.
        :param _unused_args: Plot-specific positional arguments.
        :param _kwargs: Plot-specific keyword arguments.

        :returns: The same file handle after writing.
        """
        pass
