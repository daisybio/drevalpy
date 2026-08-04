"""Abstract wrapper class for all visualizations."""

from abc import ABC, abstractmethod
from io import TextIOWrapper


class OutPlot(ABC):
    """Abstract base for report plot classes."""

    @abstractmethod
    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """Draw the plot and write it to disk.

        Args:
            out_prefix: Output directory path.
            out_suffix: Filename suffix for the saved artifact.
        """
        pass

    @abstractmethod
    def _draw(self) -> None:
        """Draw the plot."""
        pass

    @staticmethod
    @abstractmethod
    def write_to_html(test_mode: str, f: TextIOWrapper, *args, **kwargs) -> TextIOWrapper:
        """Embed or link the plot in an HTML report.

        Args:
            test_mode: Evaluation test mode (for example ``"LCO"``).
            f: Open HTML file handle to append content to.
            *args: Plot-specific positional arguments.
            **kwargs: Plot-specific keyword arguments.

        Returns:
            The same file handle after writing.
        """
        pass
