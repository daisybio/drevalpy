"""Abstract wrapper class for all visualizations."""

from abc import ABC, abstractmethod
from io import TextIOWrapper


class OutPlot(ABC):
    """Abstract wrapper class for all visualizations."""

    @abstractmethod
    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """
        Draw and save the plot.

        :param out_prefix: path to output directory for python package
        :param out_suffix: custom suffix for output file
        """
        pass

    @abstractmethod
    def _draw(self) -> None:
        """Draw the plot."""
        pass

    @staticmethod
    @abstractmethod
    def write_to_html(lpo_lco_ldo: str, f: TextIOWrapper, *args, **kwargs) -> TextIOWrapper:
        """
        Write the plot to the final report file.

        :param lpo_lco_ldo: LPO, LCO, LDO
        :param f: the file to write to
        :param args: additional arguments
        :param kwargs: additional keyword arguments
        :return: the file to write to
        """
        pass
