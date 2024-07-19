from abc import ABC, abstractmethod
from typing import TextIO


class OutPlot(ABC):
    """
    Abstract wrapper class for all visualizations
    """

    @abstractmethod
    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """
        Draw and save the plot
        :param out_prefix: path to output directory for python package
        :param out_suffix: custom suffix for output file
        :return:
        """
        pass

    @abstractmethod
    def __draw__(self) -> None:
        """
        Draw the plot
        :return:
        """
        pass

    @staticmethod
    def write_to_html(lpo_lco_ldo: str, f: TextIO, *args, **kwargs) -> TextIO:
        """
        Write the plot to html
        :param lpo_lco_ldo:
        :param f:
        :param args:
        :param kwargs:
        :return:
        """
        pass
