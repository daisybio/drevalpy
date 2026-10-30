"""Notebook and console representations shared by every registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class TableReprMixin(ABC):
    """Notebook and console representations built from a registry's ``to_dataframe``."""

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Return the registry contents as a pandas DataFrame.

        :returns: One row per registered entry.
        """

    def __repr__(self) -> str:
        """Return a tabular string representation."""
        return self.to_dataframe().to_string(index=False)

    def _repr_html_(self) -> str:
        """Return an HTML table for Jupyter notebooks."""
        return self.to_dataframe().to_html(index=False)
