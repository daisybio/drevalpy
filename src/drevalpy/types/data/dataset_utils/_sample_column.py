"""Per-sample lookup of one ``obs`` column, shared by every metadata accessor.

``get_tissue`` and the study-aware splitters read the same table, so the lookup
lives here once instead of being repeated per column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mudata as md


class SampleColumnMixin:
    """Adds :meth:`sample_column` to a class that holds an ``_mdata`` MuData object."""

    _mdata: md.MuData

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        """Return one ``obs`` column for the given sample IDs.

        :param column: Name of a column of the sample metadata (``obs``), such as
            ``"tissue"`` or ``"study"``.
        :param ids: 1-D array of sample IDs.
        :returns: 1-D array with one value per requested ID, in the order of *ids*.
            IDs that are not samples of this dataset give NaN.
        :raises KeyError: If *column* is not in ``obs``; the message lists the available columns.
        """
        import pandas as pd

        obs = self._mdata.obs
        if column not in obs.columns:
            raise KeyError(f"Sample column '{column}' not found. Available: {list(obs.columns)}")

        positions = pd.Index(obs.index).get_indexer(np.asarray(ids, dtype=str))
        values = obs[column].to_numpy()
        valid = positions >= 0
        if valid.all():
            return values[positions]
        result = np.full(len(positions), np.nan, dtype=float if values.dtype.kind in "fiub" else object)
        result[valid] = values[positions[valid]]
        return result
