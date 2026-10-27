"""A small ``response.uns["replicate_fits"]`` table in the format the curation pipeline writes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from drevalpy.data import _replicate_fits


def fits_table(samples: Sequence[str], drugs: Sequence[str], *, extra: str | None = None) -> pd.DataFrame:
    """Two replicate rows for every (sample, drug) pair, with the dtypes curation writes.

    :param samples: Canonical sample IDs.
    :param drugs: Canonical drug IDs.
    :param extra: Name of an optional dataset-specific categorical column.
    :returns: The table, with a ``RangeIndex``.
    """
    pairs = [(sample, drug) for sample in samples for drug in drugs for _ in range(2)]
    n_rows = len(pairs)
    table = pd.DataFrame(
        {
            _replicate_fits.SAMPLE_COLUMN: pd.Categorical([sample for sample, _ in pairs]),
            _replicate_fits.DRUG_COLUMN: pd.Categorical([drug for _, drug in pairs]),
            "replicate": np.tile(np.array([1, 2], dtype=np.int32), n_rows // 2),
            "n_doses": np.full(n_rows, 9, dtype=np.int16),
            "status": pd.Categorical(["fitted"] * n_rows, categories=["fitted", "fit_failed", "too_few_doses"]),
            "pEC50": np.linspace(5, 7, n_rows, dtype=np.float32),
            "AUC": np.linspace(0, 1, n_rows, dtype=np.float32),
        }
    )
    if extra is not None:
        table[extra] = pd.Categorical([f"{extra}-{i % 3}" for i in range(n_rows)])
    return table
