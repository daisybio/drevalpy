"""The per-replicate curve fits stored in ``response.uns["replicate_fits"]``.

The curation pipeline writes one row per replicate dose series of a (sample, drug)
pair: the canonical IDs in :data:`SAMPLE_COLUMN` and :data:`DRUG_COLUMN`, the fit
status and metrics, and dataset-specific metadata. Text columns are categoricals.
This module is the one place that knows that format; subsetting and
:func:`drevalpy.data.concat` go through it.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

from ._concat_tables import prefixed

if TYPE_CHECKING:
    import pandas as pd

#: Key of the table in the ``uns`` of a response-shaped modality.
KEY = "replicate_fits"
#: Column holding the canonical sample ID (a ``response.obs_names`` entry).
SAMPLE_COLUMN = "sample_id"
#: Column holding the canonical drug ID (a ``response.var_names`` entry).
DRUG_COLUMN = "pubchem_id"


def _drop_unused_categories(table: pd.DataFrame) -> pd.DataFrame:
    """Remove the categories no row uses from every categorical column."""
    import pandas as pd

    for column in table.columns:
        if isinstance(table[column].dtype, pd.CategoricalDtype):
            table[column] = table[column].cat.remove_unused_categories()
    return table


def restrict(table: pd.DataFrame, samples: Collection[str], drugs: Collection[str]) -> pd.DataFrame:
    """Keep the rows whose sample and drug are among the given IDs.

    :param table: A replicate-fits table.
    :param samples: Sample IDs to keep.
    :param drugs: Drug IDs to keep.
    :returns: A new table with a ``RangeIndex`` and no unused categories.
    """
    keep = table[SAMPLE_COLUMN].astype(object).isin(list(samples)) & table[DRUG_COLUMN].astype(object).isin(list(drugs))
    return _drop_unused_categories(table.loc[keep.to_numpy()].reset_index(drop=True))


def _with_prefixed_samples(table: pd.DataFrame, study: str) -> pd.DataFrame:
    """Copy of *table* with ``<study>::`` before each sample ID, keeping a categorical dtype."""
    import pandas as pd

    out = table.reset_index(drop=True)
    column = out[SAMPLE_COLUMN]
    ids = prefixed(study, pd.Index(column.astype(object)))
    if isinstance(column.dtype, pd.CategoricalDtype):
        out[SAMPLE_COLUMN] = pd.Categorical(ids)
    else:
        out[SAMPLE_COLUMN] = ids.to_numpy()
    return out


def _restore_dtype(stacked: pd.Series, columns: Sequence[pd.Series]) -> pd.Series:
    """Give a stacked column back the dtype its inputs shared, where NaN filling changed it."""
    import pandas as pd

    dtypes = [column.dtype for column in columns]
    if all(isinstance(dtype, pd.CategoricalDtype) for dtype in dtypes):
        categories = list(dict.fromkeys(item for dtype in dtypes for item in dtype.categories))
        return pd.Series(pd.Categorical(stacked.astype(object), categories=categories), index=stacked.index)
    first = dtypes[0]
    if all(dtype == first for dtype in dtypes) and stacked.dtype != first and first.kind == "f":
        return stacked.astype(first)
    return stacked


def stack(tables: Sequence[pd.DataFrame | None], names: Sequence[str]) -> pd.DataFrame | None:
    """Stack the replicate-fits tables of several inputs, as :func:`drevalpy.data.concat` stacks rows.

    :param tables: The table of each input, or ``None`` where an input has none.
    :param names: Dataset name of each input; it prefixes the sample IDs as ``<study>::<sample_id>``.
    :returns: One table with the union of the columns in first-seen order, ``NaN`` where an
        input lacks a column, and categoricals over the union of the categories; ``None`` if
        any input has no table.
    """
    import pandas as pd

    if any(table is None for table in tables):
        return None
    frames = [
        _with_prefixed_samples(table, name) for table, name in zip(tables, names, strict=True) if table is not None
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        stacked = pd.concat(frames, join="outer", ignore_index=True)
    for column in stacked.columns:
        present = [frame[column] for frame in frames if column in frame.columns]
        stacked[column] = _restore_dtype(stacked[column], present)
    return stacked
