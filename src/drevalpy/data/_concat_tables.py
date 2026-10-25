"""Table helpers for :func:`drevalpy.data.concat`: prefixed IDs, ``obs`` stacking, drug ``var`` union."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from drevalpy.errors import InvalidInputError
from drevalpy.evaluation._grouping import STUDY_SEPARATOR

if TYPE_CHECKING:
    import pandas as pd

SMILES_COLUMN = "canonical_smiles"
_MAX_LISTED = 8


def prefixed(study: str, ids: pd.Index) -> pd.Index:
    """Prefix every ID with ``<study>::``, keeping the index name."""
    import pandas as pd

    return pd.Index([f"{study}{STUDY_SEPARATOR}{item}" for item in ids], name=ids.name)


def listed(items: Sequence[object]) -> str:
    """Comma-join the first few *items*, saying how many more there are."""
    shown = ", ".join(str(item) for item in items[:_MAX_LISTED])
    extra = len(items) - _MAX_LISTED
    return f"{shown} and {extra} more" if extra > 0 else shown


def _tidy(column: pd.Series, categorical: bool) -> pd.Series:
    """Restore the dtype a column had before NaN filling made it ``object``."""
    if categorical:
        return column.astype("category")
    if column.dtype != object:
        return column
    column = column.infer_objects()
    if column.dtype == object and column.dropna().map(lambda value: isinstance(value, (bool, np.bool_))).all():
        return column.astype("boolean")
    return column


def _is_categorical(frames: Sequence[pd.DataFrame], column: str) -> bool:
    """Whether *column* is categorical in every frame that has it."""
    import pandas as pd

    dtypes = [frame[column].dtype for frame in frames if column in frame.columns]
    return all(isinstance(dtype, pd.CategoricalDtype) for dtype in dtypes)


def stack_obs(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Stack ``obs`` tables row-wise; a column missing in one table is NaN there.

    :param frames: Tables whose indexes are already unique across tables.
    :returns: A new table with the union of the columns in first-seen order, keeping a
        categorical dtype where every table had one.
    """
    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        stacked = pd.concat(list(frames), join="outer")
    for column in stacked.columns:
        stacked[column] = _tidy(stacked[column], _is_categorical(frames, column))
    return stacked


def _check_smiles(frames: Sequence[pd.DataFrame]) -> None:
    """Raise if one drug ID carries two different ``canonical_smiles``."""
    import pandas as pd

    with_smiles = [frame[SMILES_COLUMN].astype(object) for frame in frames if SMILES_COLUMN in frame.columns]
    if len(with_smiles) < 2:
        return
    counts = pd.concat(with_smiles).dropna().groupby(level=0).nunique()
    conflicting = list(counts.index[counts > 1])
    if conflicting:
        raise InvalidInputError(
            f"{SMILES_COLUMN} differs between datasets for {len(conflicting)} drug IDs: {listed(conflicting)}. "
            "One drug ID needs one structure; use curated files that share it."
        )


def union_var(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Union the drug tables of several inputs.

    Raises ``InvalidInputError`` if a drug has different ``canonical_smiles`` in different inputs.

    :param frames: One drug ``var`` table per input, indexed by drug ID.
    :returns: A table over the drug IDs in first-seen order. For each drug and column, the
        first non-null value wins.
    """
    import pandas as pd

    _check_smiles(frames)
    ids = pd.Index([item for frame in frames for item in frame.index]).unique()
    ids.name = frames[0].index.name
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    merged = pd.DataFrame(index=ids)
    for column in columns:
        values = pd.Series(np.nan, index=ids, dtype=object)
        for frame in frames:
            if column in frame.columns:
                values = values.where(values.notna(), frame[column].astype(object).reindex(ids))
        merged[column] = _tidy(values, _is_categorical(frames, column))
    return merged
