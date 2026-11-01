"""Turn well-level measurements into numbered dose series, the unit every replicate fit is made of.

A **series** is one dose-response series of one pair: the wells that share a value of the
input's ``replicate`` column (typically one plate) within one ``experiment``. Without that
column every pair has one series per experiment. Defining a series by a label rather than by
counting duplicate wells per dose is what keeps a missing well from shifting part of one
plate's series into another's.

:func:`number_series` holds every rule about series:

* **Labels.** ``sample`` and ``drug`` become string labels (integral floats lose their ``.0``,
  so a ``DRUG_ID`` read as ``float64`` still reads ``"1001"``). A label containing ``|`` is
  rejected, because CurveCurator packs both labels into one ``Name`` with it.
* **Repeat blocks.** When every dose of a series appears exactly ``k >= 2`` times, the wells
  are ordered by the ``position`` column and split into ``k`` standalone series, keyed
  ``<replicate>#<block>``. Any other repetition (only some doses repeated, or a single dose
  repeated) is averaged per dose, as is all repetition without ``position``.
* **Short series.** A series with fewer than ``min_doses`` distinct doses cannot be fitted on
  its own (CurveCurator needs ``5 - R`` non-missing values per curve), and as an extra pooled
  replicate its empty columns would count as missing for the whole pair. It is left out of the
  viability table and kept in the series table with ``eligible = False``.
* **Numbering.** Within a pair, ``biological_replicate`` is ``0..m-1`` over the distinct
  ``experiment`` values (sorted), and ``technical_replicate`` is ``0..k-1`` over the series of
  one experiment, ordered by series key. The flat ``replicate`` is ``0..n-1`` per pair in
  (biological, technical) order, eligible series first and then the short ones, so the
  viability table's replicates are contiguous.
* **Experiment.** One experiment is one independently seeded culture: its series are
  technical replicates of each other, and different experiments of a pair are biological
  replicates. Without an ``experiment`` column every series of a pair shares one experiment.
* **Metadata.** The ``metadata`` columns are lowercased and kept per series; a series spanning
  several values keeps them ``;``-joined.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from drevalpy.errors import InvalidInputError

#: Columns of the viability table :func:`number_series` returns, the input of the pooled fit.
VIABILITY_COLUMNS = ["drug", "sample", "concentration", "intensity", "replicate"]

#: Columns of the series table before the metadata columns.
SERIES_COLUMNS = [
    "native_sample",
    "native_drug",
    "replicate",
    "series_key",
    "experiment",
    "biological_replicate",
    "technical_replicate",
    "n_doses",
    "eligible",
]

#: Input columns: the required ones, then the optional ones.
REQUIRED_COLUMNS = ("sample", "drug", "concentration", "intensity")
OPTIONAL_COLUMNS = ("experiment", "replicate", "position")

#: Separator CurveCurator packs the two labels into its ``Name`` column with.
LABEL_SEPARATOR = "|"

_VALUE_SEPARATOR = ";"
_BLOCK_SEPARATOR = "#"
_GROUP = ["sample", "drug", "experiment", "series_key"]
_KEYS = ["experiment", "series_key"]


class Series(NamedTuple):
    """What is fitted, and what each replicate is."""

    viability: pd.DataFrame
    """Exactly :data:`VIABILITY_COLUMNS`, eligible series only, ``replicate`` numbered per pair."""

    series: pd.DataFrame
    """One row per series: :data:`SERIES_COLUMNS`, then the lowercased metadata columns."""


def number_series(df: pd.DataFrame, *, metadata: Sequence[str] = (), min_doses: int) -> Series:
    """Group wells into numbered series and average repeated doses.

    Rows missing a label, concentration or intensity are dropped.

    :param df: Well-level frame with the :data:`REQUIRED_COLUMNS`, any of the
        :data:`OPTIONAL_COLUMNS`, and the *metadata* columns.
    :param metadata: Columns to keep per series in the series table.
    :param min_doses: Minimum number of distinct doses for a series to be fitted.
    :returns: The viability table and the series table.
    """
    frame = _wells(df, metadata)
    if "position" in df.columns:
        frame = _split_repeat_blocks(frame)

    averaged = _str_keys(
        frame.groupby([*_GROUP, "concentration"], observed=True, sort=False)["intensity"].mean().reset_index()
    )
    series = _series_table(frame, averaged, metadata, min_doses=min_doses)

    eligible = series.loc[series["eligible"], ["native_sample", "native_drug", *_KEYS, "replicate"]]
    viability = averaged.merge(
        eligible.rename(columns={"native_sample": "sample", "native_drug": "drug"}), on=_GROUP, how="inner"
    )
    viability = viability.sort_values(["sample", "drug", "replicate", "concentration"], kind="stable")
    viability = viability[VIABILITY_COLUMNS].reset_index(drop=True)
    viability["sample"] = viability["sample"].astype(str)
    viability["drug"] = viability["drug"].astype(str)
    viability["replicate"] = viability["replicate"].astype("int64")
    return Series(viability=viability, series=series)


def _str_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast the experiment and series keys of a grouped frame to plain strings, for merging."""
    return frame.astype({column: str for column in _KEYS})


def _wells(df: pd.DataFrame, metadata: Sequence[str]) -> pd.DataFrame:
    """Label the pair, drop incomplete rows, and build the string series and experiment keys."""
    frame = pd.DataFrame(
        {
            "sample": _labels(df["sample"]),
            "drug": _labels(df["drug"]),
            "concentration": pd.to_numeric(df["concentration"], errors="coerce"),
            "intensity": pd.to_numeric(df["intensity"], errors="coerce"),
        },
        index=df.index,
    )
    frame["series_key"] = _key(df, "replicate")
    frame["experiment"] = _key(df, "experiment")
    for column in metadata:
        frame[f"meta:{column}"] = _as_category(df[column])
    if "position" in df.columns:
        frame["_position"] = df["position"].to_numpy()
    frame = frame.dropna(subset=["sample", "drug", "concentration", "intensity"])
    frame["concentration"] = frame["concentration"].astype(float).round(10)
    frame["sample"] = frame["sample"].astype("category")
    frame["drug"] = frame["drug"].astype("category")
    return frame


def _labels(values: pd.Series) -> pd.Series:
    """Cast a label column to strings, missing entries left missing.

    :param values: The ``sample`` or ``drug`` column, which may be numeric.
    :returns: The labels as strings, integral floats without their decimal part.
    :raises InvalidInputError: If any label contains :data:`LABEL_SEPARATOR`.
    """
    codes, uniques = pd.factorize(values)
    rendered = pd.Series([_render(value) for value in uniques], dtype=object)
    offenders = sorted({label for label in rendered if LABEL_SEPARATOR in label})
    if offenders:
        raise InvalidInputError(
            f"{len(offenders)} label(s) of {values.name!r} contain {LABEL_SEPARATOR!r}, which CurveCurator uses "
            f"to pack sample and drug into one label: {offenders[:5]}"
        )
    labels = np.full(len(values), None, dtype=object)
    present = codes >= 0
    labels[present] = rendered.to_numpy()[codes[present]]
    return pd.Series(labels, index=values.index, dtype=object)


def _render(value: object) -> str:
    """Render one label, key or metadata value as a string, dropping a float's ``.0``."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _as_category(values: pd.Series) -> pd.Categorical:
    """Render a column as categorical strings (integral floats lose their ``.0``), keeping missing values."""
    codes, uniques = pd.factorize(values)
    if len(uniques) == 0:
        return pd.Categorical.from_codes(codes, categories=pd.Index([], dtype=object))
    rendered = np.array([_render(value) for value in uniques], dtype=object)
    labels, inverse = np.unique(rendered.astype(str), return_inverse=True)
    mapped = np.where(codes >= 0, inverse[np.maximum(codes, 0)], -1)
    return pd.Categorical.from_codes(mapped, categories=pd.Index(labels, dtype=object))


def _key(df: pd.DataFrame, column: str) -> pd.Categorical:
    """Render *column* as a string key; missing values, and a missing column, render as ``""``."""
    if column not in df.columns:
        return pd.Categorical.from_codes(np.zeros(len(df), dtype=np.int64), categories=[""])
    part = _as_category(df[column])
    labels = np.asarray(["", *map(str, part.categories)], dtype=object)
    unique_labels, inverse = np.unique(labels.astype(str), return_inverse=True)
    return pd.Categorical.from_codes(inverse[part.codes + 1], categories=pd.Index(unique_labels, dtype=object))


def _split_repeat_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Split series whose every dose (of at least two) repeats ``k >= 2`` times into ``k`` series, by position."""
    per_dose = frame.groupby([*_GROUP, "concentration"], observed=True).agg(n=("intensity", "size")).reset_index()
    stats = per_dose.groupby(_GROUP, observed=True)["n"].agg(["min", "max", "size"])
    # A single dose repeated k times (a one-dose control compound) is no dose series to split.
    repeated = stats[(stats["min"] == stats["max"]) & (stats["min"] >= 2) & (stats["size"] >= 2)].index
    if repeated.empty:
        return frame
    mask = pd.MultiIndex.from_frame(frame[_GROUP]).isin(repeated)
    split = frame.loc[mask].sort_values([*_GROUP, "concentration", "_position"], kind="stable")
    block = split.groupby([*_GROUP, "concentration"], observed=True).cumcount()
    keys = split["series_key"].astype(str) + _BLOCK_SEPARATOR + block.astype(str)
    out = frame.copy()
    out["series_key"] = out["series_key"].astype(object)
    out.loc[split.index, "series_key"] = keys
    out["series_key"] = out["series_key"].astype(str).astype("category")
    return out


def _series_table(
    frame: pd.DataFrame, averaged: pd.DataFrame, metadata: Sequence[str], *, min_doses: int
) -> pd.DataFrame:
    """One row per series: numbering, experiment, replicate indices, dose count and metadata."""
    series = _str_keys(frame.groupby(_GROUP, observed=True, sort=False).size().reset_index()[_GROUP])
    n_doses = averaged.groupby(_GROUP, observed=True, sort=False).agg(n_doses=("concentration", "size")).reset_index()
    series = series.merge(n_doses, on=_GROUP, how="left")
    series["eligible"] = series["n_doses"] >= min_doses

    series = series.sort_values(["sample", "drug", "experiment", "series_key"], kind="stable")
    within = series.groupby(["sample", "drug", "experiment"], observed=True, sort=False)
    series["technical_replicate"] = within.cumcount().astype("int32")
    first_of_experiment = (series["technical_replicate"] == 0).astype("int32")
    pair_starts = first_of_experiment.groupby([series["sample"], series["drug"]], observed=True, sort=False).cumsum()
    series["biological_replicate"] = (pair_starts - 1).astype("int32")

    series = series.sort_values(
        ["sample", "drug", "eligible", "biological_replicate", "technical_replicate"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    series["replicate"] = series.groupby(["sample", "drug"], observed=True).cumcount().astype("int32")
    series["n_doses"] = series["n_doses"].astype("int16")

    for column in metadata:
        series = series.merge(_metadata_values(frame, column), on=_GROUP, how="left")

    series = series.rename(columns={"sample": "native_sample", "drug": "native_drug"})
    series["native_sample"] = series["native_sample"].astype(str)
    series["native_drug"] = series["native_drug"].astype(str)
    columns = [*SERIES_COLUMNS, *(column.lower() for column in metadata)]
    return series[columns].reset_index(drop=True)


def _metadata_values(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Collect the distinct values of metadata *column* per series, ``;``-joined when there are several.

    The join runs on plain arrays rather than a per-group ``agg``: a column that differs in
    every well (a well ID) over millions of series would otherwise take hours.
    """
    source = f"meta:{column}"
    distinct = frame[[*_GROUP, source]].dropna(subset=[source]).drop_duplicates()
    if distinct.empty:
        return _str_keys(distinct[_GROUP]).assign(**{column.lower(): pd.Series(dtype=object)})
    ids = distinct.groupby(_GROUP, observed=True, sort=False).ngroup().to_numpy()
    codes = distinct[source].cat.codes.to_numpy()
    # Categories are sorted strings, so ordering by (series, code) sorts each series' values.
    order = np.lexsort((codes, ids))
    ids = ids[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    rendered = np.asarray(distinct[source].cat.categories, dtype=object)[codes[order]]
    joined = rendered[starts]
    several = np.flatnonzero(ends - starts > 1)
    if len(several):
        listed = rendered.tolist()
        joined[several] = [
            _VALUE_SEPARATOR.join(listed[a:b]) for a, b in zip(starts[several], ends[several], strict=True)
        ]
    values = distinct[_GROUP].iloc[order[starts]].reset_index(drop=True)
    values[column.lower()] = pd.Series(joined, dtype=object)
    return _str_keys(values)
