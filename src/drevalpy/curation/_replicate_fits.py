"""The replicate-fits table: every replicate's own curve fit, stored as ``uns["replicate_fits"]``.

One row per dose series (replicate, see :mod:`drevalpy.curation._series`) of a pair, placed in
the pair's replicate hierarchy by ``biological_replicate`` (one per experiment) and
``technical_replicate`` (one per series of that experiment):

* pairs with two or more fittable series carry each series' own fit (``source = "replicate"``);
* a pair with a single fittable series carries the pooled fit - fitted on exactly that series -
  so every fitted pair appears (``source = "pooled"``);
* series too short to fit are kept with ``status = "too_few_doses"`` and NaN metrics, and fits
  CurveCurator dropped with ``status = "fit_failed"``.

Each series is fitted on its own by tagging its sample label with ``__rep_<replicate>`` and
fitting the tagged rows the same way as the pooled fit; CurveCurator treats labels as opaque, so
every tagged pair is one curve.

The table is keyed like the AnnData it is stored on: ``sample_id`` and ``pubchem_id`` hold the
``obs_names`` and ``var_names`` (the input's ``sample`` and ``drug`` labels), and
``native_sample``/``native_drug`` keep those input labels when a pipeline later renames both
axes. :mod:`drevalpy.data._replicate_fits` is the reader of this format.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

from drevalpy.curation._anndata import _REGULATION_ENCODING
from drevalpy.curation._fit import STATUS_COLUMN, Status
from drevalpy.errors import InvalidInputError
from drevalpy.log import get_logger

logger = get_logger(__name__)

#: pEC50 followed by the layer metrics of :func:`drevalpy.curation.build_anndata`.
METRICS = [
    "pEC50",
    "EC50",
    "IC50",
    "LN_IC50",
    "AUC",
    "fold_change",
    "slope",
    "front",
    "back",
    "R2",
    "RMSE",
    "p_value",
    "log_p_value",
    "f_value",
    "f_value_sam",
    "relevance_score",
    "signal_quality",
    "regulation",
    "pec50_error",
    "slope_error",
    "front_error",
    "back_error",
]


class Source(StrEnum):
    """Which fit a row's metrics come from."""

    REPLICATE = "replicate"
    POOLED = "pooled"


#: Leading columns, in order; the metadata columns follow the metrics.
ID_COLUMNS = [
    "sample_id",
    "pubchem_id",
    "native_sample",
    "native_drug",
    "replicate",
    "series_key",
    "experiment",
    "biological_replicate",
    "technical_replicate",
    "n_doses",
    "source",
    "status",
]

#: Separator tagging a sample label with its replicate during the replicate fits.
REPLICATE_TAG = "__rep_"

_SERIES_ID_COLUMNS = [
    "native_sample",
    "native_drug",
    "replicate",
    "series_key",
    "experiment",
    "biological_replicate",
    "technical_replicate",
]
_INTEGER_COLUMNS = {"replicate": "int32", "biological_replicate": "int32", "technical_replicate": "int32"}
_NATIVE_PAIR = ["native_sample", "native_drug"]


def tag_replicates(viability: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of the pairs with two or more replicates, each replicate its own pair.

    The replicate is moved into the sample label (``<sample>__rep_<replicate>``), so fitting
    the result like a pooled table fits every replicate on its own. A sample or drug label that
    already contains :data:`REPLICATE_TAG` raises :class:`~drevalpy.errors.InvalidInputError`.

    :param viability: Long table with an integer ``replicate`` per pair.
    :returns: The tagged table, without the ``replicate`` column.
    """
    n_replicates = viability.groupby(["sample", "drug"])["replicate"].nunique()
    several = n_replicates[n_replicates >= 2].index
    viability = viability.merge(several.to_frame(index=False), on=["sample", "drug"], how="inner")
    logger.info("Fitting the replicates of %d of %d pairs on their own", len(several), len(n_replicates))
    _reject_tag_collisions(viability)
    tagged = viability.copy()
    tagged["sample"] = tagged["sample"].astype(str) + REPLICATE_TAG + tagged["replicate"].astype(str)
    return tagged.drop(columns=["replicate"])


def untag_replicates(fits: pd.DataFrame) -> pd.DataFrame:
    """Decode the fits of a :func:`tag_replicates` table back to (sample, drug, replicate).

    :param fits: :func:`~drevalpy.curation._fit.collect`'s result on the tagged table.
    :returns: One row per attempted (sample, drug, replicate): ``sample``, ``drug``,
        ``replicate``, :data:`~drevalpy.curation._fit.STATUS_COLUMN` and the metric columns.
    """
    fits = fits.copy()
    if fits.empty:
        fits.insert(2, "replicate", pd.Series([], dtype="int64"))
        return fits
    parts = fits["sample"].astype(str).str.rsplit(REPLICATE_TAG, n=1, expand=True)
    fits["sample"] = parts[0].to_numpy()
    fits.insert(2, "replicate", parts[1].astype("int64").to_numpy())
    return fits


def replicate_table(own: pd.DataFrame, pooled: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Build the replicate-fits table in its stored schema.

    :param own: :func:`untag_replicates`'s result.
    :param pooled: :func:`~drevalpy.curation._fit.collect`'s result on the pooled table.
    :param series: The series table of :func:`drevalpy.curation._series.number_series`.
    :returns: One row per series.
    """
    table = build(own, pooled, series)
    table.insert(0, "pubchem_id", table["native_drug"].to_numpy())
    table.insert(0, "sample_id", table["native_sample"].to_numpy())
    return to_schema(table)


def _reject_tag_collisions(viability: pd.DataFrame) -> None:
    """Fail if a label already contains :data:`REPLICATE_TAG`, which would make decoding ambiguous."""
    offenders: list[str] = []
    for column in ("sample", "drug"):
        labels = viability[column].astype(str).drop_duplicates()
        offenders.extend(f"{column} {label!r}" for label in labels[labels.str.contains(REPLICATE_TAG, regex=False)])
    if offenders:
        raise InvalidInputError(
            f"{len(offenders)} label(s) contain {REPLICATE_TAG!r}, which tags the replicate in the replicate "
            f"fits: {offenders[:5]}"
        )


def build(own: pd.DataFrame, pooled: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Assemble the table from the replicates' own fits and the pooled fits.

    :param own: :func:`untag_replicates`'s result.
    :param pooled: :func:`~drevalpy.curation._fit.collect`'s result on the pooled table.
    :param series: The series table.
    :returns: One row per series with every column of the schema except ``sample_id`` and
        ``pubchem_id``.
    """
    series = series.copy()
    n_eligible = series.groupby(_NATIVE_PAIR)["eligible"].transform("sum")
    has_own = series["eligible"] & (n_eligible >= 2)
    single = series["eligible"] & (n_eligible == 1)

    metrics = pd.DataFrame(np.nan, index=series.index, columns=pd.Index(METRICS), dtype=np.float32)
    status = pd.Series(Status.TOO_FEW_DOSES.value, index=series.index, dtype=object)

    keys = series.loc[has_own, [*_NATIVE_PAIR, "replicate"]].astype({"replicate": "int64"})
    fitted_own = _aligned(keys, own)
    metrics.loc[has_own] = fitted_own[METRICS].to_numpy(dtype=np.float32)
    status.loc[has_own] = fitted_own[STATUS_COLUMN].to_numpy()

    fitted_pooled = _aligned(series.loc[single, _NATIVE_PAIR], pooled)[METRICS].to_numpy(dtype=np.float32)
    metrics.loc[single] = fitted_pooled
    status.loc[single] = np.where(np.isfinite(fitted_pooled[:, 0]), Status.FITTED.value, Status.FIT_FAILED.value)

    meta_columns = [column for column in series.columns if column not in {*_SERIES_ID_COLUMNS, "n_doses", "eligible"}]
    table = series[[*_SERIES_ID_COLUMNS, "n_doses"]].copy()
    table["source"] = np.where(single, Source.POOLED.value, Source.REPLICATE.value)
    table["status"] = status.to_numpy()
    table[METRICS] = metrics.to_numpy()
    for column in meta_columns:
        table[column] = series[column].to_numpy()
    return table.reset_index(drop=True)


def _aligned(keys: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Align *fits* to the rows of *keys* (native columns), encoding ``regulation``; a missing fit is failed."""
    if keys.empty:
        return keys.assign(**{STATUS_COLUMN: pd.Series(dtype=object)}, **{metric: np.nan for metric in METRICS})
    fits = fits.rename(columns={"sample": "native_sample", "drug": "native_drug"}).copy()
    fits["native_sample"] = fits["native_sample"].astype(str)
    fits["native_drug"] = fits["native_drug"].astype(str)
    if "replicate" in keys.columns:
        fits["replicate"] = fits["replicate"].astype("int64")
    for column in METRICS:
        if column not in fits.columns:
            fits[column] = np.nan
    fits["regulation"] = fits["regulation"].map(_REGULATION_ENCODING).astype(float)
    on = list(keys.columns)
    merged = keys.merge(fits[[*on, STATUS_COLUMN, *METRICS]], on=on, how="left")
    merged[STATUS_COLUMN] = merged[STATUS_COLUMN].fillna(Status.FIT_FAILED.value)
    return merged


def to_schema(table: pd.DataFrame) -> pd.DataFrame:
    """Order the columns and set the stored dtypes (text as categoricals, metrics as float32).

    :param table: The table with every column of :data:`ID_COLUMNS` and :data:`METRICS`.
    :returns: The table in its stored schema, metadata columns after the metrics.
    """
    meta = [column for column in table.columns if column not in {*ID_COLUMNS, *METRICS}]
    table = table[[*ID_COLUMNS, *METRICS, *meta]].copy()
    for column in [*ID_COLUMNS, *meta]:
        if column not in {*_INTEGER_COLUMNS, "n_doses"}:
            table[column] = table[column].astype(object).where(table[column].notna(), None)
            table[column] = table[column].astype("category")
    table = table.astype(_INTEGER_COLUMNS)
    table["n_doses"] = table["n_doses"].astype("int16")
    table[METRICS] = table[METRICS].astype(np.float32)
    return table.reset_index(drop=True)
