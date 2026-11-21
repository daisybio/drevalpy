"""Dose-response curve fitting via curve_curator, returning AnnData.

:func:`curate` is the single entry point. It groups the well-level measurements into
numbered dose series (:mod:`drevalpy.curation._series`), fits one pooled curve per pair
(:mod:`drevalpy.curation._fit`), optionally fits every series on its own
(:mod:`drevalpy.curation._replicate_fits`), and pivots the metrics into an
:class:`~anndata.AnnData`. Both fits are split into chunks of curves that are fitted
independently, in local processes here; ``drevalpy curate plan``, ``fit-chunk`` and
``collect`` run the same steps as separate jobs (:mod:`drevalpy.curation._plan`), which is how
a workflow manager fits a large screen on a cluster.

Fitting is identifier-agnostic - CurveCurator groups by dose range and treats
``sample``/``drug`` purely as labels - and :func:`curate` keys the returned
``obs_names``/``var_names`` from whatever those two columns held. A pipeline can
therefore curate on *native* identifiers, persist the ``.h5ad``, and resolve
identity (Cellosaurus accessions, PubChem CIDs, duplicate collapse) in a later,
cheap stage that only renames indices. The ``.h5ad`` is that intermediate, so no
flat metrics frame needs to escape this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.curation._anndata import build_anndata
from drevalpy.curation._fit import Status
from drevalpy.curation._replicate_fits import METRICS, Source
from drevalpy.errors import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import anndata
    import pandas as pd

    from drevalpy.curation._plan import Plan
#: Only OLS is usable today. The pinned curve_curator fork raises
#: ``TypeError: _Model.fit_mle() got an unexpected keyword argument 'weights'``
#: partway through the fit, so MLE is rejected up front instead of failing
#: after preprocessing. Re-enable by adding "MLE" back once the fork is fixed.
SUPPORTED_FIT_TYPES = ("OLS",)

#: Fitting thoroughness levels curve_curator accepts. ``exhaustive`` is the
#: default and the only one the curation pipeline should use: ``fast`` takes a
#: single shot from one initial guess, which is what made published pEC50 values
#: depend on the starting point. The cheaper levels stay reachable because the
#: test suite fits hundreds of curves whose exact parameters it does not assert.
FIT_SPEEDS = ("fast", "standard", "exhaustive", "basinhopping")

DEFAULT_FIT_SPEED = "exhaustive"

__all__ = [
    "DEFAULT_FIT_SPEED",
    "FIT_SPEEDS",
    "METRICS",
    "SUPPORTED_FIT_TYPES",
    "Source",
    "Status",
    "build_anndata",
    "curate",
]


def curate(
    df: pd.DataFrame,
    *,
    replicate_fits: bool = False,
    min_doses: int = 4,
    metadata: Sequence[str] = (),
    normalize: bool = False,
    fit_type: str = "OLS",
    fit_speed: str = DEFAULT_FIT_SPEED,
    max_chunk_size: int = 2000,
    max_workers: int = 4,
) -> anndata.AnnData:
    """Fit dose-response curves and return an AnnData of curve metrics.

    The wells of each pair are grouped into dose series (one per ``replicate`` value), and
    the series of each pair are fitted together as one pooled curve. With *replicate_fits*,
    every series is also fitted on its own, which shows how well the replicates of a pair
    agree. See :mod:`drevalpy.curation._series` for how series are formed and numbered.

    Before anything is fitted, :class:`~drevalpy.errors.InvalidInputError` rejects a *df* that
    lacks a required column or has an unexpected one, and an unsupported *fit_type* or
    *fit_speed*.

    :param df: Well-level table. Required columns: ``sample``, ``drug``, ``concentration``
        (in µM) and ``intensity`` (viability relative to the untreated control). Optional:
        ``experiment`` (one independently seeded culture: series of one experiment are
        technical replicates, different experiments of a pair biological replicates),
        ``replicate`` (one dose series, such as a plate, within its experiment) and
        ``position`` (well order, which splits a series whose every dose was measured
        several times into one series per block). The ``sample`` and ``drug`` values are
        opaque labels, so native identifiers are fine - they become the
        ``obs_names``/``var_names`` of the result.
    :param replicate_fits: Whether to also fit every series on its own and store the
        replicate-fits table in ``uns["replicate_fits"]``.
    :param min_doses: Minimum number of distinct doses for a series to be fitted at all;
        shorter series are left out of the pooled fit and listed as ``too_few_doses``.
    :param metadata: Further columns of *df* to keep per series in the replicate-fits table,
        lowercased. Values that differ within one series are ``;``-joined.
    :param normalize: Whether to apply median-centric normalization before fitting. Factors are
        computed once per dose-range group, not per chunk.
    :param fit_type: Fitting method. Only "OLS" is currently supported.
    :param fit_speed: Fitting thoroughness, one of :data:`FIT_SPEEDS`.
    :param max_chunk_size: Maximum number of curves in one chunk, the unit of parallel work.
        The result does not depend on this value.
    :param max_workers: Number of local processes fitting the chunks.

    :returns: AnnData of shape (n_samples, n_drugs), indexed by the labels that were
        fitted. ``X`` holds pEC50; every other curve metric is a layer - the derived
        ``EC50``/``IC50``/``LN_IC50``, the goodness-of-fit columns, and the
        per-parameter standard errors ``pec50_error``/``slope_error``/
        ``front_error``/``back_error``. No quality filtering is applied. With
        *replicate_fits*, ``uns["replicate_fits"]`` holds one row per series.
    """
    from drevalpy.curation._plan import assemble, run_chunks

    work = _plan_curation(
        df,
        replicate_fits=replicate_fits,
        min_doses=min_doses,
        metadata=metadata,
        normalize=normalize,
        fit_type=fit_type,
        fit_speed=fit_speed,
        max_chunk_size=max_chunk_size,
    )
    return assemble(work, run_chunks(work.chunks, max_workers))


def _plan_curation(
    df: pd.DataFrame,
    *,
    replicate_fits: bool = False,
    min_doses: int = 4,
    metadata: Sequence[str] = (),
    normalize: bool = False,
    fit_type: str = "OLS",
    fit_speed: str = DEFAULT_FIT_SPEED,
    max_chunk_size: int = 2000,
) -> Plan:
    """Validate a well table and plan its curation: the first of :func:`curate`'s three steps."""
    from drevalpy.curation._fit import FitOptions
    from drevalpy.curation._plan import plan

    _validate_fit_options(fit_type=fit_type, fit_speed=fit_speed)
    _validate_columns(df, metadata)
    options = FitOptions(normalize=normalize, fit_type=fit_type, fit_speed=fit_speed, max_chunk_size=max_chunk_size)
    return plan(df, replicate_fits=replicate_fits, min_doses=min_doses, metadata=metadata, options=options)


def _validate_columns(df: pd.DataFrame, metadata: Sequence[str]) -> None:
    """Reject a table curate cannot read before any work happens.

    :param df: The well-level input table.
    :param metadata: The requested metadata columns.
    :raises InvalidInputError: If a required or metadata column is missing, a column is neither
        known nor metadata, or two metadata columns would get the same name.
    """
    from drevalpy.curation._series import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, SERIES_COLUMNS

    missing = [column for column in [*REQUIRED_COLUMNS, *metadata] if column not in df.columns]
    if missing:
        raise InvalidInputError(f"Missing column(s): {missing}")
    known = {*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS}
    clashing = [column for column in metadata if column in known or column.lower() in SERIES_COLUMNS]
    if clashing:
        raise InvalidInputError(f"Metadata column(s) {clashing} clash with the columns curate reads or writes.")
    lowered = [column.lower() for column in metadata]
    if len(set(lowered)) != len(lowered):
        raise InvalidInputError(f"Metadata columns {list(metadata)} are not distinct once lowercased.")
    unexpected = [column for column in df.columns if column not in known and column not in metadata]
    if unexpected:
        raise InvalidInputError(
            f"Unexpected column(s) {unexpected}; curate reads {sorted(known)}. Pass the ones to keep per series "
            "as metadata=..., or drop them."
        )


def _validate_fit_options(*, fit_type: str, fit_speed: str) -> None:
    """Reject unusable fit options before any preprocessing happens.

    :param fit_type: Requested fitting method.
    :param fit_speed: Requested fitting thoroughness.
    :raises InvalidInputError: If either option is not supported.
    """
    if fit_type not in SUPPORTED_FIT_TYPES:
        raise InvalidInputError(
            f"fit_type={fit_type!r} is not supported; expected one of {list(SUPPORTED_FIT_TYPES)}. "
            "MLE fitting is unavailable because the pinned curve_curator release rejects the "
            "'weights' argument that curve fitting passes to _Model.fit_mle()."
        )
    if fit_speed not in FIT_SPEEDS:
        raise InvalidInputError(f"fit_speed={fit_speed!r} is not supported; expected one of {list(FIT_SPEEDS)}.")
