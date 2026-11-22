"""Dose-response curve fitting via curve_curator, returning AnnData.

:func:`curate` is the single entry point. It groups the well-level measurements into numbered
dose series (:mod:`drevalpy.curation._series`), fits one pooled curve per pair
(:mod:`drevalpy.curation._fit`), optionally fits every series on its own
(:mod:`drevalpy.curation._replicate_fits`), and pivots the metrics into an
:class:`~anndata.AnnData`. The fits run in chunks, here in local processes and, through
``drevalpy curate``, as separate jobs (:mod:`drevalpy.curation._plan`).

Fitting is identifier-agnostic - CurveCurator groups by dose range and treats
``sample``/``drug`` purely as labels - and :func:`curate` keys the returned
``obs_names``/``var_names`` from whatever those two columns held. A pipeline can
therefore curate on *native* identifiers, persist the ``.h5ad``, and resolve
identity (Cellosaurus accessions, PubChem CIDs, duplicate collapse) in a later,
cheap stage that only renames indices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.curation._anndata import build_anndata
from drevalpy.curation._fit import DEFAULT_FIT_SPEED, FIT_SPEEDS, SUPPORTED_FIT_TYPES, FitOptions, Source, Status
from drevalpy.curation._replicate_fits import METRICS
from drevalpy.curation._series import DEFAULT_MIN_DOSES

if TYPE_CHECKING:
    from collections.abc import Sequence

    import anndata
    import pandas as pd


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
    min_doses: int = DEFAULT_MIN_DOSES,
    metadata: Sequence[str] = (),
    normalize: bool = False,
    fit_type: str = FitOptions.fit_type,
    fit_speed: str = FitOptions.fit_speed,
    max_chunk_size: int = FitOptions.max_chunk_size,
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
    :param fit_type: Fitting method, one of :data:`SUPPORTED_FIT_TYPES`.
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
    from drevalpy.curation._plan import assemble, plan, run_chunks

    options = FitOptions(normalize=normalize, fit_type=fit_type, fit_speed=fit_speed, max_chunk_size=max_chunk_size)
    work = plan(df, replicate_fits=replicate_fits, min_doses=min_doses, metadata=metadata, options=options)
    return assemble(work, run_chunks(work.chunks, max_workers))
