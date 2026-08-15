"""Dose-response curve fitting via curve_curator, returning AnnData.

:func:`curate` is the single entry point. It preprocesses the long-form
measurements, runs the parallel CurveCurator fit, extracts the per-curve metrics
and pivots them into an :class:`~anndata.AnnData`.

Fitting is identifier-agnostic - CurveCurator groups by dose range and treats
``cell_line``/``drug`` purely as labels - and :func:`curate` keys the returned
``obs_names``/``var_names`` from whatever those two columns held. A pipeline can
therefore curate on *native* identifiers, persist the ``.h5ad``, and resolve
identity (Cellosaurus accessions, PubChem CIDs, duplicate collapse) in a later,
cheap stage that only renames indices. The ``.h5ad`` is that intermediate, so no
flat metrics frame needs to escape this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.curation._anndata import build_anndata

if TYPE_CHECKING:
    from concurrent.futures import Executor

    import anndata
    import pandas as pd

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

__all__ = ["DEFAULT_FIT_SPEED", "FIT_SPEEDS", "SUPPORTED_FIT_TYPES", "build_anndata", "curate"]


def curate(
    df: pd.DataFrame,
    *,
    cores: int = 4,
    normalize: bool = False,
    fit_type: str = "OLS",
    fit_speed: str = DEFAULT_FIT_SPEED,
    executor: Executor | None = None,
) -> anndata.AnnData:
    """Fit dose-response curves and return an AnnData of curve metrics.

    Parameters
    ----------
    df
        Long-form DataFrame with columns: drug, cell_line, concentration,
        intensity, and optionally replicate. The ``drug`` and ``cell_line``
        values are treated as opaque labels, so native identifiers are fine -
        they become the ``var_names``/``obs_names`` of the result.
    cores
        Number of CPU cores for parallel fitting. Controls both the chunk size
        and (when no *executor* is supplied) the local process pool width. The
        result does not depend on this, including when *normalize* is set.
    normalize
        Whether to apply median-centric normalization before fitting. Factors are
        computed once per dose-range group, not per parallel chunk.
    fit_type
        Fitting method. Only "OLS" is currently supported.
    fit_speed
        Fitting thoroughness, one of :data:`FIT_SPEEDS`.
    executor
        Optional :class:`~concurrent.futures.Executor` instance. When provided,
        chunk fitting is dispatched through this executor instead of an internal
        :class:`~concurrent.futures.ProcessPoolExecutor`. This enables callers to
        supply e.g. a ``submitit.AutoExecutor`` configured for SLURM, or any other
        ``concurrent.futures``-compatible executor. The caller retains ownership
        and is responsible for shutting down the executor.

    Returns:
    -------
    AnnData of shape (n_cell_lines, n_drugs), indexed by the labels that were
    fitted. ``X`` holds pEC50; every other curve metric is a layer - the derived
    ``EC50``/``IC50``/``LN_IC50``, the goodness-of-fit columns, and the
    per-parameter standard errors ``pec50_error``/``slope_error``/
    ``front_error``/``back_error``. No quality filtering is applied.

    Raises:
    ------
    ValueError
        If ``fit_type`` or ``fit_speed`` is not supported.
    """
    from drevalpy.curation._fit import fit_groups
    from drevalpy.curation._postprocess import postprocess
    from drevalpy.curation._preprocess import preprocess

    _validate_fit_options(fit_type=fit_type, fit_speed=fit_speed)

    groups = preprocess(df)
    fitted_groups = fit_groups(
        groups,
        cores=cores,
        normalize=normalize,
        fit_type=fit_type,
        fit_speed=fit_speed,
        executor=executor,
    )
    return build_anndata(postprocess(fitted_groups))


def _validate_fit_options(*, fit_type: str, fit_speed: str) -> None:
    """Reject unusable fit options before any preprocessing happens.

    :param fit_type: Requested fitting method.
    :param fit_speed: Requested fitting thoroughness.
    :raises ValueError: If either option is not supported.
    """
    if fit_type not in SUPPORTED_FIT_TYPES:
        raise ValueError(
            f"fit_type={fit_type!r} is not supported; expected one of {list(SUPPORTED_FIT_TYPES)}. "
            "MLE fitting is unavailable because the pinned curve_curator release rejects the "
            "'weights' argument that curve fitting passes to _Model.fit_mle()."
        )
    if fit_speed not in FIT_SPEEDS:
        raise ValueError(f"fit_speed={fit_speed!r} is not supported; expected one of {list(FIT_SPEEDS)}.")
