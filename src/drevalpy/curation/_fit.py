"""Fit dose-response curves chunk by chunk, so the chunks can run anywhere.

A fit has three steps, which :mod:`drevalpy.curation._plan` runs in one process or hands
to a workflow manager one by one:

1. :func:`split` preprocesses a viability table into dose-range groups and splits each group
   into chunks of at most ``max_chunk_size`` curves (normalizing each group first, once over
   all of its curves, when asked to).
2. :func:`fit_chunk` fits one chunk with CurveCurator, single-core. A chunk carries only its
   curves and its group's description, so it can be fitted on another machine.
3. :func:`collect` puts the fitted chunks of each group back together, applies the group's
   significance thresholds and returns one row per attempted curve.

Every way CurveCurator can fail to fit is turned into a row rather than a crash or a silent
gap:

* a curve CurveCurator drops (too many missing values) gets ``status = "fit_failed"`` and
  NaN metrics;
* a chunk in which it drops every curve ends with a bare ``exit()`` inside CurveCurator, which
  :func:`fit_chunk` turns into an empty result;
* a group left without any curve is skipped;
* a group with too few curves for the SAM significance thresholding keeps its fits, with NaN
  in the three columns that thresholding writes.

Significance thresholding runs once per group over all of its chunks, so the result does not
depend on the chunk size or on where the chunks ran.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from drevalpy.curation._series import LABEL_SEPARATOR
from drevalpy.errors import InvalidInputError
from drevalpy.log import get_logger

logger = get_logger(__name__)

#: Column of :func:`collect`'s result telling whether a curve was fitted.
STATUS_COLUMN = "status"

#: Columns curve_curator's SAM thresholding writes; NaN when a group is too small for it.
_SAM_COLUMNS = ("Curve F_Value SAM Corrected", "Curve Relevance Score", "Curve Regulation")


class Status(StrEnum):
    """Whether a curve, or a replicate's series, was fitted."""

    FITTED = "fitted"
    FIT_FAILED = "fit_failed"
    TOO_FEW_DOSES = "too_few_doses"


class Source(StrEnum):
    """Which fit a chunk, or a replicate-fits row, comes from."""

    POOLED = "pooled"
    REPLICATE = "replicate"


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


@dataclass(frozen=True)
class FitOptions:
    """How every curve is fitted; the same for the pooled and the replicate fits.

    Unsupported options are rejected on construction, so plans and chunks read from disk are checked too.

    :param normalize: Whether to median-normalize each dose-range group before fitting.
    :param fit_type: Fitting method, one of :data:`SUPPORTED_FIT_TYPES`.
    :param fit_speed: Fitting thoroughness, one of :data:`FIT_SPEEDS`.
    :param max_chunk_size: Maximum number of curves per chunk.
    """

    normalize: bool = False
    fit_type: str = "OLS"
    fit_speed: str = DEFAULT_FIT_SPEED
    max_chunk_size: int = 2000

    def __post_init__(self) -> None:
        """Reject unusable fit options before any preprocessing happens.

        :raises InvalidInputError: If *fit_type* or *fit_speed* is not supported.
        """
        if self.fit_type not in SUPPORTED_FIT_TYPES:
            raise InvalidInputError(
                f"fit_type={self.fit_type!r} is not supported; expected one of {list(SUPPORTED_FIT_TYPES)}. "
                "MLE fitting is unavailable because the pinned curve_curator release rejects the "
                "'weights' argument that curve fitting passes to _Model.fit_mle()."
            )
        if self.fit_speed not in FIT_SPEEDS:
            raise InvalidInputError(
                f"fit_speed={self.fit_speed!r} is not supported; expected one of {list(FIT_SPEEDS)}."
            )


def fit_chunk(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fit one chunk of curves with CurveCurator, single-core.

    Stays a picklable module-level function, so :meth:`drevalpy.curation._plan.Chunk.run` can send
    it to worker processes. CurveCurator ends a run with a bare ``exit()`` (``SystemExit(None)``)
    when missing values leave no curve to fit; that is a legitimate empty result.

    :param frame: The chunk's curves, in CurveCurator's wide format.
    :param config: The chunk's :func:`fit_config`.
    :returns: The fitted curves, or an empty frame if CurveCurator dropped every curve.
    :raises RuntimeError: If CurveCurator exits with a code other than ``None``.
    """
    from curve_curator import quantification

    from drevalpy.curation._normalize import restore_signal_quality

    cfg = copy.deepcopy(config)
    cfg["Processing"]["available_cores"] = 1
    start = time.time()
    try:
        result = quantification.run_pipeline(frame, cfg)
    except SystemExit as exc:
        if exc.code is None:
            logger.warning("CurveCurator dropped all %d curves of a chunk (missing values)", len(frame))
            return pd.DataFrame()
        raise RuntimeError(f"CurveCurator exited with code {exc.code!r}") from exc
    logger.debug("Fitted %d curves in %.1fs", len(frame), time.time() - start)
    return restore_signal_quality(result)


#: What a group is: ``n_experiments``, ``doses`` and ``n_replicates``, as :func:`preprocess` reports.
GroupInfo = dict


def fit_config(info: GroupInfo, options: FitOptions, normalize: bool) -> dict:
    """Return the CurveCurator config for the group described by *info*, equivalent to a parsed TOML.

    A chunk is fitted with ``normalize=False``, because :func:`split` already normalized its group,
    once over all of its curves; the group's thresholds and report use ``options.normalize``. See
    :mod:`drevalpy.curation._normalize`.

    :param info: The group's description.
    :param options: How the curves are fitted.
    :param normalize: Whether CurveCurator normalizes the curves it is handed.
    :returns: The config.
    """
    from curve_curator.toml_parser import set_default_values

    doses = info["doses"]
    return set_default_values(
        {
            "__file__": {"Path": "/tmp/dummy.toml"},  # noqa: S108
            "Meta": {
                "id": "drevalpy",
                "description": "drevalpy curation",
                "condition": "",
                "treatment_time": "72 h",
            },
            "Experiment": {
                "experiments": list(range(info["n_experiments"])),
                "doses": doses,
                "dose_scale": "1e-06",
                "dose_unit": "uM",
                "control_experiment": list(range(info["n_replicates"])),
                "measurement_type": "OTHER",
                "data_type": "OTHER",
                "search_engine": "OTHER",
                "search_engine_version": "0",
            },
            "Paths": {
                "input_file": "/tmp/input.tsv",  # noqa: S108
                "curves_file": "/tmp/curves.tsv",  # noqa: S108
                "normalization_file": "/tmp/norm.txt",  # noqa: S108
                "mad_file": "/tmp/mad.txt",  # noqa: S108
                "dashboard": "/tmp/dashboard.html",  # noqa: S108
            },
            "Processing": {
                "available_cores": 1,
                "max_missing": max(len(doses) - 5, 0),
                "imputation": False,
                "normalization": normalize,
            },
            "Curve Fit": {
                "type": options.fit_type,
                "speed": options.fit_speed,
                "max_iterations": 1000,
                "interpolation": False,
                "control_fold_change": True,
            },
            "F Statistic": {
                "optimized_dofs": True,
                "alpha": 0.05,
                "fc_lim": 0.45,
            },
        }
    )


def split(viability: pd.DataFrame, options: FitOptions) -> tuple[list[tuple[pd.DataFrame, int]], list[GroupInfo]]:
    """Split a viability table into the chunks that are fitted independently.

    Each dose-range group becomes chunks of at most ``options.max_chunk_size`` curves. With
    ``options.normalize`` the group is normalized first, once over all of its curves.

    :param viability: Long table with ``sample``, ``drug``, ``concentration``, ``intensity`` and
        optionally an integer ``replicate`` per pair, one measurement per dose and replicate.
    :param options: How the curves are fitted.
    :returns: The chunks as ``(chunk_df, group index)`` and one description per group; nothing
        for an empty table.
    """
    from drevalpy.curation._normalize import normalize_group
    from drevalpy.curation._preprocess import preprocess

    if viability.empty:
        return [], []
    groups = preprocess(viability)
    chunks: list[tuple[pd.DataFrame, int]] = []
    for group, (frame, info) in enumerate(groups):
        if options.normalize:
            frame = normalize_group(frame, fit_config(info, options, normalize=True))
        n_chunks = max(1, math.ceil(len(frame) / options.max_chunk_size))
        chunks.extend((chunk.reset_index(drop=True), group) for chunk in np.array_split(frame, n_chunks))
    return chunks, [info for _, info in groups]


def collect(
    chunks: list[tuple[pd.DataFrame, int]],
    fitted: list[pd.DataFrame],
    infos: list[GroupInfo],
    options: FitOptions,
) -> pd.DataFrame:
    """Combine the fitted chunks into one row per attempted curve.

    :param chunks: :func:`split`'s chunks.
    :param fitted: :func:`fit_chunk`'s result for each chunk, in the same order.
    :param infos: :func:`split`'s group descriptions.
    :param options: How the curves were fitted.
    :returns: ``sample``, ``drug``, :data:`STATUS_COLUMN` and the metric columns of
        :func:`drevalpy.curation._postprocess.postprocess` (``regulation`` as text), NaN where
        the fit failed; only the first three columns when there were no chunks.
    """
    from drevalpy.curation._postprocess import postprocess

    if not chunks:
        return pd.DataFrame(columns=pd.Index(["sample", "drug", STATUS_COLUMN]))
    configs = [fit_config(info, options, options.normalize) for info in infos]
    frames = _assemble_groups(chunks, fitted, configs)
    metrics = postprocess(frames) if frames else None
    return _with_status(metrics, chunks)


def _assemble_groups(
    chunks: list[tuple[pd.DataFrame, int]], fitted: list[pd.DataFrame], configs: list[dict]
) -> list[pd.DataFrame]:
    """Put the chunks of each group back together and apply its significance thresholds."""
    group_results: list[list[pd.DataFrame]] = [[] for _ in configs]
    for (_, group), fitted_df in zip(chunks, fitted, strict=True):
        if not fitted_df.empty:
            group_results[group].append(fitted_df)

    frames: list[pd.DataFrame] = []
    for group, config in enumerate(configs):
        if not group_results[group]:
            logger.warning("Group %d: CurveCurator dropped every curve (missing values), skipping", group)
            continue
        frames.append(_thresholded(pd.concat(group_results[group], ignore_index=True), config, group))
    return frames


def _thresholded(assembled: pd.DataFrame, config: dict, group: int) -> pd.DataFrame:
    """Apply a group's significance thresholds, or NaN the SAM columns if it is too small for them."""
    from curve_curator import thresholding

    try:
        return thresholding.apply_significance_thresholds(assembled, config)
    except ValueError as exc:
        if "number of model parameters" not in str(exc):
            raise
    logger.warning("Group %d: too few curves (%d) for significance thresholding, skipping SAM", group, len(assembled))
    for column in _SAM_COLUMNS:
        assembled[column] = np.nan
    return assembled


def _with_status(metrics: pd.DataFrame | None, chunks: list[tuple[pd.DataFrame, int]]) -> pd.DataFrame:
    """Add a ``fit_failed`` row for every attempted curve missing from *metrics*."""
    attempted = pd.Index(pd.concat([frame["Name"] for frame, _ in chunks], ignore_index=True)).unique()
    labels = attempted.to_series().str.split(LABEL_SEPARATOR, n=1, expand=True)
    frame = pd.DataFrame({"sample": labels[0].to_numpy(), "drug": labels[1].to_numpy()})
    if metrics is None or metrics.empty:
        frame[STATUS_COLUMN] = Status.FIT_FAILED.value
        return frame
    metrics = metrics.drop_duplicates(subset=["sample", "drug"], keep="first")
    out = frame.merge(metrics, on=["sample", "drug"], how="left", indicator=True)
    out[STATUS_COLUMN] = np.where(out.pop("_merge") == "both", Status.FITTED.value, Status.FIT_FAILED.value)
    n_failed = int((out[STATUS_COLUMN] == Status.FIT_FAILED.value).sum())
    if n_failed:
        logger.info("%d of %d attempted curves were dropped by CurveCurator", n_failed, len(out))
    return out
