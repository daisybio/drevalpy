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


@dataclass(frozen=True)
class FitOptions:
    """How every curve is fitted; the same for the pooled and the replicate fits.

    :param normalize: Whether to median-normalize each dose-range group before fitting.
    :param fit_type: Fitting method, one of :data:`drevalpy.curation.SUPPORTED_FIT_TYPES`.
    :param fit_speed: Fitting thoroughness, one of :data:`drevalpy.curation.FIT_SPEEDS`.
    :param max_chunk_size: Maximum number of curves per chunk.
    """

    normalize: bool = False
    fit_type: str = "OLS"
    fit_speed: str = "exhaustive"
    max_chunk_size: int = 2000


def _build_config(
    n_experiments: int,
    doses: list[float],
    n_replicates: int,
    normalize: bool = False,
    fit_type: str = "OLS",
    fit_speed: str = "exhaustive",
) -> dict:
    """Build a curve_curator config dict equivalent to a parsed TOML."""
    config = {
        "__file__": {"Path": "/tmp/dummy.toml"},  # noqa: S108
        "Meta": {
            "id": "drevalpy",
            "description": "drevalpy curation",
            "condition": "",
            "treatment_time": "72 h",
        },
        "Experiment": {
            "experiments": list(range(n_experiments)),
            "doses": doses,
            "dose_scale": "1e-06",
            "dose_unit": "uM",
            "control_experiment": list(range(n_replicates)),
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
            "type": fit_type,
            "speed": fit_speed,
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

    from curve_curator.toml_parser import set_default_values

    config = set_default_values(config)
    return config


def _build_work_items(
    groups: list[tuple[pd.DataFrame, dict]],
    max_chunk_size: int,
    normalize: bool,
    fit_type: str,
    fit_speed: str,
) -> tuple[list[tuple[pd.DataFrame, dict, int]], list[dict]]:
    """Split every group into chunks and build the matching curve_curator configs.

    Each group is split into chunks of at most *max_chunk_size* curves.
    Small groups produce a single chunk; large groups are split into as many
    chunks as needed to stay at or below the cap.

    When *normalize* is set, the group is normalized here - once, over all of its
    rows - and the config handed to each chunk has normalization switched off, so
    curve_curator cannot recompute per-chunk factors. See
    :mod:`drevalpy.curation._normalize`.

    :param groups: (wide_df, group_info) tuples from preprocess.
    :param max_chunk_size: Maximum number of curves per chunk.
    :param normalize: Whether to apply median-centric normalization.
    :param fit_type: Fitting method. Only "OLS" is currently supported.
    :param fit_speed: Fitting thoroughness.
    :returns: (work items as (chunk_df, config, group_idx), one config per group).
    """
    from drevalpy.curation._normalize import normalize_group

    work_items: list[tuple[pd.DataFrame, dict, int]] = []
    configs: list[dict] = []

    for group_idx, (df, group_info) in enumerate(groups):
        config = _build_config(
            n_experiments=group_info["n_experiments"],
            doses=group_info["doses"],
            n_replicates=group_info["n_replicates"],
            normalize=normalize,
            fit_type=fit_type,
            fit_speed=fit_speed,
        )
        configs.append(config)

        chunk_df_source, chunk_config = df, config
        if normalize:
            chunk_df_source = normalize_group(df, config)
            chunk_config = copy.deepcopy(config)
            chunk_config["Processing"]["normalization"] = False

        n_chunks = max(1, math.ceil(len(chunk_df_source) / max_chunk_size))
        for chunk_df in np.array_split(chunk_df_source, n_chunks):
            work_items.append((chunk_df.reset_index(drop=True), chunk_config, group_idx))

    return work_items, configs


def fit_chunk(item: tuple[pd.DataFrame, dict]) -> pd.DataFrame:
    """Fit one chunk of curves with CurveCurator, single-core.

    This is the function :func:`~drevalpy.curation.compute.map_chunks` sends to a backend, so
    it stays a picklable module-level function. CurveCurator ends a run with a bare ``exit()``
    (``SystemExit(None)``) when missing values leave no curve to fit; that is a legitimate
    empty result.

    :param item: ``(chunk_df, config)`` as built by :func:`_build_work_items`.
    :returns: The fitted curves, or an empty frame if CurveCurator dropped every curve.
    :raises RuntimeError: If CurveCurator exits with a code other than ``None``.
    """
    from curve_curator import quantification

    from drevalpy.curation._normalize import restore_signal_quality

    chunk_df, config = item
    cfg = copy.deepcopy(config)
    cfg["Processing"]["available_cores"] = 1
    start = time.time()
    try:
        result = quantification.run_pipeline(chunk_df, cfg)
    except SystemExit as exc:
        if exc.code is None:
            logger.warning("CurveCurator dropped all %d curves of a chunk (missing values)", len(chunk_df))
            return pd.DataFrame()
        raise RuntimeError(f"CurveCurator exited with code {exc.code!r}") from exc
    logger.debug("Fitted %d curves in %.1fs", len(chunk_df), time.time() - start)
    return restore_signal_quality(result)


#: What a group is: ``n_experiments``, ``doses`` and ``n_replicates``, as :func:`preprocess` reports.
GroupInfo = dict


def chunk_config(info: GroupInfo, options: FitOptions) -> dict:
    """Return the CurveCurator config a chunk of the group described by *info* is fitted with.

    Normalization, if requested, already happened over the whole group in :func:`split`, so
    the chunk config never normalizes again.

    :param info: The group's description.
    :param options: How the curves are fitted.
    :returns: The config.
    """
    return _build_config(
        int(info["n_experiments"]),
        list(info["doses"]),
        int(info["n_replicates"]),
        normalize=False,
        fit_type=options.fit_type,
        fit_speed=options.fit_speed,
    )


def group_config(info: GroupInfo, options: FitOptions) -> dict:
    """Return the CurveCurator config the group described by *info* is thresholded and reported with.

    :param info: The group's description.
    :param options: How the curves are fitted.
    :returns: The config.
    """
    return _build_config(
        int(info["n_experiments"]),
        list(info["doses"]),
        int(info["n_replicates"]),
        normalize=options.normalize,
        fit_type=options.fit_type,
        fit_speed=options.fit_speed,
    )


def split(viability: pd.DataFrame, options: FitOptions) -> tuple[list[tuple[pd.DataFrame, int]], list[GroupInfo]]:
    """Split a viability table into the chunks that are fitted independently.

    :param viability: Long table with ``sample``, ``drug``, ``concentration``, ``intensity`` and
        optionally an integer ``replicate`` per pair.
    :param options: How the curves are fitted.
    :returns: The chunks as ``(chunk_df, group index)`` and one description per group; nothing
        for an empty table.
    """
    from drevalpy.curation._preprocess import preprocess

    if viability.empty:
        return [], []
    groups = preprocess(viability)
    work_items, _ = _build_work_items(
        groups,
        max_chunk_size=options.max_chunk_size,
        normalize=options.normalize,
        fit_type=options.fit_type,
        fit_speed=options.fit_speed,
    )
    infos = [
        {
            "n_experiments": int(info["n_experiments"]),
            "doses": [float(dose) for dose in info["doses"]],
            "n_replicates": int(info["n_replicates"]),
        }
        for _, info in groups
    ]
    return [(chunk_df, group_idx) for chunk_df, _, group_idx in work_items], infos


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
    work_items = [(chunk_df, chunk_config(infos[group], options), group) for chunk_df, group in chunks]
    configs = [group_config(info, options) for info in infos]
    fitted_groups = _assemble_groups(work_items, list(fitted), configs)
    metrics = postprocess(fitted_groups) if fitted_groups else None
    return _with_status(metrics, work_items)


def fit_table(viability: pd.DataFrame, options: FitOptions) -> pd.DataFrame:
    """Split, fit and collect a viability table in this process.

    :param viability: As :func:`split`.
    :param options: How the curves are fitted.
    :returns: As :func:`collect`.
    """
    chunks, infos = split(viability, options)
    fitted = [fit_chunk((chunk_df, chunk_config(infos[group], options))) for chunk_df, group in chunks]
    return collect(chunks, fitted, infos, options)


def _assemble_groups(
    work_items: list[tuple[pd.DataFrame, dict, int]],
    fitted: list[pd.DataFrame | None],
    configs: list[dict],
) -> list[tuple[pd.DataFrame, dict]]:
    """Put the chunks of each group back together and apply its significance thresholds."""
    group_results: list[list[pd.DataFrame]] = [[] for _ in configs]
    for (_, _, group_idx), fitted_df in zip(work_items, fitted, strict=True):
        if fitted_df is not None and not fitted_df.empty:
            group_results[group_idx].append(fitted_df)

    fitted_groups: list[tuple[pd.DataFrame, dict]] = []
    for group_idx, config in enumerate(configs):
        if not group_results[group_idx]:
            logger.warning("Group %d: CurveCurator dropped every curve (missing values), skipping", group_idx)
            continue
        assembled = pd.concat(group_results[group_idx], ignore_index=True)
        fitted_groups.append((_thresholded(assembled, config, group_idx), config))
    return fitted_groups


def _thresholded(assembled: pd.DataFrame, config: dict, group_idx: int) -> pd.DataFrame:
    """Apply a group's significance thresholds, or NaN the SAM columns if it is too small for them."""
    from curve_curator import thresholding

    try:
        return thresholding.apply_significance_thresholds(assembled, config)
    except ValueError as exc:
        if "number of model parameters" not in str(exc):
            raise
    logger.warning(
        "Group %d: too few curves (%d) for significance thresholding, skipping SAM", group_idx, len(assembled)
    )
    for column in _SAM_COLUMNS:
        assembled[column] = np.nan
    return assembled


def _with_status(metrics: pd.DataFrame | None, work_items: list[tuple[pd.DataFrame, dict, int]]) -> pd.DataFrame:
    """Add a ``fit_failed`` row for every attempted curve missing from *metrics*."""
    attempted = pd.Index(pd.concat([chunk_df["Name"] for chunk_df, _, _ in work_items], ignore_index=True)).unique()
    labels = attempted.to_series().str.split("|", n=1, expand=True)
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
