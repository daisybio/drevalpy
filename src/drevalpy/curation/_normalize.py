"""Group-wide median-centric normalization, applied before chunking.

curve_curator normalizes inside ``quantification.run_pipeline``, and its factors
are column medians over *the rows of the frame it was handed*
(``quantification.normalize_values``). Because :mod:`drevalpy.curation._fit`
calls ``run_pipeline`` once per parallel chunk, a normalized dataset used to get
one independent set of factors per chunk, which made its output depend on the
core count.

This module hoists that step out: the factors are computed once over a whole
dose-range group, the normalized intensities are written back into the ``Raw``
columns, and each chunk then runs with ``Processing.normalization`` disabled so
``run_pipeline`` derives its ratios from values that are already normalized.
Everything downstream of the ratios is row-wise, so the result no longer depends
on how the group is split.

``Signal Quality`` is the one column that would otherwise change meaning:
``run_pipeline`` derives it from the *raw* control intensities, which we have
overwritten. It is therefore computed here, carried on the frame under
:data:`PRE_NORM_SIGNAL_QUALITY`, and restored after the fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.log import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

#: Column holding ``Signal Quality`` as computed from the pre-normalization raw
#: controls. Prefixed so it cannot collide with a curve_curator output column.
PRE_NORM_SIGNAL_QUALITY = "drevalpy Pre-Norm Signal Quality"

_SIGNAL_QUALITY = "Signal Quality"


def _column_names(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(raw, normalized, dosed raw, control raw)`` column names.

    Mirrors the names ``quantification.run_pipeline`` builds from the same
    config, so the normalization done here lands on exactly the columns it would
    have used.

    :param config: A curve_curator config dict.
    :returns: Raw, normalized, dosed-raw and control-raw column name arrays.
    """
    from curve_curator import toolbox

    experiments = np.array(config["Experiment"]["experiments"])
    control_experiments = np.array(config["Experiment"]["control_experiment"])
    doses = np.array(config["Experiment"]["doses"], dtype=float)
    dosed_mask = doses != 0.0

    raw = toolbox.build_col_names("Raw {}", experiments)
    normalized = toolbox.build_col_names("Normalized {}", experiments)
    return raw, normalized, raw[dosed_mask], toolbox.build_col_names("Raw {}", control_experiments)


def normalize_group(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Normalize a whole dose-range group in one pass.

    The returned frame is ready to be chunked and fitted with
    ``Processing.normalization`` disabled.

    :param df: Wide-form group frame from :func:`drevalpy.curation._preprocess.preprocess`.
    :param config: The curve_curator config the group will be fitted with.
    :returns: A copy whose ``Raw`` columns hold normalized intensities and which
        carries :data:`PRE_NORM_SIGNAL_QUALITY`.
    """
    from curve_curator import quantification

    raw, normalized, dosed, controls = _column_names(config)

    work = quantification.filter_nans(df, dosed, config["Processing"]["max_missing"])
    signal_quality = np.log2(work[controls].mean(axis=1))

    work, factors = quantification.normalize_values(work, raw, normalized)
    work[raw] = work[normalized].to_numpy()
    work = work.drop(columns=list(normalized))
    work[PRE_NORM_SIGNAL_QUALITY] = signal_quality

    logger.info(
        "Normalization factors for a group of %d curves: %s",
        len(work),
        factors.round(4).to_dict(),
    )
    return work


def restore_signal_quality(fitted_df: pd.DataFrame) -> pd.DataFrame:
    """Put the pre-normalization ``Signal Quality`` back and drop the carrier.

    A no-op on frames that never went through :func:`normalize_group`.

    :param fitted_df: A frame returned by ``quantification.run_pipeline``.
    :returns: The same frame with ``Signal Quality`` measured on raw controls.
    """
    if PRE_NORM_SIGNAL_QUALITY not in fitted_df.columns:
        return fitted_df
    fitted_df[_SIGNAL_QUALITY] = fitted_df[PRE_NORM_SIGNAL_QUALITY]
    return fitted_df.drop(columns=[PRE_NORM_SIGNAL_QUALITY])
