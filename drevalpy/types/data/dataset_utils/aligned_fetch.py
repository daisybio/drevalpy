"""Generic aligned-fetch utility for retrieving rows by ID."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drevalpy.log import get_logger

logger = get_logger(__name__)


def _aligned_fetch(
    index: pd.Index,
    ids: np.ndarray,
    data: np.ndarray,
    *,
    strict: bool,
    entity_label: str,
) -> np.ndarray:
    """Fetch rows from *data* aligned to *ids* using *index*, filling NaN for missing.

    Args:
        index: pd.Index mapping entity names to row positions in *data*.
        ids: 1-D array of requested entity IDs.
        data: 2-D source array to fetch rows from.
        strict: If True, raise KeyError for missing IDs instead of warning.
        entity_label: Human-readable label for error messages (e.g. "cell line").

    Returns:
        Float32 array of shape (len(ids), data.shape[1]).
    """
    positions = index.get_indexer(ids)
    missing_mask = positions == -1
    if missing_mask.any():
        n_missing = int(missing_mask.sum())
        sample = ids[missing_mask][:5].tolist()
        msg = f"{n_missing} of {len(ids)} {entity_label} IDs not found (first few: {sample}). Returning NaN rows."
        if strict:
            raise KeyError(msg)
        logger.warning(msg)

    n_features = data.shape[1]
    result = np.full((len(ids), n_features), np.nan, dtype=np.float32)
    valid = positions >= 0
    result[valid] = np.asarray(data[positions[valid]], dtype=np.float32)
    return result
