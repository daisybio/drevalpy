"""Shared utilities for accessing SMILES strings from a FeatureSource."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drevalpy.components.core.features.feature_source import FeatureSource


def get_smiles_for_entities(source: FeatureSource, entity_ids: np.ndarray) -> pd.Series | None:
    """Get canonical SMILES indexed by entity_ids from the dataset.

    :param source: Feature source backed by a MuData object.
    :param entity_ids: Drug identifiers to retrieve SMILES for.
    :returns: Series of SMILES strings indexed by entity_ids, or None if unavailable.
    """
    mdata = getattr(source, "mdata", None)
    if mdata is None:
        return None
    response = mdata.mod["response"]
    if "canonical_smiles" not in response.var.columns:
        return None
    smiles = response.var["canonical_smiles"]
    return smiles.reindex(entity_ids)
