"""Lightning-free omics helpers for the MOLIR/SuperFELTR predictors.

Split out of :mod:`drevalpy.components.predictors.literature.molir.utils` so the
registered ``predictor.py`` modules can reach the column-alignment helper without
importing ``pytorch_lightning`` at module scope. ``utils.py`` re-exports the
symbol, so the historical import path keeps working.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _realign_omic_matrix(
    values: np.ndarray,
    model_features: Sequence[str] | np.ndarray,
    meta_feature_names: Sequence[str] | np.ndarray,
) -> np.ndarray:
    """Align prediction-time omics columns to the feature order stored on the trained model.

    :param values: Omic feature matrix in incoming column order.
    :param model_features: Feature names used by the trained model.
    :param meta_feature_names: Feature names available in the incoming data.

    :returns: Matrix with columns reordered to match *model_features*.
    """
    if values.shape[1] == len(model_features):
        return values
    realigned = np.zeros((values.shape[0], len(model_features)))
    lookup_table = {feature: i for i, feature in enumerate(meta_feature_names)}
    for column, feature in enumerate(model_features):
        if feature in lookup_table:
            realigned[:, column] = values[:, lookup_table[feature]]
    return realigned
