"""Tests for pair-index helpers."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.core.batch.pair_features import pair_cell_line_indices, pair_drug_indices


def test_pair_cell_line_indices_maps_ids() -> None:
    indices = pair_cell_line_indices(
        np.array(["cl2", "cl1", "cl2"]),
        {"cl1": 0, "cl2": 1},
    )
    assert indices.tolist() == [1, 0, 1]


def test_pair_drug_indices_raises_for_missing_ids() -> None:
    with pytest.raises(ValueError, match="Missing drug identifiers"):
        pair_drug_indices(np.array(["d1", "missing"]), {"d1": 0})
