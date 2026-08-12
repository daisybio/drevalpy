"""End-to-end tests for the drevalpy.curation package surface.

The per-stage tests live in ``test_preprocess.py`` / ``test_fit.py`` /
``test_postprocess.py`` / ``test_anndata.py``; only ``curate`` itself, which
wires the four stages together, is exercised here.
"""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd

from drevalpy.curation import curate


class TestEndToEnd:
    """Integration test for drevalpy.curation.curate."""

    def test_curate_returns_anndata(self, dose_response_df: pd.DataFrame) -> None:
        adata = curate(dose_response_df, cores=1, fit_speed="fast")

        assert isinstance(adata, anndata.AnnData)
        assert adata.shape == (3, 2)

    def test_curate_x_not_all_nan(self, dose_response_df: pd.DataFrame) -> None:
        adata = curate(dose_response_df, cores=1, fit_speed="fast")

        assert not np.all(np.isnan(adata.X))

    def test_curate_layers_exist(self, dose_response_df: pd.DataFrame) -> None:
        adata = curate(dose_response_df, cores=1, fit_speed="fast")

        assert len(adata.layers) > 0
        assert "EC50" in adata.layers
        assert "AUC" in adata.layers
