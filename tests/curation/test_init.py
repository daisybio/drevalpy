"""End-to-end tests for the drevalpy.curation package surface.

The per-stage tests live in ``test_preprocess.py`` / ``test_fit.py`` /
``test_postprocess.py`` / ``test_anndata.py``; only ``curate`` itself, which
wires the four stages together, is exercised here.
"""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest

from drevalpy.curation import SUPPORTED_FIT_TYPES, curate


class TestFitTypeValidation:
    """``curate`` rejects fit types it cannot actually run."""

    def test_only_ols_is_advertised(self) -> None:
        assert SUPPORTED_FIT_TYPES == ("OLS",)

    def test_mle_is_rejected_before_any_fitting(self, dose_response_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="fit_mle"):
            curate(dose_response_df, cores=1, fit_type="MLE", fit_speed="fast")

    def test_unknown_fit_type_names_the_supported_ones(self, dose_response_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match=r"expected one of \['OLS'\]"):
            curate(dose_response_df, cores=1, fit_type="nonsense", fit_speed="fast")


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
