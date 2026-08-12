"""Tests for drevalpy.curation._anndata.build_anndata."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drevalpy.curation._anndata import build_anndata


class TestBuildAnndata:
    """Tests for drevalpy.curation._anndata.build_anndata."""

    def _make_metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cell_line": ["CL_A", "CL_A", "CL_B", "CL_B"],
                "drug": ["DrugX", "DrugY", "DrugX", "DrugY"],
                "pEC50": [6.0, 4.0, 5.5, 3.5],
                "EC50": [1.0, 100.0, 3.16, 316.0],
                "IC50": [0.5, 200.0, 1.5, 500.0],
                "LN_IC50": [-0.69, 5.3, 0.4, 6.2],
                "AUC": [0.3, 0.9, 0.4, 0.95],
                "fold_change": [0.9, 0.05, 0.8, 0.03],
                "slope": [1.5, 0.8, 1.2, 0.5],
                "front": [1.0, 1.0, 0.95, 1.0],
                "back": [0.1, 0.9, 0.15, 0.92],
                "R2": [0.99, 0.5, 0.98, 0.4],
                "RMSE": [0.02, 0.05, 0.03, 0.06],
                "p_value": [0.001, 0.5, 0.01, 0.6],
                "log_p_value": [-3.0, -0.3, -2.0, -0.2],
                "f_value": [50.0, 2.0, 30.0, 1.5],
                "f_value_sam": [45.0, 1.5, 28.0, 1.2],
                "relevance_score": [0.95, 0.1, 0.85, 0.05],
                "signal_quality": [0.9, 0.3, 0.85, 0.25],
                "regulation": ["down", "not", "down", "not"],
            }
        )

    def test_shape(self) -> None:
        df = self._make_metrics_df()
        adata = build_anndata(df)

        assert adata.shape == (2, 2)  # 2 cell lines x 2 drugs

    def test_x_is_pec50(self) -> None:
        df = self._make_metrics_df()
        adata = build_anndata(df)

        assert adata.X.dtype == np.float32
        assert not np.all(np.isnan(adata.X))

    def test_layers_present(self) -> None:
        df = self._make_metrics_df()
        adata = build_anndata(df)

        assert "EC50" in adata.layers
        assert "IC50" in adata.layers
        assert "AUC" in adata.layers
        assert "slope" in adata.layers
        assert "regulation" in adata.layers

    def test_obs_var_indices(self) -> None:
        df = self._make_metrics_df()
        adata = build_anndata(df)

        assert list(adata.obs_names) == ["CL_A", "CL_B"]
        assert list(adata.var_names) == ["DrugX", "DrugY"]

    def test_regulation_encoded(self) -> None:
        df = self._make_metrics_df()
        adata = build_anndata(df)

        reg_layer = adata.layers["regulation"]
        assert -1.0 in reg_layer
        assert 0.0 in reg_layer
