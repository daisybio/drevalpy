"""End-to-end tests for the drevalpy.curation package surface.

The per-stage tests live in ``test_preprocess.py`` / ``test_fit.py`` /
``test_postprocess.py`` / ``test_anndata.py`` / ``test_normalize.py``; only
:func:`~drevalpy.curation.curate`, which wires those stages together, is
exercised here.
"""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest

import drevalpy.curation as curation
from drevalpy.curation import (
    DEFAULT_FIT_SPEED,
    FIT_SPEEDS,
    SUPPORTED_FIT_TYPES,
    build_anndata,
    curate,
)
from tests.curation.conftest import build_dose_response_df


class TestPackageSurface:
    """``curate`` is the only entry point, and ``__all__`` says so."""

    def test_curate_and_build_anndata_are_exported(self) -> None:
        assert {"build_anndata", "curate"} <= set(curation.__all__)

    def test_no_second_entry_point_is_advertised(self) -> None:
        """``fit_curves`` was removed: the .h5ad is the pipeline's intermediate."""
        assert "fit_curves" not in curation.__all__
        assert not hasattr(curation, "fit_curves")


class TestFitOptionValidation:
    """``curate`` rejects options it cannot actually run, before fitting."""

    def test_only_ols_is_advertised(self) -> None:
        assert SUPPORTED_FIT_TYPES == ("OLS",)

    def test_the_documented_default_speed_is_exhaustive(self) -> None:
        """``fast`` takes one shot from a single guess, so it is not the default."""
        assert DEFAULT_FIT_SPEED == "exhaustive"
        assert DEFAULT_FIT_SPEED in FIT_SPEEDS

    def test_mle_is_rejected_before_any_fitting(self, dose_response_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="fit_mle"):
            curate(dose_response_df, max_workers=1, fit_type="MLE", fit_speed="fast")

    def test_unknown_fit_type_names_the_supported_ones(self, dose_response_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match=r"expected one of \['OLS'\]"):
            curate(dose_response_df, max_workers=1, fit_type="nonsense", fit_speed="fast")

    def test_unknown_fit_speed_is_rejected(self, dose_response_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="fit_speed='turbo'"):
            curate(dose_response_df, max_workers=1, fit_speed="turbo")


class TestCurate:
    """The one public entry point, end to end.

    Extended tier: shares the session-scoped ``curated_adata`` fixture, which is
    six real CurveCurator fits.
    """

    pytestmark = pytest.mark.slow

    def test_curate_returns_anndata(self, curated_adata: anndata.AnnData) -> None:
        assert isinstance(curated_adata, anndata.AnnData)
        assert curated_adata.shape == (3, 2)

    def test_the_labels_become_the_index(self, curated_adata: anndata.AnnData) -> None:
        assert curated_adata.obs_names.tolist() == ["CL_A", "CL_B", "CL_C"]
        assert curated_adata.var_names.tolist() == ["DrugX", "DrugY"]

    def test_curate_x_not_all_nan(self, curated_adata: anndata.AnnData) -> None:
        assert not np.all(np.isnan(curated_adata.X))

    def test_x_is_finite_for_the_sigmoid_drug(self, curated_adata: anndata.AnnData) -> None:
        sigmoid = curated_adata[:, "DrugX"].X

        assert np.isfinite(sigmoid).all()

    def test_curate_layers_exist(self, curated_adata: anndata.AnnData) -> None:
        assert len(curated_adata.layers) > 0
        assert "EC50" in curated_adata.layers
        assert "AUC" in curated_adata.layers

    @pytest.mark.parametrize("metric", ["EC50", "IC50", "LN_IC50", "AUC", "R2", "regulation"])
    def test_the_derived_and_quality_metrics_reach_the_caller(
        self, curated_adata: anndata.AnnData, metric: str
    ) -> None:
        assert metric in curated_adata.layers

    def test_the_recovered_error_layers_are_present(self, curated_adata: anndata.AnnData) -> None:
        assert {"pec50_error", "slope_error", "front_error", "back_error"} <= set(curated_adata.layers)

    def test_every_layer_is_shaped_like_x(self, curated_adata: anndata.AnnData) -> None:
        for name, layer in curated_adata.layers.items():
            assert layer.shape == curated_adata.shape, name


class TestNativeIdentifiersSurvive:
    """The reason no flat metrics frame is needed.

    The curation pipeline curates on native identifiers, persists the ``.h5ad``,
    and remaps ``obs_names``/``var_names`` in a later, cheap stage - so the labels
    it was given have to come back verbatim as the index.

    Extended tier: one real six-curve CurveCurator fit.
    """

    pytestmark = pytest.mark.slow

    @pytest.fixture(scope="class")
    def natively_curated(self) -> anndata.AnnData:
        """``curate`` over the same data relabelled with native identifiers."""
        native = build_dose_response_df()
        native["cell_line"] = native["cell_line"].map({"CL_A": "ACH-1", "CL_B": "SIDM00400", "CL_C": "2004"})
        native["drug"] = native["drug"].map({"DrugX": "DRUG_1047", "DrugY": "BRD-K02251932"})
        return curate(native, max_workers=1, fit_speed="fast")

    def test_the_cell_line_labels_are_the_obs_index(self, natively_curated: anndata.AnnData) -> None:
        assert set(natively_curated.obs_names) == {"ACH-1", "SIDM00400", "2004"}

    def test_the_drug_labels_are_the_var_index(self, natively_curated: anndata.AnnData) -> None:
        assert set(natively_curated.var_names) == {"DRUG_1047", "BRD-K02251932"}

    def test_a_purely_numeric_label_stays_a_string(self, natively_curated: anndata.AnnData) -> None:
        """``2004`` is a valid Cellosaurus-adjacent ID; it must not become an int."""
        assert "2004" in natively_curated.obs_names

    def test_the_values_are_the_same_as_under_the_original_labels(
        self, natively_curated: anndata.AnnData, curated_adata: anndata.AnnData
    ) -> None:
        """Relabelling must not change a single fit - the labels are opaque."""
        renamed = natively_curated[["ACH-1", "SIDM00400", "2004"], ["DRUG_1047", "BRD-K02251932"]]

        np.testing.assert_array_equal(renamed.X, curated_adata.X)


class TestCurateIsPreprocessFitPostprocessBuild:
    """``curate`` must stay exactly the composition of the four private stages.

    Extended tier: shares the session-scoped fitted fixture.
    """

    pytestmark = pytest.mark.slow

    @staticmethod
    def _stages(df: pd.DataFrame) -> anndata.AnnData:
        """Run the private stages by hand, as ``curate`` is documented to."""
        from drevalpy.curation._fit import fit_groups
        from drevalpy.curation._postprocess import postprocess
        from drevalpy.curation._preprocess import preprocess

        return build_anndata(postprocess(fit_groups(preprocess(df), max_workers=1, fit_speed="fast")))

    def test_running_the_stages_by_hand_reproduces_curate(
        self, curated_adata: anndata.AnnData, dose_response_df: pd.DataFrame
    ) -> None:
        composed = self._stages(dose_response_df)

        np.testing.assert_array_equal(composed.X, curated_adata.X)
        assert set(composed.layers) == set(curated_adata.layers)

    def test_the_composed_layers_agree_value_for_value(
        self, curated_adata: anndata.AnnData, dose_response_df: pd.DataFrame
    ) -> None:
        composed = self._stages(dose_response_df)

        for name in curated_adata.layers:
            np.testing.assert_array_equal(composed.layers[name], curated_adata.layers[name], err_msg=name)
