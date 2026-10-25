"""End-to-end tests for the drevalpy.curation package surface (``curate`` wires the private stages)."""

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
from drevalpy.errors import InvalidInputError
from tests.curation.conftest import build_dose_response_df


def test_curate_and_build_anndata_are_exported_and_fit_curves_is_gone() -> None:
    assert {"build_anndata", "curate"} <= set(curation.__all__)
    assert "fit_curves" not in curation.__all__
    assert not hasattr(curation, "fit_curves")


class TestFitOptionValidation:
    def test_only_ols_and_the_exhaustive_default_speed_are_advertised(self) -> None:
        assert SUPPORTED_FIT_TYPES == ("OLS",)
        assert DEFAULT_FIT_SPEED == "exhaustive"
        assert DEFAULT_FIT_SPEED in FIT_SPEEDS

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"fit_type": "MLE"}, "fit_mle"),
            ({"fit_type": "nonsense"}, r"expected one of \['OLS'\]"),
            ({"fit_speed": "turbo"}, "fit_speed='turbo'"),
        ],
    )
    def test_unsupported_options_are_rejected_before_any_fitting(
        self, dose_response_df: pd.DataFrame, kwargs: dict, match: str
    ) -> None:
        with pytest.raises(InvalidInputError, match=match):
            curate(dose_response_df, max_workers=1, **{"fit_speed": "fast", **kwargs})


@pytest.mark.slow
class TestCurate:
    def test_returns_samples_by_drugs_anndata_with_finite_sigmoid_values(self, curated_adata: anndata.AnnData) -> None:
        assert isinstance(curated_adata, anndata.AnnData)
        assert curated_adata.shape == (3, 2)
        assert curated_adata.obs_names.tolist() == ["CL_A", "CL_B", "CL_C"]
        assert curated_adata.var_names.tolist() == ["DrugX", "DrugY"]
        assert np.isfinite(curated_adata[:, "DrugX"].X).all()

    def test_the_derived_quality_and_error_layers_reach_the_caller_shaped_like_x(
        self, curated_adata: anndata.AnnData
    ) -> None:
        expected = {"EC50", "IC50", "LN_IC50", "AUC", "R2", "regulation"}
        expected |= {"pec50_error", "slope_error", "front_error", "back_error"}

        assert expected <= set(curated_adata.layers)
        for name, layer in curated_adata.layers.items():
            assert layer.shape == curated_adata.shape, name

    def test_native_identifiers_survive_as_strings_without_changing_a_fit(self, curated_adata: anndata.AnnData) -> None:
        native = build_dose_response_df()
        native["sample"] = native["sample"].map({"CL_A": "ACH-1", "CL_B": "SIDM00400", "CL_C": "2004"})
        native["drug"] = native["drug"].map({"DrugX": "DRUG_1047", "DrugY": "BRD-K02251932"})

        relabelled = curate(native, max_workers=1, fit_speed="fast")

        assert set(relabelled.obs_names) == {"ACH-1", "SIDM00400", "2004"}
        assert set(relabelled.var_names) == {"DRUG_1047", "BRD-K02251932"}
        renamed = relabelled[["ACH-1", "SIDM00400", "2004"], ["DRUG_1047", "BRD-K02251932"]]
        np.testing.assert_array_equal(renamed.X, curated_adata.X)

    def test_curate_is_exactly_preprocess_fit_postprocess_build(
        self, curated_adata: anndata.AnnData, dose_response_df: pd.DataFrame
    ) -> None:
        from drevalpy.curation._fit import fit_groups
        from drevalpy.curation._postprocess import postprocess
        from drevalpy.curation._preprocess import preprocess

        composed = build_anndata(postprocess(fit_groups(preprocess(dose_response_df), max_workers=1, fit_speed="fast")))

        np.testing.assert_array_equal(composed.X, curated_adata.X)
        assert set(composed.layers) == set(curated_adata.layers)
        for name in curated_adata.layers.keys():
            np.testing.assert_array_equal(composed.layers[name], curated_adata.layers[name], err_msg=name)
