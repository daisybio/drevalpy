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
    Source,
    Status,
    build_anndata,
    curate,
)
from drevalpy.errors import InvalidInputError
from tests._barrel_surface import DeclaredSurface
from tests.curation.conftest import build_dose_response_df, build_replicated_wells


class TestCurationSurface(DeclaredSurface):
    barrel = curation
    origins = {
        "METRICS": "drevalpy.curation._replicate_fits",
        "Source": "drevalpy.curation._replicate_fits",
        "Status": "drevalpy.curation._fit",
        "build_anndata": "drevalpy.curation._anndata",
    }
    unpinned_names = ("DEFAULT_FIT_SPEED", "FIT_SPEEDS", "SUPPORTED_FIT_TYPES", "curate")
    callable_names = ("build_anndata", "curate")


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

    def test_curate_is_exactly_series_fit_and_build(
        self, curated_adata: anndata.AnnData, dose_response_df: pd.DataFrame
    ) -> None:
        from drevalpy.curation._fit import FitOptions, fit_table
        from drevalpy.curation._series import number_series

        viability = number_series(dose_response_df, min_doses=4).viability
        fits = fit_table(viability, FitOptions(fit_speed="fast"))
        composed = build_anndata(fits.drop(columns="status"))

        np.testing.assert_array_equal(composed.X, curated_adata.X)
        assert set(composed.layers) == set(curated_adata.layers)
        for name in curated_adata.layers.keys():
            np.testing.assert_array_equal(composed.layers[name], curated_adata.layers[name], err_msg=name)

    def test_no_replicate_fits_table_unless_asked_for(self, curated_adata: anndata.AnnData) -> None:
        assert "replicate_fits" not in curated_adata.uns


class TestColumnValidation:
    @pytest.mark.parametrize(
        ("drop", "extra", "metadata", "match"),
        [
            ("intensity", {}, (), r"Missing column\(s\): \['intensity'\]"),
            (None, {}, ("plate",), r"Missing column\(s\): \['plate'\]"),
            (None, {"plate": 1}, (), "Unexpected column"),
            (None, {"replicate": 1}, ("replicate",), "clash"),
            (None, {"Eligible": True}, ("Eligible",), "clash"),
            (None, {"Plate": 1, "plate": 2}, ("Plate", "plate"), "not distinct"),
        ],
        ids=["required", "metadata", "unexpected", "known-as-metadata", "output-name", "case-duplicate"],
    )
    def test_a_table_curate_cannot_read_is_rejected_before_any_fitting(
        self, dose_response_df: pd.DataFrame, drop: str | None, extra: dict, metadata: tuple, match: str
    ) -> None:
        df = dose_response_df.drop(columns=[drop] if drop else []).assign(**extra)

        with pytest.raises(InvalidInputError, match=match):
            curate(df, metadata=metadata, max_workers=1, fit_speed="fast")

    def test_a_label_with_the_curvecurator_separator_is_rejected(self, dose_response_df: pd.DataFrame) -> None:
        dose_response_df.loc[0, "sample"] = "A|B"

        with pytest.raises(InvalidInputError, match="contain '\\|'"):
            curate(dose_response_df, max_workers=1, fit_speed="fast")


@pytest.mark.slow
class TestReplicateFits:
    @pytest.fixture(scope="class")
    def curated(self) -> anndata.AnnData:
        return curate(
            build_replicated_wells(), replicate_fits=True, metadata=("plate_date",), max_workers=1, fit_speed="fast"
        )

    def test_the_table_is_keyed_like_the_anndata(self, curated: anndata.AnnData) -> None:
        table = curated.uns["replicate_fits"]

        assert set(table["sample_id"]) <= set(curated.obs_names)
        assert set(table["pubchem_id"]) <= set(curated.var_names)
        assert (table["sample_id"].astype(str) == table["native_sample"].astype(str)).all()

    def test_the_hierarchy_is_numbered_per_pair(self, curated: anndata.AnnData) -> None:
        table = curated.uns["replicate_fits"]
        pair = table[(table["native_sample"] == "CL_A") & (table["native_drug"] == "DrugX")]

        assert pair[["replicate", "biological_replicate", "technical_replicate"]].values.tolist() == [
            [0, 0, 0],
            [1, 0, 1],
            [2, 1, 0],
            [3, 1, 1],
        ]
        assert pair["experiment"].astype(str).tolist() == ["day1", "day1", "day2", "day2"]
        assert set(pair["source"]) == {Source.REPLICATE}
        assert set(pair["status"]) == {Status.FITTED}
        assert np.isfinite(pair["pEC50"]).all()

    def test_a_pair_with_one_fittable_series_carries_the_pooled_fit(self, curated: anndata.AnnData) -> None:
        table = curated.uns["replicate_fits"]
        pair = table[(table["native_sample"] == "CL_B") & (table["native_drug"] == "DrugX")]

        fittable = pair[pair["status"] != Status.TOO_FEW_DOSES]
        assert fittable["source"].astype(str).tolist() == [Source.POOLED]
        assert fittable["pEC50"].iloc[0] == pytest.approx(float(np.asarray(curated["CL_B", "DrugX"].X)[0, 0]))
        short = pair[pair["status"] == Status.TOO_FEW_DOSES]
        assert short["n_doses"].tolist() == [2]
        assert short["pEC50"].isna().all()

    def test_metadata_is_kept_per_series_lowercased(self, curated: anndata.AnnData) -> None:
        table = curated.uns["replicate_fits"]

        assert "plate_date" in table.columns
        assert set(table["plate_date"].astype(str)) == {"2020-01-01", "2020-01-02"}

    def test_the_pooled_fit_is_the_same_with_and_without_replicate_fits(self, curated: anndata.AnnData) -> None:
        plain = curate(build_replicated_wells(), metadata=("plate_date",), max_workers=1, fit_speed="fast")

        np.testing.assert_array_equal(plain.X, curated.X)

    def test_the_table_survives_an_h5ad_round_trip(self, curated: anndata.AnnData, tmp_path) -> None:
        path = tmp_path / "curves.h5ad"
        curated.write_h5ad(path)

        table = anndata.read_h5ad(path).uns["replicate_fits"]

        pd.testing.assert_frame_equal(table, curated.uns["replicate_fits"], check_categorical=False)
