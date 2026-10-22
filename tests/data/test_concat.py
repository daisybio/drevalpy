"""Tests for :mod:`drevalpy.data._concat`, the ``drevalpy.data.concat`` entry point."""

from __future__ import annotations

import inspect
import logging

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from upath import UPath

import drevalpy
from drevalpy import data
from drevalpy.data._concat import concat
from drevalpy.types.data.dataset import Dataset

pytestmark = pytest.mark.slow

FEW = [f"FEATURE{i:04d}" for i in range(6)]
MANY = [f"FEATURE{i:04d}" for i in range(10)]


def _first() -> Dataset:
    """Ten samples, drugs 100000-100003, ten features."""
    return drevalpy.synthetic(name="FIRST", n_samples=10, n_drugs=4, feature_names=MANY, seed=1)


def _second() -> Dataset:
    """Eight samples named like the first's, drugs 100000-100005, ten features."""
    return drevalpy.synthetic(name="SECOND", n_samples=8, n_drugs=6, feature_names=MANY, seed=2)


def _rebuilt(dataset: Dataset, **extra: ad.AnnData) -> Dataset:
    """A new dataset with the same modalities plus *extra*, so a test can add a view."""
    md.set_options(pull_on_update=False)
    mdata = md.MuData({**dataset.mdata.mod, **extra})
    mdata.obs = dataset.mdata.obs.reindex(mdata.obs_names)
    return Dataset(mdata, name=dataset.name)


def _without(dataset: Dataset, key: str) -> Dataset:
    """A new dataset lacking the modality *key*."""
    md.set_options(pull_on_update=False)
    mdata = md.MuData({name: adata for name, adata in dataset.mdata.mod.items() if name != key})
    mdata.obs = dataset.mdata.obs.reindex(mdata.obs_names)
    return Dataset(mdata, name=dataset.name)


def _extra_response(dataset: Dataset, seed: int = 0) -> ad.AnnData:
    """A response-shaped modality like the curated ``response_rr1``."""
    source = dataset.response
    extra = ad.AnnData(
        X=source.X + seed,
        obs=pd.DataFrame(index=source.obs_names),
        var=pd.DataFrame(index=source.var_names),
        uns={"x_column": "pEC50", "rr_seed": seed},
    )
    extra.layers["LN_IC50"] = np.asarray(source.layers["LN_IC50"])
    return extra


class TestInterface:
    def test_signature_is_the_planned_interface(self) -> None:
        parameters = inspect.signature(concat).parameters

        assert list(parameters) == ["datasets", "join", "name"]
        assert parameters["join"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["join"].default == "exact"
        assert parameters["name"].default is None

    def test_is_exported_from_the_data_package(self) -> None:
        assert data.concat is concat


class TestRows:
    def test_ids_are_prefixed_and_kept_in_input_order(self) -> None:
        result = concat([_first(), _second()])

        ids = list(result.sample_ids)
        assert len(ids) == 18
        assert ids[:2] == ["FIRST::SYNTH-000", "FIRST::SYNTH-001"]
        assert ids[10] == "SECOND::SYNTH-000"

    def test_obs_gains_study_and_source_sample_id(self) -> None:
        result = concat([_first(), _second()])

        obs = result.mdata.obs
        assert isinstance(obs["study"].dtype, pd.CategoricalDtype)
        assert list(obs["study"].cat.categories) == ["FIRST", "SECOND"]
        assert obs.loc["SECOND::SYNTH-003", "study"] == "SECOND"
        assert obs.loc["SECOND::SYNTH-003", "source_sample_id"] == "SYNTH-003"
        assert obs.loc["FIRST::SYNTH-003", "tissue"] == _first().sample_meta.loc["SYNTH-003", "tissue"]

    def test_obs_columns_are_the_union(self) -> None:
        first, second = _first(), _second()
        first.mdata.obs["only_first"] = 1.0
        second.mdata.obs["only_second"] = "x"

        obs = concat([first, second]).mdata.obs

        assert {"tissue", "only_first", "only_second", "study", "source_sample_id"} <= set(obs.columns)
        assert obs.loc["SECOND::SYNTH-000", "only_first"] != obs.loc["SECOND::SYNTH-000", "only_first"]
        assert obs.loc["FIRST::SYNTH-000", "only_first"] == 1.0

    def test_sample_column_reads_the_study(self) -> None:
        result = concat([_first(), _second()])

        studies = result.sample_column("study", np.array(["FIRST::SYNTH-001", "SECOND::SYNTH-001", "missing"]))

        assert list(studies[:2]) == ["FIRST", "SECOND"]
        assert studies[2] != studies[2]

    def test_name_defaults_to_the_joined_input_names_or_the_argument(self) -> None:
        assert concat([_first(), _second()]).name == "FIRST+SECOND"
        assert concat([_first(), _second()], name="BOTH").name == "BOTH"
        assert concat([_first(), _second()], name="BOTH").mdata.uns["dataset_name"] == "BOTH"


class TestViews:
    def test_view_rows_are_stacked_with_prefixed_names(self) -> None:
        first, second = _first(), _second()

        view = concat([first, second]).mdata.mod["gene_expression"]

        assert list(view.obs_names) == [f"FIRST::{i}" for i in first.sample_ids] + [
            f"SECOND::{i}" for i in second.sample_ids
        ]
        assert list(view.var_names) == MANY
        np.testing.assert_array_equal(view.X[:10], first.mdata.mod["gene_expression"].X)
        np.testing.assert_array_equal(view.X[10:], second.mdata.mod["gene_expression"].X)

    def test_exact_join_rejects_different_feature_names(self) -> None:
        second = drevalpy.synthetic(name="SECOND", n_samples=8, feature_names=FEW)

        with pytest.raises(ValueError, match=r"View 'gene_expression'.*4 features of FIRST missing in SECOND") as error:
            concat([_first(), second])

        assert "FEATURE0006" in str(error.value)

    def test_exact_join_rejects_a_different_order(self) -> None:
        second = drevalpy.synthetic(name="SECOND", n_samples=8, feature_names=MANY[::-1])

        with pytest.raises(ValueError, match="different order"):
            concat([_first(), second])

    def test_inner_join_keeps_the_shared_features_in_the_first_inputs_order(self) -> None:
        first = drevalpy.synthetic(name="FIRST", n_samples=8, feature_names=MANY)
        second = drevalpy.synthetic(name="SECOND", n_samples=8, feature_names=["FEATURE0007", "FEATURE0002", "OTHER"])

        view = concat([first, second], join="inner").mdata.mod["gene_expression"]

        assert list(view.var_names) == ["FEATURE0002", "FEATURE0007"]
        expected = second.mdata.mod["gene_expression"][:, ["FEATURE0002", "FEATURE0007"]].X
        np.testing.assert_array_equal(view.X[8:], expected)

    def test_inner_join_without_shared_features_raises(self) -> None:
        second = drevalpy.synthetic(name="SECOND", n_samples=8, feature_names=["A", "B"])

        with pytest.raises(ValueError, match="no feature names shared"):
            concat([_first(), second], join="inner")

    @pytest.mark.parametrize("join", ["exact", "inner"])
    def test_a_unit_mismatch_always_raises(self, join: str) -> None:
        first, second = _first(), _second()
        first.mdata.mod["gene_expression"].uns["expression_unit"] = "log2_tpm_plus_1"
        second.mdata.mod["gene_expression"].uns["expression_unit"] = "tpm"

        with pytest.raises(ValueError, match="expression_unit.*FIRST='log2_tpm_plus_1'.*SECOND='tpm'"):
            concat([first, second], join=join)

    def test_a_unit_recorded_by_only_one_input_raises(self) -> None:
        first = _first()
        first.mdata.mod["gene_expression"].uns["expression_unit"] = "tpm"

        with pytest.raises(ValueError, match="SECOND=not recorded"):
            concat([first, _second()])

    def test_equal_units_are_kept(self) -> None:
        first, second = _first(), _second()
        for dataset in (first, second):
            dataset.mdata.mod["gene_expression"].uns["expression_unit"] = "tpm"

        assert concat([first, second]).mdata.mod["gene_expression"].uns["expression_unit"] == "tpm"

    def test_a_view_missing_in_one_input_raises_for_exact(self) -> None:
        second = drevalpy.synthetic(
            name="SECOND", n_samples=8, omics=("gene_expression", "methylation"), feature_names=MANY
        )

        with pytest.raises(ValueError, match=r"View 'methylation' is missing in \['FIRST'\]"):
            concat([_first(), second])

    def test_a_view_missing_in_one_input_is_dropped_with_a_warning_for_inner(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        second = drevalpy.synthetic(
            name="SECOND", n_samples=8, omics=("gene_expression", "methylation"), feature_names=MANY
        )

        with caplog.at_level(logging.WARNING):
            result = concat([_first(), second], join="inner")

        assert "methylation" not in result.mdata.mod
        assert "gene_expression" in result.mdata.mod
        assert "dropped view 'methylation'" in caplog.text

    def test_layers_present_in_every_input_are_stacked_and_others_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        first, second = _first(), _second()
        for dataset in (first, second):
            view = dataset.mdata.mod["gene_expression"]
            view.layers["counts"] = view.X * 2
        first.mdata.mod["gene_expression"].layers["only_first"] = first.mdata.mod["gene_expression"].X

        with caplog.at_level(logging.WARNING):
            view = concat([first, second]).mdata.mod["gene_expression"]

        assert set(view.layers.keys()) - {None} == {"counts"}
        np.testing.assert_array_equal(view.layers["counts"][10:], second.mdata.mod["gene_expression"].X * 2)
        assert "only_first" in caplog.text

    def test_a_sample_without_a_view_row_stays_a_row_of_the_result(self) -> None:
        first = drevalpy.synthetic(name="FIRST", n_samples=10, omics={"gene_expression": 6}, feature_names=MANY)

        result = concat([first, _second()])

        assert result.mdata.mod["gene_expression"].n_obs == 6 + 8
        assert result.response.n_obs == 18


class TestResponse:
    def test_columns_are_the_union_in_first_seen_order(self) -> None:
        result = concat([_first(), _second()])

        assert list(result.drug_ids) == [str(100000 + i) for i in range(6)]

    def test_matrix_is_stacked_with_nan_where_a_drug_is_absent(self) -> None:
        first, second = _first(), _second()

        matrix = concat([first, second]).response_matrix

        np.testing.assert_array_equal(matrix[:10, :4], first.response_matrix)
        np.testing.assert_array_equal(matrix[10:], second.response_matrix)
        assert np.isnan(matrix[:10, 4:]).all()

    def test_layers_are_unioned_and_a_layer_missing_in_one_input_is_nan_there(self) -> None:
        first, second = _first(), _second()
        del second.response.layers["R2"]

        result = concat([first, second])

        r2 = result.get_response_layer("R2")
        np.testing.assert_array_equal(r2[:10, :4], first.get_response_layer("R2"))
        assert np.isnan(r2[10:]).all()
        np.testing.assert_array_equal(result.get_response_layer("LN_IC50")[10:], second.get_response_layer("LN_IC50"))

    def test_response_obs_carries_the_prefixed_ids(self) -> None:
        first, second = _first(), _second()
        first.response.obs["native_ids"] = "a"

        obs = concat([first, second]).response.obs

        assert obs.index[0] == "FIRST::SYNTH-000"
        assert obs.loc["FIRST::SYNTH-000", "native_ids"] == "a"
        assert obs.loc["SECOND::SYNTH-000", "tissue"] == second.sample_meta.loc["SYNTH-000", "tissue"]

    def test_var_takes_the_first_non_null_value(self) -> None:
        first, second = _first(), _second()
        first.response.var["target"] = [np.nan, "EGFR", np.nan, np.nan]
        second.response.var["target"] = ["ABL", "OTHER", np.nan, "MEK", "RAF", np.nan]
        first.response.var["is_combination"] = False
        second.response.var["is_combination"] = False

        var = concat([first, second]).response.var

        assert var["target"].iloc[:6].tolist()[:2] == ["ABL", "EGFR"]
        assert var.loc["100003", "target"] == "MEK"
        assert var["is_combination"].dtype == bool
        assert var.loc["100005", "drug_name"] == "synthdrug5"

    def test_a_smiles_conflict_raises_and_lists_the_drug_ids(self) -> None:
        second = _second()
        second.response.var.loc["100001", "canonical_smiles"] = "CCO"
        second.response.var.loc["100003", "canonical_smiles"] = "CCN"

        with pytest.raises(ValueError, match=r"canonical_smiles differs.*2 drug IDs: 100001, 100003"):
            concat([_first(), second])

    def test_a_missing_smiles_is_filled_from_the_other_input(self) -> None:
        second = _second()
        second.response.var["canonical_smiles"] = second.response.var["canonical_smiles"].astype(object)
        second.response.var.loc["100001", "canonical_smiles"] = np.nan

        var = concat([_first(), second]).response.var

        assert var.loc["100001", "canonical_smiles"] == _first().response.var.loc["100001", "canonical_smiles"]

    def test_response_uns_must_be_equal(self) -> None:
        first, second = _first(), _second()
        first.response.uns["x_column"] = "pEC50"
        second.response.uns["x_column"] = "AUC"

        with pytest.raises(ValueError, match=r"response: uns\['x_column'\]"):
            concat([first, second])

    def test_equal_response_uns_is_kept(self) -> None:
        first, second = _first(), _second()
        for dataset in (first, second):
            dataset.response.uns["x_column"] = "pEC50"

        assert concat([first, second]).response.uns["x_column"] == "pEC50"


class TestExtraResponses:
    def test_modalities_every_input_has_are_stacked_like_the_response(self) -> None:
        first, second = _first(), _second()
        first = _rebuilt(first, response_rr1=_extra_response(first, 1))
        second = _rebuilt(second, response_rr1=_extra_response(second, 2))

        extra = concat([first, second]).mdata.mod["response_rr1"]

        assert extra.shape == (18, 6)
        assert extra.obs_names[10] == "SECOND::SYNTH-000"
        assert extra.uns["x_column"] == "pEC50"
        assert "rr_seed" not in extra.uns
        assert "LN_IC50" in extra.layers
        np.testing.assert_array_equal(extra.X[:10, :4], first.mdata.mod["response_rr1"].X)

    def test_a_modality_missing_in_one_input_is_dropped_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        first = _first()
        first = _rebuilt(first, response_rr1=_extra_response(first))

        with caplog.at_level(logging.WARNING):
            result = concat([first, _second()])

        assert "response_rr1" not in result.mdata.mod
        assert "dropped modality 'response_rr1'" in caplog.text


class TestUns:
    def test_other_keys_are_kept_when_equal_and_dropped_with_a_warning_otherwise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        first, second = _first(), _second()
        first.mdata.uns.update({"shared": "same", "differs": "a", "drug_graphs": {"1": [1, 2]}})
        second.mdata.uns.update({"shared": "same", "differs": "b", "drug_graphs": {"1": [1, 2]}})

        with caplog.at_level(logging.WARNING):
            uns = concat([first, second]).mdata.uns

        assert uns["shared"] == "same"
        assert uns["drug_graphs"] == {"1": [1, 2]}
        assert "differs" not in uns
        assert "differs" in caplog.text

    def test_stored_featurizer_variants_are_not_carried(self, caplog: pytest.LogCaptureFixture) -> None:
        first = _first()
        first.response.obsm["pca_v0"] = np.zeros((10, 2))
        first.mdata.uns["sample_featurizer_variants"] = "{}"

        with caplog.at_level(logging.WARNING):
            result = concat([first, _second()])

        assert "sample_featurizer_variants" not in result.mdata.uns
        assert len(result.response.obsm) == 0
        assert "pca_v0" in caplog.text


class TestRejectedInputs:
    def test_fewer_than_two_datasets(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            concat([_first()])

    def test_duplicate_names(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            concat([_first(), drevalpy.synthetic(name="FIRST", n_samples=6)])

    def test_a_name_containing_the_separator(self) -> None:
        with pytest.raises(ValueError, match="must not contain"):
            concat([_first(), drevalpy.synthetic(name="A::B", n_samples=6)])

    def test_a_randomized_input(self) -> None:
        randomized = _second()
        randomized.randomization = ("permutation", "gene_expression")

        with pytest.raises(ValueError, match=r"Randomized datasets cannot be concatenated: \['SECOND'\]"):
            concat([_first(), randomized])

    def test_an_already_concatenated_input(self) -> None:
        combined = concat([_first(), _second()])

        with pytest.raises(ValueError, match="already have a 'study'"):
            concat([combined, drevalpy.synthetic(name="THIRD", n_samples=6)])

    def test_unknown_join(self) -> None:
        with pytest.raises(ValueError, match="join must be"):
            concat([_first(), _second()], join="outer")  # type: ignore[arg-type]


class TestInputsAreUntouched:
    def test_inputs_are_unchanged_after_concat(self) -> None:
        first, second = _first(), _second()
        for dataset in (first, second):
            dataset.mdata.uns["shared"] = {"k": [1, 2]}
        before = [
            (
                dataset.name,
                dataset.response_matrix.copy(),
                dataset.mdata.obs.copy(),
                dataset.mdata.mod["gene_expression"].X.copy(),
                list(dataset.sample_ids),
                list(dataset.mdata.mod),
            )
            for dataset in (first, second)
        ]

        result = concat([first, second])
        result.response.X[:] = 0
        result.mdata.uns["shared"]["k"].append(3)

        for dataset, (name, matrix, obs, expression, ids, modalities) in zip((first, second), before, strict=True):
            assert dataset.name == name
            np.testing.assert_array_equal(dataset.response_matrix, matrix)
            pd.testing.assert_frame_equal(dataset.mdata.obs, obs)
            np.testing.assert_array_equal(dataset.mdata.mod["gene_expression"].X, expression)
            assert list(dataset.sample_ids) == ids
            assert list(dataset.mdata.mod) == modalities
            assert "study" not in dataset.mdata.obs.columns
            assert dataset.mdata.uns["shared"] == {"k": [1, 2]}


class TestRoundTrip:
    def test_save_and_load_preserve_the_result(self, tmp_path: UPath) -> None:
        result = concat([_first(), _second()])
        path = tmp_path / "both.h5mu"

        result.save(str(path))
        loaded = Dataset.load(path)

        assert loaded.name == "FIRST+SECOND"
        assert list(loaded.sample_ids) == list(result.sample_ids)
        assert list(loaded.drug_ids) == list(result.drug_ids)
        np.testing.assert_array_equal(loaded.response_matrix, result.response_matrix)
        np.testing.assert_array_equal(loaded.get_response_layer("R2"), result.get_response_layer("R2"))
        assert list(loaded.mdata.obs["study"].astype(str)) == list(result.mdata.obs["study"].astype(str))
        assert list(loaded.mdata.obs["source_sample_id"].astype(str)) == list(result.mdata.obs["source_sample_id"])

    def test_the_loaded_result_splits_with_lpo(self, tmp_path: UPath) -> None:
        path = tmp_path / "both.h5mu"
        concat([_first(), _second()]).save(str(path))
        loaded = Dataset.load(path)

        folds = drevalpy.split(loaded, "LPO", n_splits=3)

        assert len(folds) == 3
        assert all(fold.test.mask.any() for fold in folds)
        observed = np.isfinite(loaded.response_matrix)
        assert all(not (fold.train.mask & fold.test.mask).any() for fold in folds)
        assert sum(int(fold.test.mask.sum()) for fold in folds) == int(observed.sum())

    def test_the_result_splits_with_lpo_without_saving(self) -> None:
        result = concat([_first(), _second()])

        folds = drevalpy.split(result, "LPO", n_splits=2)

        assert len(folds) == 2
        assert folds[0].metadata["dataset"] == "FIRST+SECOND"


@pytest.mark.network
class TestPublishedScreens:
    """GDSC2 and CTRPv2 from the cache or the bucket; about 2.5 GB of memory."""

    def test_the_screens_concatenate_and_split(self) -> None:
        gdsc2, ctrpv2 = drevalpy.load("GDSC2"), drevalpy.load("CTRPv2")
        shared = ctrpv2.response.var.index.intersection(gdsc2.response.var.index)
        smiles = ctrpv2.response.var["canonical_smiles"].astype(object)
        smiles.loc[shared] = gdsc2.response.var.loc[shared, "canonical_smiles"].astype(object)
        ctrpv2.response.var["canonical_smiles"] = smiles
        n_samples = gdsc2.response.n_obs + ctrpv2.response.n_obs
        n_drugs = len(set(gdsc2.drug_ids) | set(ctrpv2.drug_ids))

        result = concat([gdsc2, ctrpv2])
        del gdsc2, ctrpv2

        assert result.response.shape == (n_samples, n_drugs)
        assert result.response.n_obs == 1854
        assert len(drevalpy.split(result, "LPO")) == 5
