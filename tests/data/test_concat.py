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
from drevalpy.data._replicate_fits import KEY, SAMPLE_COLUMN
from drevalpy.errors import InvalidInputError
from drevalpy.types.data.dataset import Dataset
from tests.data._replicate_fits_table import fits_table

pytestmark = pytest.mark.slow

FEW = [f"FEATURE{i:04d}" for i in range(6)]
MANY = [f"FEATURE{i:04d}" for i in range(10)]


def _first() -> Dataset:
    return drevalpy.synthetic(name="FIRST", n_samples=10, n_drugs=4, feature_names=MANY, seed=1)


def _second() -> Dataset:
    return drevalpy.synthetic(name="SECOND", n_samples=8, n_drugs=6, feature_names=MANY, seed=2)


def _rebuilt(dataset: Dataset, **extra: ad.AnnData) -> Dataset:
    md.set_options(pull_on_update=False)
    mdata = md.MuData({**dataset.mdata.mod, **extra})
    mdata.obs = dataset.mdata.obs.reindex(mdata.obs_names)
    return Dataset(mdata, name=dataset.name)


def _extra_response(dataset: Dataset, seed: int = 0) -> ad.AnnData:
    source = dataset.response
    extra = ad.AnnData(
        X=np.asarray(source.X) + seed,
        obs=pd.DataFrame(index=source.obs_names),
        var=pd.DataFrame(index=source.var_names),
        uns={"x_column": "pEC50", "rr_seed": seed},
    )
    extra.layers["LN_IC50"] = np.asarray(source.layers["LN_IC50"])
    return extra


def test_signature_and_export() -> None:
    parameters = inspect.signature(concat).parameters

    assert list(parameters) == ["datasets", "join", "name"]
    assert parameters["join"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["join"].default == "exact"
    assert parameters["name"].default is None
    assert data.concat is concat


def test_rows_are_prefixed_in_input_order_and_obs_gains_study_and_source_id() -> None:
    result = concat([_first(), _second()])

    ids = list(result.sample_ids)
    obs = result.mdata.obs
    assert len(ids) == 18
    assert ids[:2] == ["FIRST::SYNTH-000", "FIRST::SYNTH-001"]
    assert ids[10] == "SECOND::SYNTH-000"
    assert isinstance(obs["study"].dtype, pd.CategoricalDtype)
    assert list(obs["study"].cat.categories) == ["FIRST", "SECOND"]
    assert obs.loc["SECOND::SYNTH-003", "study"] == "SECOND"
    assert obs.loc["SECOND::SYNTH-003", "source_sample_id"] == "SYNTH-003"
    assert obs.loc["FIRST::SYNTH-003", "tissue"] == _first().sample_meta.loc["SYNTH-003", "tissue"]
    studies = result.sample_column("study", np.array(["FIRST::SYNTH-001", "SECOND::SYNTH-001", "missing"]))
    assert list(studies[:2]) == ["FIRST", "SECOND"]
    assert studies[2] != studies[2]


def test_obs_columns_are_the_union() -> None:
    first, second = _first(), _second()
    first.mdata.obs["only_first"] = 1.0
    second.mdata.obs["only_second"] = "x"

    obs = concat([first, second]).mdata.obs

    assert {"tissue", "only_first", "only_second", "study", "source_sample_id"} <= set(obs.columns)
    assert np.isnan(obs.loc["SECOND::SYNTH-000", "only_first"])
    assert obs.loc["FIRST::SYNTH-000", "only_first"] == 1.0


def test_name_defaults_to_the_joined_input_names_or_the_argument() -> None:
    assert concat([_first(), _second()]).name == "FIRST+SECOND"
    named = concat([_first(), _second()], name="BOTH")
    assert named.name == "BOTH"
    assert named.mdata.uns["dataset_name"] == "BOTH"


def test_view_rows_are_stacked_with_prefixed_names() -> None:
    first, second = _first(), _second()

    view = concat([first, second]).mdata.mod["gene_expression"]

    assert list(view.obs_names) == [f"FIRST::{i}" for i in first.sample_ids] + [
        f"SECOND::{i}" for i in second.sample_ids
    ]
    assert list(view.var_names) == MANY
    np.testing.assert_array_equal(np.asarray(view.X)[:10], np.asarray(first.mdata.mod["gene_expression"].X))
    np.testing.assert_array_equal(np.asarray(view.X)[10:], np.asarray(second.mdata.mod["gene_expression"].X))


def test_exact_join_rejects_different_feature_names_and_orders() -> None:
    fewer = drevalpy.synthetic(name="SECOND", n_samples=8, feature_names=FEW)
    reversed_ = drevalpy.synthetic(name="SECOND", n_samples=8, feature_names=MANY[::-1])

    with pytest.raises(ValueError, match=r"View 'gene_expression'.*4 features of FIRST missing in SECOND") as error:
        concat([_first(), fewer])
    with pytest.raises(InvalidInputError, match="different order"):
        concat([_first(), reversed_])

    assert "FEATURE0006" in str(error.value)


def test_inner_join_keeps_the_shared_features_in_the_first_inputs_order_or_raises_without_any() -> None:
    first = drevalpy.synthetic(name="FIRST", n_samples=8, feature_names=MANY)
    second = drevalpy.synthetic(name="SECOND", n_samples=8, feature_names=["FEATURE0007", "FEATURE0002", "OTHER"])
    disjoint = drevalpy.synthetic(name="SECOND", n_samples=8, feature_names=["A", "B"])

    view = concat([first, second], join="inner").mdata.mod["gene_expression"]

    assert list(view.var_names) == ["FEATURE0002", "FEATURE0007"]
    expected = second.mdata.mod["gene_expression"][:, ["FEATURE0002", "FEATURE0007"]].X
    np.testing.assert_array_equal(np.asarray(view.X)[8:], np.asarray(expected))
    with pytest.raises(ValueError, match="no feature names shared"):
        concat([first, disjoint], join="inner")


@pytest.mark.parametrize("join", ["exact", "inner"])
def test_a_unit_mismatch_always_raises(join: str) -> None:
    first, second = _first(), _second()
    first.mdata.mod["gene_expression"].uns["expression_unit"] = "log2_tpm_plus_1"
    second.mdata.mod["gene_expression"].uns["expression_unit"] = "tpm"

    with pytest.raises(ValueError, match="expression_unit.*FIRST='log2_tpm_plus_1'.*SECOND='tpm'"):
        concat([first, second], join=join)


def test_a_unit_recorded_by_only_one_input_raises_and_equal_units_are_kept() -> None:
    first, second = _first(), _second()
    first.mdata.mod["gene_expression"].uns["expression_unit"] = "tpm"

    with pytest.raises(ValueError, match="SECOND=not recorded"):
        concat([first, second])

    second.mdata.mod["gene_expression"].uns["expression_unit"] = "tpm"
    assert concat([first, second]).mdata.mod["gene_expression"].uns["expression_unit"] == "tpm"


def test_a_view_missing_in_one_input_raises_for_exact_and_is_dropped_with_a_warning_for_inner(
    caplog: pytest.LogCaptureFixture,
) -> None:
    second = drevalpy.synthetic(
        name="SECOND", n_samples=8, omics=("gene_expression", "methylation"), feature_names=MANY
    )

    with pytest.raises(ValueError, match=r"View 'methylation' is missing in \['FIRST'\]"):
        concat([_first(), second])
    with caplog.at_level(logging.WARNING):
        result = concat([_first(), second], join="inner")

    assert "methylation" not in result.mdata.mod
    assert "gene_expression" in result.mdata.mod
    assert "dropped view 'methylation'" in caplog.text


def test_layers_present_in_every_input_are_stacked_and_others_dropped(caplog: pytest.LogCaptureFixture) -> None:
    first, second = _first(), _second()
    for dataset in (first, second):
        view = dataset.mdata.mod["gene_expression"]
        view.layers["counts"] = np.asarray(view.X) * 2
    first.mdata.mod["gene_expression"].layers["only_first"] = first.mdata.mod["gene_expression"].X

    with caplog.at_level(logging.WARNING):
        view = concat([first, second]).mdata.mod["gene_expression"]

    assert view.layers is not None
    assert set(view.layers.keys()) - {None} == {"counts"}
    np.testing.assert_array_equal(
        np.asarray(view.layers["counts"])[10:], np.asarray(second.mdata.mod["gene_expression"].X) * 2
    )
    assert "only_first" in caplog.text


def test_a_sample_without_a_view_row_stays_a_row_of_the_result() -> None:
    first = drevalpy.synthetic(name="FIRST", n_samples=10, omics={"gene_expression": 6}, feature_names=MANY)

    result = concat([first, _second()])

    assert result.mdata.mod["gene_expression"].n_obs == 6 + 8
    assert result.response.n_obs == 18


def test_response_columns_are_the_union_with_nan_where_a_drug_is_absent() -> None:
    first, second = _first(), _second()

    result = concat([first, second])
    matrix = result.response_matrix

    assert list(result.drug_ids) == [str(100000 + i) for i in range(6)]
    np.testing.assert_array_equal(matrix[:10, :4], first.response_matrix)
    np.testing.assert_array_equal(matrix[10:], second.response_matrix)
    assert np.isnan(matrix[:10, 4:]).all()


def test_layers_are_unioned_and_a_layer_missing_in_one_input_is_nan_there() -> None:
    first, second = _first(), _second()
    del second.response.layers["R2"]

    result = concat([first, second])

    r2 = result.get_response_layer("R2")
    np.testing.assert_array_equal(r2[:10, :4], first.get_response_layer("R2"))
    assert np.isnan(r2[10:]).all()
    np.testing.assert_array_equal(result.get_response_layer("LN_IC50")[10:], second.get_response_layer("LN_IC50"))


def test_response_obs_and_var_merge() -> None:
    first, second = _first(), _second()
    first.response.obs["native_ids"] = "a"
    first.response.var["target"] = [np.nan, "EGFR", np.nan, np.nan]
    second.response.var["target"] = ["ABL", "OTHER", np.nan, "MEK", "RAF", np.nan]
    first.response.var["is_combination"] = False
    second.response.var["is_combination"] = False

    response = concat([first, second]).response

    assert response.obs.index[0] == "FIRST::SYNTH-000"
    assert response.obs.loc["FIRST::SYNTH-000", "native_ids"] == "a"
    assert response.obs.loc["SECOND::SYNTH-000", "tissue"] == second.sample_meta.loc["SYNTH-000", "tissue"]
    assert response.var["target"].iloc[:2].tolist() == ["ABL", "EGFR"]
    assert response.var.loc["100003", "target"] == "MEK"
    assert response.var["is_combination"].dtype == bool
    assert response.var.loc["100005", "drug_name"] == "synthdrug5"


def test_smiles_conflicts_raise_and_a_missing_smiles_is_filled_from_the_other_input() -> None:
    conflicting = _second()
    conflicting.response.var.loc["100001", "canonical_smiles"] = "CCO"
    conflicting.response.var.loc["100003", "canonical_smiles"] = "CCN"
    gap = _second()
    gap.response.var["canonical_smiles"] = gap.response.var["canonical_smiles"].astype(object)
    gap.response.var.loc["100001", "canonical_smiles"] = np.nan

    with pytest.raises(ValueError, match=r"canonical_smiles differs.*2 drug IDs: 100001, 100003"):
        concat([_first(), conflicting])

    filled = concat([_first(), gap]).response.var
    assert filled.loc["100001", "canonical_smiles"] == _first().response.var.loc["100001", "canonical_smiles"]


def test_response_uns_must_be_equal_to_be_kept() -> None:
    first, second = _first(), _second()
    first.response.uns["x_column"] = "pEC50"
    second.response.uns["x_column"] = "AUC"

    with pytest.raises(ValueError, match=r"response: uns\['x_column'\]"):
        concat([first, second])

    second.response.uns["x_column"] = "pEC50"
    assert concat([first, second]).response.uns["x_column"] == "pEC50"


def test_extra_response_modalities_are_stacked_like_the_response_or_dropped(caplog: pytest.LogCaptureFixture) -> None:
    first, second = _first(), _second()
    both_first = _rebuilt(first, response_rr1=_extra_response(first, 1))
    both_second = _rebuilt(second, response_rr1=_extra_response(second, 2))

    extra = concat([both_first, both_second]).mdata.mod["response_rr1"]
    with caplog.at_level(logging.WARNING):
        dropped = concat([both_first, _second()])

    assert extra.shape == (18, 6)
    assert extra.obs_names[10] == "SECOND::SYNTH-000"
    assert extra.uns["x_column"] == "pEC50"
    assert "rr_seed" not in extra.uns
    assert extra.layers is not None
    assert "LN_IC50" in extra.layers
    np.testing.assert_array_equal(np.asarray(extra.X)[:10, :4], np.asarray(both_first.mdata.mod["response_rr1"].X))
    assert "response_rr1" not in dropped.mdata.mod
    assert "dropped modality 'response_rr1'" in caplog.text


def test_replicate_fits_are_stacked_or_dropped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    first, second = _first(), _second()
    for dataset in (first, second):
        dataset.response.uns[KEY] = fits_table(list(dataset.sample_ids), list(dataset.drug_ids))

    fits = concat([first, second]).response.uns[KEY]
    with caplog.at_level(logging.WARNING):
        dropped = concat([first, _second()])

    assert len(fits) == len(first.response.uns[KEY]) + len(second.response.uns[KEY])
    assert fits[SAMPLE_COLUMN].iloc[0] == "FIRST::SYNTH-000"
    assert fits[SAMPLE_COLUMN].iloc[-1] == "SECOND::SYNTH-007"
    assert first.response.uns[KEY][SAMPLE_COLUMN].iloc[0] == "SYNTH-000"
    assert KEY not in dropped.response.uns
    assert f"dropped response uns['{KEY}'], missing in ['SECOND']" in caplog.text


def test_uns_keys_are_kept_when_equal_and_dropped_with_a_warning_otherwise(caplog: pytest.LogCaptureFixture) -> None:
    first, second = _first(), _second()
    first.mdata.uns.update({"shared": "same", "differs": "a", "drug_graphs": {"1": [1, 2]}})
    second.mdata.uns.update({"shared": "same", "differs": "b", "drug_graphs": {"1": [1, 2]}})
    first.response.obsm["pca_v0"] = np.zeros((10, 2))
    first.mdata.uns["sample_featurizer_variants"] = "{}"

    with caplog.at_level(logging.WARNING):
        result = concat([first, second])

    assert result.mdata.uns["shared"] == "same"
    assert result.mdata.uns["drug_graphs"] == {"1": [1, 2]}
    assert "differs" not in result.mdata.uns
    assert "differs" in caplog.text
    assert "sample_featurizer_variants" not in result.mdata.uns
    assert len(result.response.obsm) == 0
    assert "pca_v0" in caplog.text


def test_rejected_inputs() -> None:
    randomized = _second()
    randomized.randomization = ("permutation", "gene_expression")
    combined = concat([_first(), _second()])

    with pytest.raises(InvalidInputError, match="at least two"):
        concat([_first()])
    with pytest.raises(InvalidInputError, match="unique"):
        concat([_first(), drevalpy.synthetic(name="FIRST", n_samples=6)])
    with pytest.raises(InvalidInputError, match="must not contain"):
        concat([_first(), drevalpy.synthetic(name="A::B", n_samples=6)])
    with pytest.raises(InvalidInputError, match=r"Randomized datasets cannot be concatenated: \['SECOND'\]"):
        concat([_first(), randomized])
    with pytest.raises(InvalidInputError, match="already have a 'study'"):
        concat([combined, drevalpy.synthetic(name="THIRD", n_samples=6)])
    with pytest.raises(InvalidInputError, match="join must be"):
        concat([_first(), _second()], join="outer")  # type: ignore[arg-type]


def test_inputs_are_unchanged_after_concat() -> None:
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


def test_save_load_round_trip_preserves_the_result_and_the_loaded_result_splits(tmp_path: UPath) -> None:
    result = concat([_first(), _second()])
    path = tmp_path / "both.h5mu"

    result.save(str(path))
    loaded = Dataset.load(path)
    folds = drevalpy.split(loaded, "LPO", n_splits=3)

    assert loaded.name == "FIRST+SECOND"
    assert list(loaded.sample_ids) == list(result.sample_ids)
    assert list(loaded.drug_ids) == list(result.drug_ids)
    np.testing.assert_array_equal(loaded.response_matrix, result.response_matrix)
    np.testing.assert_array_equal(loaded.get_response_layer("R2"), result.get_response_layer("R2"))
    assert list(loaded.mdata.obs["study"].astype(str)) == list(result.mdata.obs["study"].astype(str))
    assert list(loaded.mdata.obs["source_sample_id"].astype(str)) == list(result.mdata.obs["source_sample_id"])
    assert len(folds) == 3
    assert all(not (fold.train.mask & fold.test.mask).any() for fold in folds)
    assert sum(int(fold.test.mask.sum()) for fold in folds) == int(np.isfinite(loaded.response_matrix).sum())
    assert drevalpy.split(result, "LPO", n_splits=2)[0].metadata["dataset"] == "FIRST+SECOND"


@pytest.mark.network
def test_the_published_screens_concatenate_and_split() -> None:
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
