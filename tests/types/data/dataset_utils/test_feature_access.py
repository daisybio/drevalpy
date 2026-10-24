"""Dataset feature access: sample/drug views, aligned fetching, densification and sample columns."""

from __future__ import annotations

import logging

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from drevalpy.data import synthetic
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.dataset_utils._dense import to_dense
from drevalpy.types.data.dataset_utils.aligned_fetch import _aligned_fetch
from drevalpy.types.data.mudatalike import MuDataLike

SAMPLES = np.array(["cl1", "cl2", "cl3"])
DRUGS = np.array(["d1", "d2"])


def _dataset(
    *,
    varm: dict[str, np.ndarray] | None = None,
    obsm: dict[str, np.ndarray] | None = None,
    uns: dict[str, object] | None = None,
    gene_expression: np.ndarray | None = None,
    cnv_under_stored_name: bool = False,
) -> Dataset:
    response = ad.AnnData(
        X=np.arange(6.0, dtype=np.float32).reshape(3, 2),
        obs=pd.DataFrame({"tissue": ["Lung", "Blood", "Skin"]}, index=SAMPLES),
        var=pd.DataFrame(index=DRUGS),
    )
    for key, value in (varm or {}).items():
        response.varm[key] = value
    for key, value in (obsm or {}).items():
        response.obsm[key] = value

    mods: dict[str, ad.AnnData] = {"response": response}
    default_expression = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    mods["gene_expression"] = ad.AnnData(
        X=default_expression if gene_expression is None else gene_expression,
        obs=pd.DataFrame(index=SAMPLES),
        var=pd.DataFrame(index=["GENE1", "GENE2"]),
    )
    if cnv_under_stored_name:
        mods["copy_number_variation"] = ad.AnnData(
            X=np.zeros((3, 2), dtype=np.float32),
            obs=pd.DataFrame(index=SAMPLES),
            var=pd.DataFrame(index=["GENE1", "GENE2"]),
        )

    md.set_options(pull_on_update=False)
    mdata = md.MuData(mods)
    for key, value in (uns or {}).items():
        mdata.uns[key] = value
    return Dataset(mdata, name="feature-access")


def _zeros(*shape: int) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


class TestSampleFeatures:
    def test_modality_resolution(self):
        dataset = _dataset(cnv_under_stored_name=True)

        assert dataset.get_sample_features("copy_number_variation_gistic", SAMPLES).shape == (3, 2)
        with pytest.raises(KeyError, match="copy_number_variation_gistic"):
            dataset.get_sample_features("absent", SAMPLES)

    def test_feature_names(self):
        dataset = _dataset(obsm={"pathway_features": _zeros(3, 4)})

        assert dataset.get_sample_feature_names("gene_expression") == ("GENE1", "GENE2")
        assert dataset.get_sample_feature_names("absent") is None
        assert dataset.get_sample_feature_names("pathway_features") is None

    def test_obsm_features(self):
        dataset = _dataset(obsm={"pathway_features": np.ones((3, 4), dtype=np.float32)})

        assert dataset.get_sample_features("pathway_features", SAMPLES[:2]).shape == (2, 4)
        with pytest.raises(KeyError, match="sample IDs not found"):
            dataset.get_sample_features("pathway_features", np.array(["absent"]), strict=True)
        with pytest.raises(KeyError, match="obsm key 'pathway_features' not found"):
            _dataset().get_sample_features("pathway_features", SAMPLES)


class TestDrugFeatures:
    @pytest.mark.parametrize("key", ["pca", "pca:0"], ids=["exact", "variant-prefix"])
    def test_varm_key_resolution(self, key):
        assert _dataset(varm={key: _zeros(2, 3)}).get_drug_features("pca", DRUGS).shape == (2, 3)

    def test_a_prefix_match_requires_the_colon(self):
        with pytest.raises(KeyError, match="Drug feature 'pca' not found"):
            _dataset(varm={"pca_extra": _zeros(2, 3)}).get_drug_features("pca", DRUGS)

    def test_views_and_feature_names(self):
        frame = pd.DataFrame(_zeros(2, 2), columns=["bit0", "bit1"], index=DRUGS)
        dataset = _dataset(varm={"zeta": _zeros(2, 1), "alpha": _zeros(2, 1), "pca": _zeros(2, 3), "fp": frame})

        assert dataset.available_drug_views == ["alpha", "fp", "pca", "zeta"]
        assert dataset.get_drug_feature_names("pca") == ("0", "1", "2")
        assert dataset.get_drug_feature_names("fp") == ("bit0", "bit1")
        assert dataset.get_drug_feature_names("absent") is None
        with pytest.raises(KeyError, match="drug IDs not found"):
            dataset.get_drug_features("pca", np.array(["absent"]), strict=True)

    def test_drug_graphs(self):
        dataset = _dataset(uns={"drug_graphs": {"d1": {"x": np.zeros(1)}}})

        graphs = dataset.get_drug_graphs(np.array(["d2", "d1"]))

        assert graphs[0] is None
        assert graphs[1] is not None
        with pytest.raises(KeyError, match="'drug_graphs' not found"):
            _dataset().get_drug_graphs(DRUGS)


class TestRaggedDrugView:
    @staticmethod
    def _ragged() -> np.ndarray:
        ragged = np.empty((2, 1), dtype=object)
        ragged[0, 0] = np.ones((2, 3), dtype=np.float32)
        ragged[1, 0] = np.ones((3, 3), dtype=np.float32)
        return ragged

    def test_a_ragged_view_is_a_clear_input_error_not_a_bare_value_error(self):
        from drevalpy.errors import InvalidInputError

        dataset = _dataset(varm={"molgnet_features": self._ragged()})

        with pytest.raises(InvalidInputError, match="molgnet_features.*object array.*computes it from the SMILES"):
            dataset.get_drug_features("molgnet_features", DRUGS)

    def test_the_molgnet_featurizer_surfaces_the_error_instead_of_computing_over_it(self):
        from drevalpy.components.featurizers.drug.molgnet import MolGNetDrugFeaturizer
        from drevalpy.errors import InvalidInputError
        from drevalpy.types.data.feature_source import DrugFeatureSource

        dataset = _dataset(varm={"molgnet_features": self._ragged()})

        with pytest.raises(InvalidInputError, match="object array"):
            MolGNetDrugFeaturizer().fit(DrugFeatureSource(dataset, DRUGS), entity_ids=DRUGS)


class TestEntitiesWithModality:
    @pytest.mark.parametrize(
        ("matrix", "expected"),
        [
            ([[0.0, 1.0], [np.nan, np.nan], [4.0, 5.0]], {"cl1", "cl3"}),
            ([[0.0, np.nan], [2.0, 3.0], [4.0, 5.0]], {"cl1", "cl2", "cl3"}),
        ],
        ids=["all-nan-row-excluded", "partially-nan-row-counts"],
    )
    def test_gene_expression_nan_rows(self, matrix, expected):
        dataset = _dataset(gene_expression=np.array(matrix, dtype=np.float32))

        assert dataset.entities_with_modality("gene_expression") == frozenset(expected)

    def test_obsm_and_varm_views_are_filtered_the_same_way(self):
        dataset = _dataset(
            obsm={"pathway_features": np.array([[1.0], [np.nan], [3.0]], dtype=np.float32)},
            varm={"pca": np.array([[1.0], [np.nan]], dtype=np.float32)},
        )

        assert dataset.entities_with_modality("pathway_features") == frozenset({"cl1", "cl3"})
        assert dataset.entities_with_modality("pca", side="drug") == frozenset({"d1"})
        assert _dataset().entities_with_modality("pathway_features") == frozenset()

    def test_drug_graph_membership_comes_from_uns(self):
        dataset = _dataset(uns={"drug_graphs": {"d1": {"x": np.zeros(1)}}})

        assert dataset.entities_with_modality("drug_graph", side="drug") == frozenset({"d1"})
        assert _dataset().entities_with_modality("drug_graph", side="drug") == frozenset()

    def test_an_absent_drug_view_raises(self):
        with pytest.raises(KeyError, match="Drug feature 'absent' not found"):
            _dataset().entities_with_modality("absent", side="drug")


INDEX = pd.Index(["a", "b", "c"])
DATA = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def _fetch(ids: list[str], *, strict: bool = False) -> np.ndarray:
    return _aligned_fetch(INDEX, np.array(ids), DATA, strict=strict, entity_label="sample")


class TestAlignment:
    def test_rows_follow_the_requested_order(self):
        result = _fetch(["c", "a"])

        np.testing.assert_array_equal(result, np.array([[5.0, 6.0], [1.0, 2.0]], dtype=np.float32))

    def test_a_repeated_id_yields_a_repeated_row(self):
        result = _fetch(["b", "b"])

        np.testing.assert_array_equal(result[0], result[1])

    def test_the_result_is_float32(self):
        assert _fetch(["a"]).dtype == np.float32

    def test_an_empty_request_keeps_the_feature_width(self):
        result = _fetch([])

        assert result.shape == (0, DATA.shape[1])


class TestMissingIds:
    def test_missing_ids_become_nan_rows(self):
        result = _fetch(["a", "absent"])

        assert not np.isnan(result[0]).any()
        assert np.isnan(result[1]).all()

    def test_an_all_missing_request_is_entirely_nan(self):
        result = _fetch(["absent1", "absent2"])

        assert np.isnan(result).all()

    def test_missing_ids_are_logged_once(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger="drevalpy.types.data.dataset_utils.aligned_fetch"):
            _fetch(["a", "absent"])

        assert "1 of 2 sample IDs not found" in caplog.text

    def test_the_warning_previews_at_most_five_ids(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger="drevalpy.types.data.dataset_utils.aligned_fetch"):
            _fetch([f"absent{i}" for i in range(7)])

        assert caplog.text.count("absent") == 5

    def test_no_warning_is_logged_when_every_id_is_present(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger="drevalpy.types.data.dataset_utils.aligned_fetch"):
            _fetch(["a", "b"])

        assert caplog.text == ""


class TestStrictMode:
    def test_strict_mode_raises_for_a_missing_id(self):
        with pytest.raises(KeyError, match="1 of 2 sample IDs not found"):
            _fetch(["a", "absent"], strict=True)

    def test_strict_mode_passes_when_every_id_is_present(self):
        result = _fetch(["a", "b"], strict=True)

        assert result.shape == (2, 2)

    def test_the_entity_label_appears_in_the_error(self):
        with pytest.raises(KeyError, match="drug IDs not found"):
            _aligned_fetch(INDEX, np.array(["absent"]), DATA, strict=True, entity_label="drug")


class TestSparseInput:
    @pytest.mark.parametrize(
        "make_sparse",
        [
            pytest.param(sparse.csr_matrix, id="csr"),
            pytest.param(sparse.csc_matrix, id="csc"),
            pytest.param(sparse.coo_matrix, id="coo"),
        ],
    )
    def test_sparse_matrices_are_densified(self, make_sparse):
        dense = np.array([[1.0, 0.0], [0.0, 2.0]])

        result = to_dense(make_sparse(dense))

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, dense)

    def test_densification_preserves_the_shape(self):
        result = to_dense(sparse.csr_matrix((3, 5), dtype=np.float32))

        assert result.shape == (3, 5)

    def test_implicit_zeros_materialize(self):
        matrix = sparse.csr_matrix(([1.0], ([0], [0])), shape=(2, 2))

        assert to_dense(matrix).sum() == 1.0


class TestDenseInput:
    def test_a_dense_array_is_returned_unchanged(self):
        array = np.array([[1.0, 2.0]])

        assert to_dense(array) is array

    def test_a_dataframe_is_returned_unchanged(self):
        frame = pd.DataFrame({"a": [1.0]})

        assert to_dense(frame) is frame

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(None, id="none"),
            pytest.param([[1.0]], id="nested-list"),
            pytest.param(3.5, id="scalar"),
        ],
    )
    def test_objects_without_toarray_pass_through(self, value):
        assert to_dense(value) is value


class TestDuckTyping:
    def test_any_object_exposing_toarray_is_densified(self):
        class FakeSparse:
            def toarray(self) -> np.ndarray:
                return np.array([[7.0]])

        np.testing.assert_array_equal(to_dense(FakeSparse()), np.array([[7.0]]))

    def test_a_non_callable_toarray_attribute_is_ignored(self):
        class NotSparse:
            toarray = "not callable"

        instance = NotSparse()

        assert to_dense(instance) is instance


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    ds = synthetic(n_samples=2, n_drugs=2)
    ds.mdata.obs["study"] = ["A", "B"]
    ds.mdata.obs["score"] = [0.5, 1.5]
    return ds


class TestSampleColumn:
    def test_returns_values_in_the_order_of_the_requested_ids(self, dataset: Dataset) -> None:
        ids = dataset.sample_ids[::-1]

        assert dataset.sample_column("study", ids).tolist() == ["B", "A"]

    def test_unknown_ids_give_nan(self, dataset: Dataset) -> None:
        values = dataset.sample_column("study", np.array([dataset.sample_ids[0], "NOPE"]))

        assert values[0] == "A"
        assert np.isnan(values[1])

    def test_a_numeric_column_stays_numeric_with_unknown_ids(self, dataset: Dataset) -> None:
        values = dataset.sample_column("score", np.array([dataset.sample_ids[1], "NOPE"]))

        assert values.dtype == np.float64
        assert values[0] == 1.5
        assert np.isnan(values[1])

    def test_missing_column_names_the_available_ones(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError, match=r"'nope'.*Available:.*tissue.*study"):
            dataset.sample_column("nope", dataset.sample_ids)

    def test_get_tissue_delegates_to_the_tissue_column(self, dataset: Dataset) -> None:
        ids = np.array([*dataset.sample_ids, "NOPE"])

        expected = dataset.sample_column("tissue", ids)
        actual = dataset.get_tissue(ids)

        assert actual[:-1].tolist() == expected[:-1].tolist()
        assert np.isnan(actual[-1])

    def test_dataset_satisfies_the_protocol_with_it(self, dataset: Dataset) -> None:
        assert isinstance(dataset, MuDataLike)
        assert hasattr(MuDataLike, "sample_column")
