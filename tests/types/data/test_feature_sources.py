"""Reading features through the sample and drug feature sources."""

from __future__ import annotations

import json

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import FeatureSource, store_variant
from tests.conftest import MockFeatureSource
from tests.synthetic import CHEMBERTA_DIM, N_GENES


@pytest.fixture()
def sample_source(synthetic_dataset: Dataset) -> FeatureSource:
    return FeatureSource(synthetic_dataset, "sample", synthetic_dataset.sample_ids[:5])


@pytest.fixture()
def drug_source(synthetic_dataset: Dataset) -> FeatureSource:
    return FeatureSource(synthetic_dataset, "drug", synthetic_dataset.drug_ids)


class TestSharedBase:
    def test_an_unknown_side_is_rejected(self, synthetic_dataset: Dataset):
        with pytest.raises(ValueError, match="Unknown feature source side 'cell'"):
            FeatureSource(synthetic_dataset, "cell", np.array(["x"]))  # type: ignore[arg-type]

    def test_identifiers_are_coerced_to_strings_in_the_requested_order(self, synthetic_dataset: Dataset):
        source = FeatureSource(synthetic_dataset, "sample", np.array([1, 2]))
        assert source.identifiers.dtype.kind == "U"
        assert source.identifiers.tolist() == ["1", "2"]

        ids = synthetic_dataset.sample_ids[[3, 0, 1]]
        assert FeatureSource(synthetic_dataset, "sample", ids).identifiers.tolist() == list(ids)

    def test_mdata_and_metadata(self, synthetic_dataset: Dataset, sample_source: FeatureSource):
        assert sample_source.mdata is synthetic_dataset.mdata
        assert set(sample_source.get_metadata("sparsego")) == {"gene2ind", "ontology"}
        with pytest.raises(KeyError, match="nonexistent"):
            sample_source.get_metadata("nonexistent")


class TestSampleSideFeatureSource:
    def test_view_matrix_and_feature_names(self, sample_source: FeatureSource):
        assert sample_source.get_view_matrix("gene_expression", sample_source.identifiers).shape == (5, N_GENES)
        assert len(sample_source.get_feature_names("gene_expression") or ()) == N_GENES
        assert sample_source.get_feature_names("absent_view") is None

    def test_entity_view_special_cases_tissue(self, synthetic_dataset: Dataset, sample_source: FeatureSource):
        entity_id = str(sample_source.identifiers[0])

        assert (
            sample_source.get_entity_view(entity_id, "tissue") == synthetic_dataset.get_tissue(np.array([entity_id]))[0]
        )
        assert np.isnan(sample_source.get_entity_view("NOT_A_SAMPLE", "tissue"))

    def test_tissue_is_none_when_the_dataset_has_no_tissue_column(self, synthetic_dataset: Dataset):
        untissued = synthetic_dataset.subset_samples(synthetic_dataset.sample_ids)
        del untissued.mdata.obs["tissue"]
        source = FeatureSource(untissued, "sample", untissued.sample_ids[:3])

        assert source.get_entity_view(str(source.identifiers[0]), "tissue") is None

    def test_entity_view_row_matches_the_matrix_row(self, sample_source: FeatureSource):
        entity_id = str(sample_source.identifiers[2])

        row = sample_source.get_entity_view(entity_id, "gene_expression")

        assert row.shape == (N_GENES,)
        np.testing.assert_array_equal(row, sample_source.get_view_matrix("gene_expression", np.array([entity_id]))[0])


class TestDrugSideFeatureSource:
    def test_view_matrix_and_feature_names(self, drug_source: FeatureSource):
        assert drug_source.get_view_matrix("chemberta", drug_source.identifiers[:3]).shape == (3, CHEMBERTA_DIM)
        assert len(drug_source.get_feature_names("chemberta") or ()) == CHEMBERTA_DIM
        assert drug_source.get_feature_names("absent_view") is None

    def test_entity_view_special_cases_drug_graph(self, drug_source: FeatureSource):
        graph = drug_source.get_entity_view(str(drug_source.identifiers[0]), "drug_graph")

        assert graph is not None
        assert set(graph) >= {"x", "edge_index", "edge_attr"}
        assert drug_source.get_entity_view("NOT_A_DRUG", "drug_graph") is None

    def test_entity_view_returns_a_varm_row_and_tolerates_a_missing_optional_view(self, drug_source: FeatureSource):
        drug_id = str(drug_source.identifiers[0])

        assert drug_source.get_entity_view(drug_id, "chemberta").shape == (CHEMBERTA_DIM,)
        assert drug_source.get_entity_view(drug_id, "molgnet_features") is None


class TestHasView:
    @pytest.mark.parametrize(
        ("view", "expected"),
        [
            ("gene_expression", True),
            ("copy_number_variation_gistic", True),
            ("pathway_features", True),
            ("tissue", True),
            ("chemberta", False),
            ("absent_view", False),
        ],
    )
    def test_sample_views(self, sample_source: FeatureSource, view: str, expected: bool):
        assert sample_source.has_view(view) is expected

    @pytest.mark.parametrize(
        ("view", "expected"),
        [
            ("chemberta", True),
            ("morgan_fingerprint", True),
            ("canonical_smiles", True),
            ("drug_graph", True),
            ("gene_expression", False),
            ("molgnet_features", False),
        ],
    )
    def test_drug_views(self, drug_source: FeatureSource, view: str, expected: bool):
        assert drug_source.has_view(view) is expected

    def test_default_reports_views_with_feature_names(self):
        source = MockFeatureSource({"a": {"v": [1.0]}}, {"v": ["f0"]})
        assert source.has_view("v")
        assert not source.has_view("w")

    def test_sides(self, sample_source: FeatureSource, drug_source: FeatureSource):
        assert sample_source.side == "sample"
        assert drug_source.side == "drug"


_SAMPLES = np.array(["s1", "s2"])
_DRUGS = np.array(["d1", "d2"])


def _stored_dataset() -> Dataset:
    response = ad.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame(index=_SAMPLES),
        var=pd.DataFrame(index=_DRUGS),
    )
    response.obsm["pca_0"] = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    response.varm["fingerprints_0"] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    learned = ad.AnnData(
        X=np.array([[5.0], [6.0]], dtype=np.float32),
        obs=pd.DataFrame(index=_SAMPLES),
        var=pd.DataFrame(index=["f0"]),
    )
    mdata = md.MuData({"response": response, "encoder_0": learned})
    mdata.uns["sample_featurizer_variants"] = json.dumps(
        {"pca": {"pca_0": {"n_components": 2}}, "encoder": {"encoder_0": {}}, "lost": {"lost_0": {}}}
    )
    mdata.uns["drug_featurizer_variants"] = {"fingerprints": {"fingerprints_0": {}}, "lost": {"lost_0": {}}}
    return Dataset(mdata, name="stored")


class TestStored:
    def test_sample_variant_in_obsm_matches_its_hyperparameters(self):
        source = FeatureSource(_stored_dataset(), "sample", _SAMPLES)

        stored = source.stored("pca", {"n_components": 2}, np.array(["s2", "s1", "unknown"]))

        assert stored is not None
        np.testing.assert_array_equal(stored[:2], np.array([[0.3, 0.4], [0.1, 0.2]], dtype=np.float32))
        assert np.isnan(stored[2]).all()
        assert source.stored("pca", {"n_components": 8}, _SAMPLES) is None
        assert source.stored("pca", None, _SAMPLES) is None

    def test_a_modality_variant_is_read_before_the_response_slots(self):
        source = FeatureSource(_stored_dataset(), "sample", _SAMPLES)

        stored = source.stored("encoder", None, _SAMPLES)

        assert stored is not None
        np.testing.assert_array_equal(stored, [[5.0], [6.0]])

    def test_drug_variant_in_varm_from_a_dict_valued_registry(self):
        source = FeatureSource(_stored_dataset(), "drug", _DRUGS)

        stored = source.stored("fingerprints", {}, np.array(["d2"]))

        assert stored is not None
        np.testing.assert_array_equal(stored, [[0.0, 1.0]])

    @pytest.mark.parametrize("side", ["sample", "drug"])
    def test_missing_variants_and_matrices_give_none(self, side):
        source = FeatureSource(_stored_dataset(), side, _SAMPLES)

        assert source.stored("absent", None, _SAMPLES) is None
        assert source.stored("lost", None, _SAMPLES) is None

    def test_a_source_without_variants_or_mudata_gives_none(self, sample_source: FeatureSource):
        assert sample_source.stored("pca", None, sample_source.identifiers) is None
        assert MockFeatureSource({}).stored("pca", None, np.array(["a"])) is None


class TestStoreVariant:
    @pytest.mark.parametrize(("side", "ids", "slot"), [("sample", _SAMPLES, "obsm"), ("drug", _DRUGS, "varm")])
    def test_a_stored_variant_is_read_back_by_its_params(self, side, ids, slot):
        dataset = _stored_dataset()
        matrix = np.array([[1.5], [2.5]], dtype=np.float32)

        key = store_variant(dataset, side, "new[view]:x", matrix, {"k": 3})

        assert key == "new_view_x_0"
        assert key in getattr(dataset.mdata.mod["response"], slot)
        source = FeatureSource(dataset, side, ids)
        np.testing.assert_array_equal(source.stored("new[view]:x", {"k": 3}, ids[::-1]), matrix[::-1])
        assert source.stored("new[view]:x", {"k": 4}, ids) is None

    def test_variants_of_one_key_get_increasing_indices_and_keep_the_registry_json(self):
        dataset = _stored_dataset()
        matrix = np.zeros((2, 1), dtype=np.float32)

        assert store_variant(dataset, "sample", "pca", matrix, {"n_components": 1}) == "pca_1"
        assert store_variant(dataset, "sample", "fresh", matrix) == "fresh_0"

        registry = json.loads(dataset.mdata.uns["sample_featurizer_variants"])
        assert registry["pca"] == {"pca_0": {"n_components": 2}, "pca_1": {"n_components": 1}}
        assert registry["fresh"] == {"fresh_0": {}}
        assert FeatureSource(dataset, "sample", _SAMPLES).stored("fresh", None, _SAMPLES) is not None
