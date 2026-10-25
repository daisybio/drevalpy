"""Reading features through the sample and drug feature sources."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import (
    DrugFeatureSource,
    FeatureSource,
    SampleFeatureSource,
)
from tests.synthetic import CHEMBERTA_DIM, N_GENES


@pytest.fixture()
def sample_source(synthetic_dataset: Dataset) -> SampleFeatureSource:
    return SampleFeatureSource(synthetic_dataset, synthetic_dataset.sample_ids[:5])


@pytest.fixture()
def drug_source(synthetic_dataset: Dataset) -> DrugFeatureSource:
    return DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)


class TestSharedBase:
    def test_feature_source_cannot_be_instantiated(self, synthetic_dataset: Dataset):
        with pytest.raises(TypeError, match="abstract"):
            FeatureSource(synthetic_dataset, np.array(["x"]))  # type: ignore[abstract]

    def test_identifiers_are_coerced_to_strings_in_the_requested_order(self, synthetic_dataset: Dataset):
        source = SampleFeatureSource(synthetic_dataset, np.array([1, 2]))
        assert source.identifiers.dtype.kind == "U"
        assert source.identifiers.tolist() == ["1", "2"]

        ids = synthetic_dataset.sample_ids[[3, 0, 1]]
        assert SampleFeatureSource(synthetic_dataset, ids).identifiers.tolist() == list(ids)

    def test_mdata_and_metadata(self, synthetic_dataset: Dataset, sample_source: SampleFeatureSource):
        assert sample_source.mdata is synthetic_dataset.mdata
        assert set(sample_source.get_metadata("sparsego")) == {"gene2ind", "ontology"}
        with pytest.raises(KeyError, match="nonexistent"):
            sample_source.get_metadata("nonexistent")


class TestSampleFeatureSource:
    def test_view_matrix_and_feature_names(self, sample_source: SampleFeatureSource):
        assert sample_source.get_view_matrix("gene_expression", sample_source.identifiers).shape == (5, N_GENES)
        assert len(sample_source.get_feature_names("gene_expression") or ()) == N_GENES
        assert sample_source.get_feature_names("absent_view") is None

    def test_entity_view_special_cases_tissue(self, synthetic_dataset: Dataset, sample_source: SampleFeatureSource):
        entity_id = str(sample_source.identifiers[0])

        assert (
            sample_source.get_entity_view(entity_id, "tissue") == synthetic_dataset.get_tissue(np.array([entity_id]))[0]
        )
        assert np.isnan(sample_source.get_entity_view("NOT_A_SAMPLE", "tissue"))

    def test_tissue_is_none_when_the_dataset_has_no_tissue_column(self, synthetic_dataset: Dataset):
        untissued = synthetic_dataset.subset_samples(synthetic_dataset.sample_ids)
        del untissued.mdata.obs["tissue"]
        source = SampleFeatureSource(untissued, untissued.sample_ids[:3])

        assert source.get_entity_view(str(source.identifiers[0]), "tissue") is None

    def test_entity_view_row_matches_the_matrix_row(self, sample_source: SampleFeatureSource):
        entity_id = str(sample_source.identifiers[2])

        row = sample_source.get_entity_view(entity_id, "gene_expression")

        assert row.shape == (N_GENES,)
        np.testing.assert_array_equal(row, sample_source.get_view_matrix("gene_expression", np.array([entity_id]))[0])


class TestDrugFeatureSource:
    def test_view_matrix_and_feature_names(self, drug_source: DrugFeatureSource):
        assert drug_source.get_view_matrix("chemberta", drug_source.identifiers[:3]).shape == (3, CHEMBERTA_DIM)
        assert len(drug_source.get_feature_names("chemberta") or ()) == CHEMBERTA_DIM
        assert drug_source.get_feature_names("absent_view") is None

    def test_entity_view_special_cases_drug_graph(self, drug_source: DrugFeatureSource):
        graph = drug_source.get_entity_view(str(drug_source.identifiers[0]), "drug_graph")

        assert graph is not None
        assert set(graph) >= {"x", "edge_index", "edge_attr"}
        assert drug_source.get_entity_view("NOT_A_DRUG", "drug_graph") is None

    def test_entity_view_returns_a_varm_row_and_tolerates_a_missing_optional_view(self, drug_source: DrugFeatureSource):
        drug_id = str(drug_source.identifiers[0])

        assert drug_source.get_entity_view(drug_id, "chemberta").shape == (CHEMBERTA_DIM,)
        assert drug_source.get_entity_view(drug_id, "molgnet_features") is None
