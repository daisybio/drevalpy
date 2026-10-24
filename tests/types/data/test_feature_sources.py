"""Reading features: feature sources, omics modality naming, the MuDataLike protocol and the pair loader."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.types.data import mudatalike
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import (
    DrugFeatureSource,
    FeatureSource,
    SampleFeatureSource,
)
from drevalpy.types.data.modalities import (
    OMICS_ACCESSORS,
    backing_modality,
    public_omics_name,
    resolve_omics_accessor,
)
from drevalpy.types.data.mudatalike import MuDataLike
from drevalpy.types.data.tensor_data import IndexedPairDataset, make_pair_loader
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints
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


class TestAccessorMap:
    def test_map_is_read_only(self):
        with pytest.raises(TypeError):
            OMICS_ACCESSORS["gene_expression"] = "something_else"  # type: ignore[index]

    def test_map_is_injective(self):
        assert len(set(OMICS_ACCESSORS.values())) == len(OMICS_ACCESSORS)

    def test_copy_number_is_the_only_non_identity_entry(self):
        non_identity = {public for public, accessor in OMICS_ACCESSORS.items() if public != accessor}

        assert non_identity == {"copy_number_variation_gistic"}


class TestResolveOmicsAccessor:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param("gene_expression", "gene_expression", id="identity-entry"),
            pytest.param("copy_number_variation_gistic", "copy_number_variation", id="suffixed-name"),
            pytest.param("pathway_features", "pathway_features", id="non-omics-view"),
            pytest.param("", "", id="empty-string"),
        ],
    )
    def test_public_names_resolve_to_stored_keys(self, name, expected):
        assert resolve_omics_accessor(name) == expected


class TestPublicOmicsName:
    @pytest.mark.parametrize(
        ("accessor", "expected"),
        [
            pytest.param("copy_number_variation", "copy_number_variation_gistic", id="stored-key"),
            pytest.param("methylation", "methylation", id="identity-entry"),
            pytest.param("custom_matrix", "custom_matrix", id="non-omics-view"),
        ],
    )
    def test_stored_keys_resolve_back_to_public_names(self, accessor, expected):
        assert public_omics_name(accessor) == expected

    def test_round_trip_is_lossless_for_every_registered_view(self):
        assert all(public_omics_name(resolve_omics_accessor(name)) == name for name in OMICS_ACCESSORS)


class TestBackingModality:
    def test_the_name_as_written_wins_over_the_accessor_map(self):
        """A file already carrying the suffixed name is read directly, with no rename."""
        available = {"copy_number_variation_gistic", "copy_number_variation"}

        assert backing_modality("copy_number_variation_gistic", available) == "copy_number_variation_gistic"

    def test_the_accessor_map_is_the_fallback(self):
        assert backing_modality("copy_number_variation_gistic", {"copy_number_variation"}) == "copy_number_variation"

    def test_a_view_absent_from_the_file_is_unbacked(self):
        assert backing_modality("proteomics", {"gene_expression"}) is None

    def test_a_non_omics_view_is_matched_by_name_only(self):
        assert backing_modality("pathway_features", {"pathway_features"}) == "pathway_features"

    def test_an_unbacked_non_omics_view_is_none(self):
        assert backing_modality("pathway_features", set()) is None

    def test_any_container_of_keys_is_accepted(self):
        """Callers pass ``mdata.mod``, sets and lists interchangeably."""
        assert backing_modality("methylation", ["methylation"]) == "methylation"


class _MinimalDataset:
    @property
    def sample_ids(self) -> np.ndarray:
        return np.array(["cl1", "cl2"])

    @property
    def drug_ids(self) -> np.ndarray:
        return np.array(["d1"])

    @property
    def response_matrix(self) -> np.ndarray:
        return np.array([[1.0], [np.nan]])

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        return np.array(["Lung"] * len(ids))

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        return self.get_tissue(ids)

    def response_layer_names(self) -> list[str]:
        return ["relevance_score"]

    def get_response_layer(self, name: str) -> np.ndarray:
        return np.array([[9.0], [np.nan]])


class _MissingGetTissue(_MinimalDataset):
    get_tissue = None  # type: ignore[assignment]


class _MissingGetResponseLayer(_MinimalDataset):
    get_response_layer = None  # type: ignore[assignment]


class TestProtocolConformance:
    def test_dataset_satisfies_the_protocol(self):
        assert isinstance(synthetic_mudataset_gene_expression_fingerprints(), MuDataLike)
        assert MuDataLike in Dataset.__mro__

    def test_a_hand_built_stand_in_satisfies_the_protocol(self):
        dataset = _MinimalDataset()

        assert isinstance(dataset, MuDataLike)
        assert np.isnan(dataset.response_matrix).sum() == 1
        assert len(dataset.get_tissue(dataset.sample_ids)) == 2

    @pytest.mark.parametrize("candidate", [_MissingGetTissue(), _MissingGetResponseLayer(), object()])
    def test_an_object_missing_a_member_does_not_satisfy_the_protocol(self, candidate):
        assert not isinstance(candidate, MuDataLike)


class TestProtocolLimits:
    def test_the_protocol_cannot_be_instantiated(self):
        construct = getattr(mudatalike, "MuDataLike")  # noqa: B009 - defeats static resolution on purpose

        with pytest.raises(TypeError):
            construct()

    def test_isinstance_does_not_check_signatures(self):
        class WrongSignatures:
            sample_ids = "not an array"
            drug_ids = "not an array"
            response_matrix = "not an array"
            response_layer_names = "not a method"

            def get_tissue(self) -> None:
                return None

            def sample_column(self) -> None:
                return None

            def get_response_layer(self) -> None:
                return None

        assert isinstance(WrongSignatures(), MuDataLike)


SAMPLE_FEATURES = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
DRUG_FEATURES = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
SAMPLE_PAIR_IDX = np.array([0, 1, 0, 1])
DRUG_PAIR_IDX = np.array([0, 1, 2, 0])
RESPONSE = np.array([1.0, 2.0, 3.0, 4.0])


class TestIndexedPairDataset:
    def test_length(self):
        assert len(IndexedPairDataset((SAMPLE_FEATURES, SAMPLE_PAIR_IDX))) == 4
        assert len(IndexedPairDataset()) == 0

    def test_getitem_reads_through_to_the_entity_row(self):
        dataset = IndexedPairDataset((SAMPLE_FEATURES, SAMPLE_PAIR_IDX), (DRUG_FEATURES, DRUG_PAIR_IDX))

        assert len(dataset[0]) == 2
        torch.testing.assert_close(dataset[1][0], torch.tensor([0.3, 0.4]))
        torch.testing.assert_close(dataset[0][0], dataset[2][0])

    def test_response_is_appended_as_a_trailing_scalar(self):
        item = IndexedPairDataset((SAMPLE_FEATURES, SAMPLE_PAIR_IDX), response=RESPONSE)[2]

        assert len(item) == 2
        torch.testing.assert_close(item[-1], torch.tensor(3.0))
        assert item[-1].dtype is torch.float32

    def test_features_are_cast_to_float32(self):
        dataset = IndexedPairDataset((SAMPLE_FEATURES.astype(np.float64), SAMPLE_PAIR_IDX))

        assert dataset[0][0].dtype is torch.float32

    def test_pair_count_follows_the_first_spec_and_a_shorter_spec_fails_loudly(self):
        dataset = IndexedPairDataset((SAMPLE_FEATURES, SAMPLE_PAIR_IDX), (DRUG_FEATURES, DRUG_PAIR_IDX[:2]))

        assert len(dataset) == len(SAMPLE_PAIR_IDX)
        with pytest.raises(IndexError):
            dataset[3]


class TestMakePairLoader:
    def test_batches_stack_the_entity_rows_in_pair_order(self):
        loader = make_pair_loader(
            (SAMPLE_FEATURES, SAMPLE_PAIR_IDX),
            (DRUG_FEATURES, DRUG_PAIR_IDX),
            response=RESPONSE,
            batch_size=4,
            shuffle=False,
        )

        samples, drugs, response = next(iter(loader))

        assert samples.shape == (4, 2)
        assert drugs.shape == (4, 1)
        torch.testing.assert_close(response, torch.tensor(RESPONSE, dtype=torch.float32))

    @pytest.mark.parametrize(("drop_last", "sizes"), [(True, [3]), (False, [3, 1])])
    def test_drop_last(self, drop_last, sizes):
        loader = make_pair_loader(
            (SAMPLE_FEATURES, SAMPLE_PAIR_IDX), response=RESPONSE, batch_size=3, shuffle=False, drop_last=drop_last
        )

        assert [batch[-1].shape[0] for batch in loader] == sizes

    def test_loader_without_a_response_yields_features_only(self):
        loader = make_pair_loader((SAMPLE_FEATURES, SAMPLE_PAIR_IDX), batch_size=4, shuffle=False)

        assert len(next(iter(loader))) == 1
