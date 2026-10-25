"""Tests for ``Dataset`` against a synthetic dataset written to and read from disk."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.data import split
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.types.data.dataset import Dataset
from tests.synthetic import (
    BPE_LENGTH,
    BUILTIN_MEASURE,
    CHEMBERTA_DIM,
    CNV_MODALITY,
    FINGERPRINT_BITS,
    N_DRUGS,
    N_GENES,
    N_PATHWAYS,
    N_SAMPLES,
    OMICS_MODALITIES,
    build_synthetic_dataset,
)


@pytest.fixture(scope="module")
def h5mu_path(tmp_path_factory: pytest.TempPathFactory) -> UPath:
    path = UPath(tmp_path_factory.mktemp("dataset")) / "synthetic.h5mu"
    build_synthetic_dataset().save(path)
    return path


@pytest.fixture()
def dataset(h5mu_path: UPath) -> Dataset:
    return Dataset.load(h5mu_path)


class TestRoundTrip:
    def test_identity_and_structure_survive(self, dataset: Dataset):
        assert dataset.name == build_synthetic_dataset().name
        assert set(dataset.mdata.mod) == {"response", *OMICS_MODALITIES}
        assert CNV_MODALITY == "copy_number_variation"
        assert "Samples:" in repr(dataset)

    @pytest.mark.parametrize(("attr", "n"), [("sample_ids", N_SAMPLES), ("drug_ids", N_DRUGS)])
    def test_ids(self, dataset: Dataset, attr: str, n: int):
        ids = getattr(dataset, attr)
        assert ids.ndim == 1
        assert len(ids) == n
        assert ids.dtype.kind in ("U", "O")


class TestRandomizationRoundTrip:
    pytestmark = pytest.mark.slow

    def test_label_and_provenance_survive(self, tmp_path):
        provenance = {"root_seed": 42, "effective_seed": 3, "type": "permutation", "views": ["gene_expression"]}
        randomized = build_synthetic_dataset().with_randomized_views(
            ["gene_expression"], random_state=3, randomization=("SVRC", "gene_expression")
        )
        randomized.mdata.uns["randomization_provenance"] = dict(provenance)
        randomized.save(UPath(tmp_path) / "randomized.h5mu")

        reloaded = Dataset.load(UPath(tmp_path) / "randomized.h5mu")

        assert reloaded.randomization == ("SVRC", "gene_expression")
        assert reloaded.mdata.uns["randomization_provenance"] == provenance
        assert isinstance(reloaded.mdata.uns["randomization_provenance"]["views"], list)

    def test_an_unrandomized_dataset_reads_back_without_a_label(self, dataset: Dataset):
        assert dataset.randomization is None
        assert "randomization_provenance" not in dataset.mdata.uns


class TestIdentity:
    def test_name_and_randomization_are_read_from_uns(self, dataset: Dataset):
        dataset.randomization = ("SVRC", "gene_expression")

        assert dataset.name == dataset.mdata.uns["dataset_name"]
        assert list(dataset.mdata.uns["randomization"]) == ["SVRC", "gene_expression"]
        assert "SVRC (gene_expression)" in repr(dataset)

        dataset.randomization = None

        assert "randomization" not in dataset.mdata.uns

    def test_save_creates_the_parent_directory_and_leaves_the_dataset_untouched(self, dataset: Dataset, tmp_path):
        keys_before = set(dataset.mdata.uns)

        dataset.save(UPath(tmp_path) / "nested" / "dir" / "copy.h5mu")

        assert (UPath(tmp_path) / "nested" / "dir" / "copy.h5mu").exists()
        assert set(dataset.mdata.uns) == keys_before


class TestResponse:
    def test_response_matrix(self, dataset: Dataset):
        mat = dataset.response_matrix
        assert mat.shape == (N_SAMPLES, N_DRUGS)
        assert mat.dtype == np.float32
        assert np.isnan(mat).any()
        assert not np.isnan(mat).all()

    def test_response_layers(self, dataset: Dataset):
        auc = dataset.get_response_layer("AUC")
        assert auc.shape == (N_SAMPLES, N_DRUGS)
        layer = dataset.get_response_layer(BUILTIN_MEASURE)
        np.testing.assert_array_equal(np.isnan(layer), np.isnan(dataset.response_matrix))
        with pytest.raises(KeyError, match="nonexistent"):
            dataset.get_response_layer("nonexistent")


class TestFeatures:
    @pytest.mark.parametrize(
        ("modality", "n_ids", "dim"),
        [("gene_expression", 5, N_GENES), ("pathway_features", 3, N_PATHWAYS)],
    )
    def test_sample_features(self, dataset: Dataset, modality: str, n_ids: int, dim: int):
        features = dataset.get_sample_features(modality, dataset.sample_ids[:n_ids])
        assert features.shape == (n_ids, dim)
        assert features.dtype == np.float32

    def test_missing_ids_get_nan(self, dataset: Dataset):
        features = dataset.get_sample_features("gene_expression", np.array(["FAKE_ID_1", "FAKE_ID_2"]))
        assert features.shape == (2, N_GENES)
        assert np.all(np.isnan(features))

    def test_gene_names(self, dataset: Dataset):
        names = dataset.get_sample_feature_names("gene_expression")
        assert names is not None
        assert len(names) == N_GENES

    @pytest.mark.parametrize(
        ("view", "dim"),
        [("chemberta", CHEMBERTA_DIM), ("morgan_fingerprint", FINGERPRINT_BITS), ("bpe_smiles", BPE_LENGTH)],
    )
    def test_drug_features(self, dataset: Dataset, view: str, dim: int):
        features = dataset.get_drug_features(view, dataset.drug_ids[:4])
        assert features.shape == (4, dim)
        assert features.dtype == np.float32

    def test_unknown_views_raise(self, dataset: Dataset):
        with pytest.raises(KeyError, match="nonexistent"):
            dataset.get_sample_features("nonexistent", dataset.sample_ids[:1])
        with pytest.raises(KeyError, match="nonexistent"):
            dataset.get_drug_features("nonexistent", dataset.drug_ids[:1])

    def test_canonical_smiles_is_the_single_raw_drug_view(self, dataset: Dataset):
        assert dataset.response.var["canonical_smiles"].notna().all()

    def test_get_drug_graphs(self, dataset: Dataset):
        graphs = dataset.get_drug_graphs(dataset.drug_ids[:3])
        assert len(graphs) == 3
        assert all({"x", "edge_index", "edge_attr"} <= set(g) for g in graphs)


class TestMetadata:
    def test_sample_meta_and_tissue(self, dataset: Dataset):
        meta = dataset.sample_meta
        assert "tissue" in meta.columns
        assert "cell_line_name" not in meta.columns  # a dataset without the optional cell-line columns
        assert len(dataset.get_tissue(dataset.sample_ids[:5])) == 5
        assert len(dataset.get_tissue(np.array(["FAKE_ID"]))) == 1
        assert len(np.unique(dataset.get_tissue(dataset.sample_ids))) >= 3

    def test_get_uns(self, dataset: Dataset):
        assert isinstance(dataset.get_uns("bpe_codes"), str)
        gmt = dataset.get_uns("pathways_gmt")
        assert all("\t" in line for line in gmt.strip().splitlines())
        assert set(dataset.get_uns("sparsego")) == {"gene2ind", "ontology"}
        with pytest.raises(KeyError, match="nonexistent"):
            dataset.get_uns("nonexistent")


class TestSubsetting:
    def test_subsets(self, dataset: Dataset):
        sub = dataset.subset_samples(dataset.sample_ids[:10])
        assert len(sub.sample_ids) == 10
        assert sub.response_matrix.shape == (10, N_DRUGS)
        assert "drug_graphs" in sub.mdata.uns
        sub = dataset.subset_drugs(dataset.drug_ids[:5])
        assert len(sub.drug_ids) == 5
        assert sub.response_matrix.shape == (N_SAMPLES, 5)

    def test_subset_drugs_filters_drug_graphs(self, dataset: Dataset):
        retained = dataset.drug_ids[:1]
        sub = dataset.subset_drugs(retained)

        assert set(sub.mdata.uns["drug_graphs"]) == set(map(str, retained))
        assert sub.get_drug_graphs(np.array([dataset.drug_ids[1]])) == [None]

    @pytest.mark.parametrize("method", ["subset_samples", "subset_drugs"])
    def test_subset_does_not_alias_drug_graphs_and_keeps_randomization(self, dataset: Dataset, method: str):
        dataset.randomization = ("permutation", "gene_expression")
        ids = dataset.sample_ids[:1] if method == "subset_samples" else dataset.drug_ids[:1]
        sub = getattr(dataset, method)(ids)
        drug_id = str(sub.drug_ids[0])
        parent_value = dataset.mdata.uns["drug_graphs"][drug_id]["x"][0].copy()

        sub.mdata.uns["drug_graphs"][drug_id]["x"][0] = parent_value + 1

        np.testing.assert_array_equal(dataset.mdata.uns["drug_graphs"][drug_id]["x"][0], parent_value)
        assert sub.randomization == dataset.randomization


class TestTissueOnlyObs:
    def test_obs_has_only_tissue_and_an_unnamed_index(self, dataset: Dataset):
        assert list(dataset.sample_meta.columns) == ["tissue"]
        assert dataset.sample_meta.index.name is None

    @pytest.mark.parametrize("mode", ["LSO", "LTO"])
    def test_sample_and_tissue_out_splits_run(self, dataset: Dataset, mode: str):
        folds = split(dataset, mode=mode, n_splits=3)

        assert len(folds) == 3
        assert all(fold.train.mask.any() and fold.test.mask.any() for fold in folds)

    def test_a_naive_baseline_trains_and_predicts(self, dataset: Dataset):
        fold = split(dataset, mode="LTO", n_splits=3)[0]
        model = DRPModel(ModelSpec.from_name("NaiveMeanEffectsPredictor"))

        model.train(dataset, fold.train)
        predictions = model.predict(dataset, fold.test)

        assert predictions.shape == (int(fold.test.mask.sum()),)
        assert np.isfinite(predictions).any()
