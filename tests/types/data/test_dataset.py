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
from tests.types.data._helpers import stub_featurizer, stub_fit_transform_featurizer


@pytest.fixture(scope="module")
def h5mu_path(tmp_path_factory: pytest.TempPathFactory) -> UPath:
    path = UPath(tmp_path_factory.mktemp("dataset")) / "synthetic.h5mu"
    build_synthetic_dataset().save(path)
    return path


@pytest.fixture()
def mudataset(h5mu_path: UPath) -> Dataset:
    return Dataset.load(h5mu_path)


@pytest.fixture()
def records() -> list[dict]:
    return []


class TestRoundTrip:
    def test_identity_and_structure_survive(self, mudataset: Dataset):
        assert mudataset.name == build_synthetic_dataset().name
        assert set(mudataset.mdata.mod) == {"response", *OMICS_MODALITIES}
        assert CNV_MODALITY == "copy_number_variation"
        assert "Samples:" in repr(mudataset)

    @pytest.mark.parametrize(("attr", "n"), [("sample_ids", N_SAMPLES), ("drug_ids", N_DRUGS)])
    def test_ids(self, mudataset: Dataset, attr: str, n: int):
        ids = getattr(mudataset, attr)
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

    def test_an_unrandomized_dataset_reads_back_without_a_label(self, mudataset: Dataset):
        assert mudataset.randomization is None
        assert "randomization_provenance" not in mudataset.mdata.uns


class TestResponse:
    def test_response_matrix(self, mudataset: Dataset):
        mat = mudataset.response_matrix
        assert mat.shape == (N_SAMPLES, N_DRUGS)
        assert mat.dtype == np.float32
        assert np.isnan(mat).any()
        assert not np.isnan(mat).all()

    def test_response_layers(self, mudataset: Dataset):
        auc = mudataset.get_response_layer("AUC")
        assert auc.shape == (N_SAMPLES, N_DRUGS)
        layer = mudataset.get_response_layer(BUILTIN_MEASURE)
        np.testing.assert_array_equal(np.isnan(layer), np.isnan(mudataset.response_matrix))
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_response_layer("nonexistent")


class TestFeatures:
    @pytest.mark.parametrize(
        ("modality", "n_ids", "dim"),
        [("gene_expression", 5, N_GENES), ("pathway_features", 3, N_PATHWAYS)],
    )
    def test_sample_features(self, mudataset: Dataset, modality: str, n_ids: int, dim: int):
        features = mudataset.get_sample_features(modality, mudataset.sample_ids[:n_ids])
        assert features.shape == (n_ids, dim)
        assert features.dtype == np.float32

    def test_missing_ids_get_nan(self, mudataset: Dataset):
        features = mudataset.get_sample_features("gene_expression", np.array(["FAKE_ID_1", "FAKE_ID_2"]))
        assert features.shape == (2, N_GENES)
        assert np.all(np.isnan(features))

    def test_gene_names(self, mudataset: Dataset):
        names = mudataset.get_sample_feature_names("gene_expression")
        assert names is not None
        assert len(names) == N_GENES

    @pytest.mark.parametrize(
        ("view", "dim"),
        [("chemberta", CHEMBERTA_DIM), ("morgan_fingerprint", FINGERPRINT_BITS), ("bpe_smiles", BPE_LENGTH)],
    )
    def test_drug_features(self, mudataset: Dataset, view: str, dim: int):
        features = mudataset.get_drug_features(view, mudataset.drug_ids[:4])
        assert features.shape == (4, dim)
        assert features.dtype == np.float32

    def test_unknown_views_raise(self, mudataset: Dataset):
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_sample_features("nonexistent", mudataset.sample_ids[:1])
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_drug_features("nonexistent", mudataset.drug_ids[:1])

    def test_canonical_smiles_is_the_single_raw_drug_view(self, mudataset: Dataset):
        assert mudataset.response.var["canonical_smiles"].notna().all()

    def test_get_drug_graphs(self, mudataset: Dataset):
        graphs = mudataset.get_drug_graphs(mudataset.drug_ids[:3])
        assert len(graphs) == 3
        assert all({"x", "edge_index", "edge_attr"} <= set(g) for g in graphs)


class TestMetadata:
    def test_sample_meta_and_tissue(self, mudataset: Dataset):
        meta = mudataset.sample_meta
        assert "tissue" in meta.columns
        assert "cell_line_name" not in meta.columns  # a dataset without the optional cell-line columns
        assert len(mudataset.get_tissue(mudataset.sample_ids[:5])) == 5
        assert len(mudataset.get_tissue(np.array(["FAKE_ID"]))) == 1
        assert len(np.unique(mudataset.get_tissue(mudataset.sample_ids))) >= 3

    def test_get_uns(self, mudataset: Dataset):
        assert isinstance(mudataset.get_uns("bpe_codes"), str)
        gmt = mudataset.get_uns("pathways_gmt")
        assert all("\t" in line for line in gmt.strip().splitlines())
        assert set(mudataset.get_uns("sparsego")) == {"gene2ind", "ontology"}
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_uns("nonexistent")


class TestSubsetting:
    def test_subsets(self, mudataset: Dataset):
        sub = mudataset.subset_samples(mudataset.sample_ids[:10])
        assert len(sub.sample_ids) == 10
        assert sub.response_matrix.shape == (10, N_DRUGS)
        assert "drug_graphs" in sub.mdata.uns
        sub = mudataset.subset_drugs(mudataset.drug_ids[:5])
        assert len(sub.drug_ids) == 5
        assert sub.response_matrix.shape == (N_SAMPLES, 5)

    def test_subset_drugs_filters_drug_graphs(self, mudataset: Dataset):
        retained = mudataset.drug_ids[:1]
        sub = mudataset.subset_drugs(retained)

        assert sub.entities_with_modality("drug_graph", side="drug") == frozenset(map(str, retained))
        assert sub.get_drug_graphs(np.array([mudataset.drug_ids[1]])) == [None]

    @pytest.mark.parametrize("method", ["subset_samples", "subset_drugs"])
    def test_subset_does_not_alias_drug_graphs_and_keeps_randomization(self, mudataset: Dataset, method: str):
        mudataset.randomization = ("permutation", "gene_expression")
        ids = mudataset.sample_ids[:1] if method == "subset_samples" else mudataset.drug_ids[:1]
        sub = getattr(mudataset, method)(ids)
        drug_id = str(sub.drug_ids[0])
        parent_value = mudataset.mdata.uns["drug_graphs"][drug_id]["x"][0].copy()

        sub.mdata.uns["drug_graphs"][drug_id]["x"][0] = parent_value + 1

        np.testing.assert_array_equal(mudataset.mdata.uns["drug_graphs"][drug_id]["x"][0], parent_value)
        assert sub.randomization == mudataset.randomization


class TestPrecompute:
    def test_default_config_is_prepended_to_explicit_configs(self, mudataset: Dataset, records: list[dict]):
        mudataset.precompute(stub_featurizer(records), [{"n_components": 2}])

        assert [call["hyperparameters"] for call in records] == [{}, {"n_components": 2}]

    @pytest.mark.parametrize(("side", "n"), [("sample", N_SAMPLES), ("drug", N_DRUGS)])
    def test_every_entity_of_the_side_is_precomputed(self, mudataset: Dataset, records: list[dict], side, n):
        mudataset.precompute(stub_featurizer(records, side=side), [])

        assert records[0]["n_entities"] == n

    def test_featurizer_without_compute_from_source_is_fitted_first(self, mudataset: Dataset, records: list[dict]):
        mudataset.precompute(stub_fit_transform_featurizer(records), [])

        assert records[0] == {"call": "fit"}
        assert records[1]["shape"] == (N_SAMPLES, 3)

    def test_view_is_forwarded_to_the_featurizer_constructor(self, mudataset: Dataset, records: list[dict]):
        seen: list[dict] = []
        featurizer_cls = stub_featurizer(records)
        original_init = featurizer_cls.__init__

        def spy_init(self, **kwargs):
            seen.append(kwargs)
            original_init(self, **kwargs)

        featurizer_cls.__init__ = spy_init
        mudataset.precompute(featurizer_cls, [], view="methylation")

        assert seen == [{"view": "methylation"}]

    def test_int_hyperparameters_are_drawn(self, mudataset: Dataset, records: list[dict]):
        mudataset.precompute(stub_featurizer(records), 2)

        assert len(records) == 2


class TestPrecomputeSingle:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"precompute": False},
            {"entity_id_only": True},
            {"source_views": ("absent_modality",)},
            {"requires_view": True, "input_views": None},
        ],
        ids=["not-marked", "entity-id-only", "missing-source", "view-without-input-views"],
    )
    def test_ineligible_featurizers_are_skipped(self, mudataset: Dataset, records: list[dict], kwargs):
        mudataset._precompute_single(stub_featurizer(records, **kwargs), 1)

        assert records == []

    def test_eligible_featurizer_is_precomputed(self, mudataset: Dataset, records: list[dict]):
        mudataset._precompute_single(stub_featurizer(records, source_views=("gene_expression",)), 1)

        assert len(records) == 1

    def test_featurizer_errors_are_swallowed(self, mudataset: Dataset, records: list[dict]):
        def raise_value_error(source, entity_ids):
            raise ValueError("no features")

        mudataset._precompute_single(stub_featurizer(records, compute=raise_value_error), 1)

        assert records == []

    def test_precompute_all_visits_every_registered_featurizer(self, mudataset: Dataset, monkeypatch):
        from drevalpy.registry.drug_featurizer import drug_featurizer_registry
        from drevalpy.registry.sample_featurizer import sample_featurizer_registry

        visited: list[str] = []
        monkeypatch.setattr(Dataset, "_precompute_single", lambda self, cls, n_variants: visited.append(cls.__name__))

        mudataset.precompute_all(n_variants=1)

        assert len(visited) == len(sample_featurizer_registry.list_names()) + len(drug_featurizer_registry.list_names())


class TestSourceAvailability:
    @pytest.mark.parametrize(
        ("side", "source", "expected"),
        [
            ("drug", "canonical_smiles", True),
            ("sample", "copy_number_variation_gistic", True),
            ("sample", "absent_modality", False),
            ("drug", "morgan_fingerprint", True),
            ("drug", "absent_view", False),
        ],
    )
    def test_has_source_data(self, mudataset: Dataset, records: list[dict], side, source, expected):
        assert mudataset._has_source_data((source,), stub_featurizer(records, side=side)) is expected


class TestTissueOnlyObs:
    def test_obs_has_only_tissue_and_an_unnamed_index(self, mudataset: Dataset):
        assert list(mudataset.sample_meta.columns) == ["tissue"]
        assert mudataset.sample_meta.index.name is None

    @pytest.mark.parametrize("mode", ["LSO", "LTO"])
    def test_sample_and_tissue_out_splits_run(self, mudataset: Dataset, mode: str):
        folds = split(mudataset, mode=mode, n_splits=3)

        assert len(folds) == 3
        assert all(fold.train.mask.any() and fold.test.mask.any() for fold in folds)

    def test_a_naive_baseline_trains_and_predicts(self, mudataset: Dataset):
        fold = split(mudataset, mode="LTO", n_splits=3)[0]
        model = DRPModel(ModelSpec.from_name("NaiveMeanEffectsPredictor"))

        model.train(mudataset, fold.train)
        predictions = model.predict(mudataset, fold.test)

        assert predictions.shape == (int(fold.test.mask.sum()),)
        assert np.isfinite(predictions).any()
