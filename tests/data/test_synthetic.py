"""Tests for :mod:`drevalpy.data._synthetic`, i.e. :func:`drevalpy.data.synthetic`."""

from __future__ import annotations

import numpy as np
import pytest

import drevalpy
from drevalpy import ModelSpec
from drevalpy.components.featurizers.sample.gene_lists import GENE_LISTS_BY_VIEW, resolve_gene_list, view_gene_union
from drevalpy.data._synthetic import (
    DATASET_NAME,
    MAX_SYNTHETIC_DRUGS,
    MEASURE,
    N_DRUGS,
    N_FEATURES,
    N_SAMPLES,
    _punch_holes,
    synthetic,
)
from drevalpy.data.quality import CurveQuality
from drevalpy.data.utils import TISSUE_IDENTIFIER
from drevalpy.errors import InvalidInputError
from drevalpy.testing import build_synthetic_dataset
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.results import ExperimentResult


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return build_synthetic_dataset()


def _measured(matrix: np.ndarray) -> np.ndarray:
    return ~np.isnan(matrix)


def test_public_names_alias_the_one_builder():
    assert drevalpy.synthetic is drevalpy.data.synthetic is synthetic
    assert build_synthetic_dataset is synthetic


def test_default_shape_name_and_modalities(dataset):
    assert dataset.name == DATASET_NAME
    assert dataset.response_matrix.shape == (N_SAMPLES, N_DRUGS)
    assert sorted(dataset.mdata.mod) == ["gene_expression", "response"]
    assert len(set(dataset.sample_ids)) == N_SAMPLES
    assert len(set(dataset.drug_ids)) == N_DRUGS


def test_name_and_shape_are_configurable():
    small = build_synthetic_dataset(name="MYDATA", n_samples=6, n_drugs=3, n_tissues=3)

    assert small.name == "MYDATA"
    assert small.response_matrix.shape == (6, 3)
    assert len(set(small.response.obs[TISSUE_IDENTIFIER])) == 3


@pytest.mark.parametrize("omics", [(), None])
def test_no_omics_leaves_only_the_response(omics):
    assert list(build_synthetic_dataset(omics=omics).mdata.mod) == ["response"]


def test_metadata_layout(dataset):
    obs = dataset.response.obs
    tissues = dataset.get_tissue(np.asarray(dataset.sample_ids))

    assert list(obs.columns) == [TISSUE_IDENTIFIER]
    assert obs.index.name is None
    assert obs.index[0] == "SYNTH-000"
    assert len(tissues) == N_SAMPLES
    assert all(isinstance(tissue, str) for tissue in tissues)
    np.testing.assert_array_equal(np.isnan(dataset.get_response_layer(MEASURE)), np.isnan(dataset.response_matrix))
    assert dataset.response.var["drug_name"].notna().all()


def test_every_drug_has_a_distinct_parseable_canonical_smiles():
    from rdkit import Chem

    full = synthetic(n_drugs=MAX_SYNTHETIC_DRUGS)
    smiles = full.response.var["canonical_smiles"]

    assert MAX_SYNTHETIC_DRUGS >= 8
    assert full.response_matrix.shape[1] == MAX_SYNTHETIC_DRUGS
    assert smiles.nunique() == MAX_SYNTHETIC_DRUGS
    assert all(Chem.MolFromSmiles(value) is not None for value in smiles)


def test_one_drug_above_the_limit_raises_naming_it():
    with pytest.raises(InvalidInputError, match=str(MAX_SYNTHETIC_DRUGS)):
        synthetic(n_drugs=MAX_SYNTHETIC_DRUGS + 1)


def test_some_pairs_are_unmeasured_but_every_sample_and_drug_keeps_a_measurement(dataset):
    measured = _measured(dataset.response_matrix)

    assert not measured.all()
    assert measured.any(axis=1).all()
    assert measured.any(axis=0).all()


def test_missing_fraction_controls_how_many_pairs_are_removed():
    complete = build_synthetic_dataset(missing_fraction=0.0)
    sparse = build_synthetic_dataset(missing_fraction=0.4)
    dense = build_synthetic_dataset(missing_fraction=0.05)
    extreme = _measured(build_synthetic_dataset(missing_fraction=1.0).response_matrix)

    assert not np.isnan(complete.response_matrix).any()
    assert np.isnan(sparse.response_matrix).sum() > np.isnan(dense.response_matrix).sum()
    assert extreme.any(axis=1).all()
    assert extreme.any(axis=0).all()


def test_punch_holes_leaves_a_degenerate_matrix_alone_and_modifies_in_place():
    degenerate = np.ones((1, 4), dtype=np.float32)
    matrix = np.ones((6, 6), dtype=np.float32)

    _punch_holes(degenerate, np.random.default_rng(0), 0.5)
    _punch_holes(matrix, np.random.default_rng(0), 0.2)

    assert not np.isnan(degenerate).any()
    assert np.isnan(matrix).any()


def test_omics_modalities_and_feature_widths():
    both = build_synthetic_dataset(omics=["gene_expression", "proteomics"])
    named = build_synthetic_dataset(omics=["gene_expression"], feature_names=["A", "B", "C"])
    narrow = build_synthetic_dataset(omics=["gene_expression"], n_features=3)
    ids = np.asarray(both.sample_ids)

    assert set(both.mdata.mod) == {"response", "gene_expression", "proteomics"}
    assert both.get_sample_features("gene_expression", ids).shape == (
        N_SAMPLES,
        len(view_gene_union("gene_expression")),
    )
    assert named.get_sample_feature_names("gene_expression") == ("A", "B", "C")
    assert named.mdata.mod["gene_expression"].shape[1] == 3
    assert narrow.mdata.mod["gene_expression"].shape[1] == len(view_gene_union("gene_expression"))


def test_a_mapping_sets_partial_coverage_and_uncovered_samples_come_back_as_nan():
    partial = build_synthetic_dataset(omics={"gene_expression": 10})

    matrix = partial.get_sample_features("gene_expression", np.asarray(partial.sample_ids))

    assert partial.mdata.mod["gene_expression"].shape[0] == 10
    assert np.isnan(matrix[10:]).all()
    assert not np.isnan(matrix[:10]).any()


def test_the_gistic_alias_resolves_to_the_stored_modality():
    with_cnv = build_synthetic_dataset(omics=["copy_number_variation_gistic"])

    assert "copy_number_variation" in with_cnv.mdata.mod
    assert with_cnv.get_sample_features("copy_number_variation_gistic", np.asarray([])).size == 0


ALL_OMICS = ("gene_expression", "proteomics", "mutations", "methylation", "copy_number_variation_gistic")


@pytest.mark.slow
class TestOmicsViews:
    @pytest.fixture(scope="class")
    def every_view(self) -> Dataset:
        return synthetic(omics=ALL_OMICS, n_samples=40, n_features=12)

    def _matrix(self, dataset: Dataset, view: str) -> np.ndarray:
        return dataset.get_sample_features(view, np.asarray(dataset.sample_ids))

    def test_every_view_is_stored_under_its_accessor(self, every_view):
        assert set(every_view.mdata.mod) == {
            "response",
            "gene_expression",
            "proteomics",
            "mutations",
            "methylation",
            "copy_number_variation",
        }

    @pytest.mark.parametrize("view", sorted(GENE_LISTS_BY_VIEW))
    def test_every_shipped_list_is_a_subset_of_its_view(self, every_view, view):
        names = set(every_view.get_sample_feature_names(view))

        for list_name in GENE_LISTS_BY_VIEW[view]:
            assert set(resolve_gene_list(list_name)) <= names, list_name

    def test_feature_names(self, every_view):
        landmark = resolve_gene_list("landmark_genes")
        sites = every_view.get_sample_feature_names("methylation")

        assert every_view.get_sample_feature_names("gene_expression")[: len(landmark)] == landmark
        assert len(set(sites)) == len(sites)
        assert all(name.startswith("chr") for name in sites)

    def test_value_ranges(self, every_view):
        mutations = self._matrix(every_view, "mutations")
        methylation = self._matrix(every_view, "methylation")

        assert set(np.unique(mutations)) == {0.0, 1.0}
        assert 0.1 < mutations.mean() < 0.4
        assert 0.0 < methylation.min() and methylation.max() < 1.0
        assert set(np.unique(self._matrix(every_view, "copy_number_variation_gistic"))) <= {-2.0, -1.0, 0.0, 1.0, 2.0}
        assert self._matrix(every_view, "proteomics").min() > 0.0

    def test_a_mapping_of_feature_names_applies_per_view(self):
        dataset = synthetic(omics=["gene_expression", "methylation"], feature_names={"methylation": ["s1", "s2", "s3"]})

        assert dataset.get_sample_feature_names("methylation") == ("s1", "s2", "s3")
        assert len(dataset.get_sample_feature_names("gene_expression")) == len(view_gene_union("gene_expression"))

    def test_the_landmark_featurizer_selects_the_whole_landmark_list(self):
        from drevalpy.components.featurizers.sample.landmark import LandmarkGenesFeaturizer
        from drevalpy.types.data.feature_source import SampleFeatureSource

        built = synthetic()
        ids = np.asarray(built.sample_ids)
        featurizer = LandmarkGenesFeaturizer()

        featurizer.fit(SampleFeatureSource(built, ids), entity_ids=ids)

        assert featurizer.output_dim == len(resolve_gene_list("landmark_genes"))

    def test_more_features_than_the_shipped_lists_pads_with_placeholder_names(self):
        width = len(view_gene_union("proteomics")) + 3

        names = synthetic(omics=["proteomics"], n_features=width).get_sample_feature_names("proteomics")

        assert names is not None
        assert len(names) == width
        assert names[-3:] == ("SYNTHGENE0", "SYNTHGENE1", "SYNTHGENE2")


@pytest.mark.slow
class TestExtras:
    @pytest.fixture(scope="class")
    def full(self) -> Dataset:
        return synthetic(
            n_samples=24,
            n_drugs=8,
            sample_views=["pathway_features"],
            drug_views=["morgan_fingerprint", "chemberta", "bpe_smiles", "smilesvec"],
            uns_payloads=["drug_graphs", "pathways_gmt", "sparsego", "bpe_codes"],
            extra_layers=["AUC", "IC50"],
        )

    def test_nothing_extra_is_added_by_default(self, dataset):
        response = dataset.mdata.mod["response"]

        assert not response.varm
        assert not response.obsm
        assert not {"drug_graphs", "pathways_gmt", "sparsego", "bpe_codes"} & set(dataset.mdata.uns)
        assert "AUC" not in response.layers

    @pytest.mark.parametrize(
        ("view", "width"), [("morgan_fingerprint", 128), ("chemberta", 768), ("bpe_smiles", 128), ("smilesvec", 100)]
    )
    def test_drug_views_have_the_production_width(self, full, view, width):
        features = full.get_drug_features(view, full.drug_ids)

        assert features.shape == (8, width)
        assert features.dtype == np.float32

    def test_the_fingerprint_is_the_real_featurizers_output_and_noise_views_are_seeded(self, full):
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        smiles = full.response.var["canonical_smiles"].iloc[0]
        expected = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128).GetFingerprintAsNumPy(
            Chem.MolFromSmiles(smiles)
        )
        first = synthetic(drug_views=["chemberta"]).get_drug_features("chemberta", np.asarray(["100000"]))
        second = synthetic(drug_views=["chemberta"]).get_drug_features("chemberta", np.asarray(["100000"]))

        np.testing.assert_array_equal(full.get_drug_features("morgan_fingerprint", full.drug_ids)[0], expected)
        np.testing.assert_array_equal(first, second)

    def test_payloads(self, full):
        graphs = full.get_uns("drug_graphs")
        rows = [line.split("\t") for line in full.get_uns("pathways_gmt").splitlines()]
        genes = set(full.get_sample_feature_names("gene_expression")[:N_FEATURES])
        ontology = full.get_uns("sparsego")

        assert set(graphs) == set(full.drug_ids)
        assert {"x", "edge_index", "edge_attr"} <= set(next(iter(graphs.values())))
        assert len(rows) == 2
        assert all(len(row) - 2 >= 5 for row in rows)
        assert {line.split("\t")[1] for line in ontology["gene2ind"].splitlines()} == genes
        assert "default" in ontology["ontology"]
        assert len(full.get_uns("bpe_codes").splitlines()) > 1
        assert full.get_sample_features("pathway_features", np.asarray(full.sample_ids)).shape == (24, 2)

    @pytest.mark.parametrize("layer", ["AUC", "IC50"])
    def test_extra_layers_keep_the_unmeasured_pairs(self, full, layer):
        np.testing.assert_array_equal(np.isnan(full.get_response_layer(layer)), np.isnan(full.response_matrix))

    def test_layer_values_follow_the_response(self, full):
        measured = _measured(full.response_matrix)
        correlation = np.corrcoef(full.response_matrix[measured], full.get_response_layer("AUC")[measured])[0, 1]

        assert correlation < -0.9
        np.testing.assert_allclose(
            full.get_response_layer("IC50")[measured], 10.0 ** (6.0 - full.response_matrix[measured]), rtol=1e-4
        )

    def test_extras_do_not_change_the_response(self, full):
        np.testing.assert_array_equal(synthetic(n_samples=24, n_drugs=8).response_matrix, full.response_matrix)

    @pytest.mark.parametrize("keyword", ["sample_views", "drug_views", "uns_payloads", "extra_layers"])
    def test_an_unknown_name_is_rejected_with_the_valid_ones(self, keyword):
        with pytest.raises(ValueError, match=keyword):
            synthetic(**{keyword: ["nonsense"]})


@pytest.mark.slow
def test_the_suite_builder_keeps_the_shape_views_and_extras_the_suite_relies_on():
    from tests.synthetic import build_synthetic_dataset as suite_builder

    built = suite_builder()
    ids = np.asarray(built.sample_ids)

    assert built.name == "SYNTH"
    assert built.response_matrix.shape == (24, 8)
    assert len(set(built.response.obs[TISSUE_IDENTIFIER])) == 6
    assert int(np.isnan(built.response_matrix).sum()) == 6
    assert set(built.mdata.mod) == {
        "response",
        "gene_expression",
        "proteomics",
        "mutations",
        "methylation",
        "copy_number_variation",
    }
    assert set(built.mdata.mod["response"].varm) == {"morgan_fingerprint", "chemberta", "bpe_smiles", "smilesvec"}
    assert set(built.mdata.uns) >= {"drug_graphs", "pathways_gmt", "sparsego", "bpe_codes"}
    assert {"AUC", "IC50", "LN_IC50"} <= set(built.mdata.mod["response"].layers)
    assert built.get_sample_features("gene_expression", ids).shape == (24, len(view_gene_union("gene_expression")))
    assert built.get_sample_features("methylation", ids).shape == (24, len(view_gene_union("methylation")))


def test_determinism_is_byte_exact_and_seed_dependent():
    first = build_synthetic_dataset(omics=["gene_expression"])
    second = build_synthetic_dataset(omics=["gene_expression"])
    other = build_synthetic_dataset(seed=2)
    ids = np.asarray(first.sample_ids)

    assert first.response_matrix.tobytes() == second.response_matrix.tobytes()
    assert (
        first.get_sample_features("gene_expression", ids).tobytes()
        == second.get_sample_features("gene_expression", ids).tobytes()
    )
    assert not np.array_equal(first.response_matrix, other.response_matrix)


def test_drug_means_differ_by_more_than_the_within_drug_spread():
    matrix = build_synthetic_dataset(missing_fraction=0.0).response_matrix

    assert matrix.mean(axis=0).std() > 1.5 * matrix.std(axis=0).mean()


def test_omics_features_are_centred_on_six_and_share_hidden_factors():
    both = build_synthetic_dataset(omics=["gene_expression", "proteomics"], n_samples=200, n_features=40)
    ids = np.asarray(both.sample_ids)
    expression = both.get_sample_features("gene_expression", ids)
    proteins = both.get_sample_features("proteomics", ids)

    shared = np.linalg.svd(np.hstack([expression, proteins]) - 6.0, compute_uv=False)

    assert expression.mean() == pytest.approx(6.0, abs=0.5)
    assert shared[3] < 0.5 * shared[2]


@pytest.mark.parametrize("mode", ["LPO", "LSO", "LDO", "LTO"])
def test_every_builtin_splitter_accepts_it_and_never_selects_unmeasured_pairs(dataset, mode):
    from drevalpy.registry import splitter

    folds = splitter.get(mode)(dataset, n_splits=2, validation_ratio=0.1, quality=CurveQuality.default())
    unmeasured = ~_measured(dataset.response_matrix)

    assert len(folds) == 2
    for fold in folds:
        assert fold.train.any()
        assert fold.test.any()
        assert not (fold.train.mask & unmeasured).any()
        assert not (fold.test.mask & unmeasured).any()


@pytest.mark.slow
class TestSimpsonPattern:
    """On the default seed, drug potency dominates the pooled score but not the per-drug one."""

    @pytest.fixture(scope="class")
    def experiment(self) -> ExperimentResult:
        data = build_synthetic_dataset(omics=["gene_expression"])
        models = [
            ModelSpec.from_name(name) for name in ("NaiveDrugMeanPredictor", "NaiveMeanEffectsPredictor", "ElasticNet")
        ]
        return drevalpy.run(models, data, "LPO", hyperparameter_tuning=True, hpo_num_trials=8)

    def test_the_drug_mean_baseline_scores_high_pooled_but_is_undefined_per_drug(self, experiment):
        pooled = experiment.summary_table().loc["NaiveDrugMeanPredictor", "Pearson"]
        per_drug = experiment.summary_table(group_by="drug").loc["NaiveDrugMeanPredictor", "Pearson"]

        assert pooled >= 0.7
        assert experiment.summary_group_counts(group_by="drug").loc["NaiveDrugMeanPredictor", "Pearson"] == 0
        assert np.isnan(per_drug)

    def test_naive_mean_effects_loses_per_drug_and_elasticnet_beats_it(self, experiment):
        pooled = experiment.summary_table().loc["NaiveMeanEffectsPredictor", "Pearson"]
        per_drug = experiment.summary_table(group_by="drug")
        naive = per_drug.loc["NaiveMeanEffectsPredictor", "Pearson"]

        assert naive <= pooled - 0.3
        assert per_drug.loc["ElasticNet", "Pearson"] > naive
