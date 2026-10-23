"""Tests for :mod:`drevalpy.data._synthetic`, i.e. :func:`drevalpy.data.synthetic`."""

from __future__ import annotations

import numpy as np
import pytest

import drevalpy
from drevalpy import ModelSpec
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
from drevalpy.data.utils import TISSUE_IDENTIFIER
from drevalpy.errors import InvalidInputError
from drevalpy.testing import build_synthetic_dataset
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.results import ExperimentResult


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return build_synthetic_dataset()


class TestPublicSurface:
    def test_the_top_level_name_is_the_data_function(self):
        assert drevalpy.synthetic is drevalpy.data.synthetic

    def test_the_data_function_is_the_builder(self):
        assert drevalpy.data.synthetic is synthetic

    def test_the_testing_name_is_an_alias(self):
        assert build_synthetic_dataset is synthetic

    def test_the_default_sample_count_is_sixty(self):
        assert N_SAMPLES == 60


class TestShapeAndNaming:
    def test_it_returns_a_dataset(self, dataset):
        assert isinstance(dataset, Dataset)

    def test_the_default_name_is_recorded(self, dataset):
        assert dataset.name == DATASET_NAME

    def test_the_name_is_configurable(self):
        assert build_synthetic_dataset(name="MYDATA").name == "MYDATA"

    def test_the_response_matrix_has_the_default_shape(self, dataset):
        assert dataset.response_matrix.shape == (N_SAMPLES, N_DRUGS)

    def test_the_shape_is_configurable(self):
        small = build_synthetic_dataset(n_samples=6, n_drugs=3, n_tissues=3)

        assert small.response_matrix.shape == (6, 3)

    def test_the_default_carries_the_response_and_gene_expression(self, dataset):
        assert sorted(dataset.mdata.mod) == ["gene_expression", "response"]

    @pytest.mark.parametrize("omics", [(), None])
    def test_no_omics_leaves_only_the_response(self, omics):
        assert list(build_synthetic_dataset(omics=omics).mdata.mod) == ["response"]

    def test_identifiers_are_unique(self, dataset):
        assert len(set(dataset.sample_ids)) == N_SAMPLES
        assert len(set(dataset.drug_ids)) == N_DRUGS


class TestMetadata:
    def test_obs_holds_only_tissue_under_an_unnamed_sample_id_index(self, dataset):
        obs = dataset.response.obs

        assert list(obs.columns) == [TISSUE_IDENTIFIER]
        assert obs.index.name is None
        assert obs.index[0] == "SYNTH-000"

    def test_tissue_labels_are_resolvable_through_the_dataset(self, dataset):
        tissues = dataset.get_tissue(np.asarray(dataset.sample_ids))

        assert len(tissues) == N_SAMPLES
        assert all(isinstance(tissue, str) for tissue in tissues)

    def test_the_requested_number_of_tissues_is_produced(self):
        small = build_synthetic_dataset(n_tissues=3)

        assert len(set(small.response.obs[TISSUE_IDENTIFIER])) == 3

    def test_the_published_measure_is_stored_as_a_layer(self, dataset):
        layer = dataset.get_response_layer(MEASURE)

        np.testing.assert_array_equal(np.isnan(layer), np.isnan(dataset.response_matrix))

    def test_drug_names_are_recorded(self, dataset):
        assert dataset.response.var["drug_name"].notna().all()


class TestDrugs:
    def test_every_drug_has_a_parseable_canonical_smiles(self, dataset):
        from rdkit import Chem

        smiles = dataset.response.var["canonical_smiles"]

        assert smiles.notna().all()
        assert all(Chem.MolFromSmiles(value) is not None for value in smiles)

    def test_smiles_are_distinct(self):
        full = synthetic(n_drugs=MAX_SYNTHETIC_DRUGS)

        assert full.response.var["canonical_smiles"].nunique() == MAX_SYNTHETIC_DRUGS

    def test_the_limit_is_at_least_eight(self):
        assert MAX_SYNTHETIC_DRUGS >= 8

    def test_the_limit_itself_builds(self):
        assert synthetic(n_drugs=MAX_SYNTHETIC_DRUGS).response_matrix.shape[1] == MAX_SYNTHETIC_DRUGS

    def test_one_drug_above_the_limit_raises_naming_it(self):
        with pytest.raises(InvalidInputError, match=str(MAX_SYNTHETIC_DRUGS)):
            synthetic(n_drugs=MAX_SYNTHETIC_DRUGS + 1)


class TestMissingResponses:
    def test_some_pairs_are_unmeasured_by_default(self, dataset):
        assert np.isnan(dataset.response_matrix).any()

    def test_every_sample_keeps_a_measurement(self, dataset):
        assert (~np.isnan(dataset.response_matrix)).any(axis=1).all()

    def test_every_drug_keeps_a_measurement(self, dataset):
        assert (~np.isnan(dataset.response_matrix)).any(axis=0).all()

    def test_zero_fraction_leaves_a_complete_matrix(self):
        complete = build_synthetic_dataset(missing_fraction=0.0)

        assert not np.isnan(complete.response_matrix).any()

    def test_a_larger_fraction_removes_more_pairs(self):
        sparse = build_synthetic_dataset(missing_fraction=0.4)
        dense = build_synthetic_dataset(missing_fraction=0.05)

        assert np.isnan(sparse.response_matrix).sum() > np.isnan(dense.response_matrix).sum()

    def test_an_extreme_fraction_still_honours_the_row_guarantee(self):
        """Clamped rather than obeyed: an empty row breaks the LSO splitters."""
        extreme = build_synthetic_dataset(missing_fraction=1.0)

        assert (~np.isnan(extreme.response_matrix)).any(axis=1).all()
        assert (~np.isnan(extreme.response_matrix)).any(axis=0).all()


class TestPunchHoles:
    def test_a_degenerate_matrix_is_left_alone(self):
        matrix = np.ones((1, 4), dtype=np.float32)

        _punch_holes(matrix, np.random.default_rng(0), 0.5)

        assert not np.isnan(matrix).any()

    def test_it_modifies_in_place(self):
        matrix = np.ones((6, 6), dtype=np.float32)

        _punch_holes(matrix, np.random.default_rng(0), 0.2)

        assert np.isnan(matrix).any()


class TestOmics:
    def test_a_sequence_adds_full_coverage_modalities(self):
        with_omics = build_synthetic_dataset(omics=["gene_expression", "proteomics"])

        assert set(with_omics.mdata.mod) == {"response", "gene_expression", "proteomics"}

    def test_features_are_retrievable_through_the_public_accessor(self):
        with_omics = build_synthetic_dataset(omics=["gene_expression"])

        matrix = with_omics.get_sample_features("gene_expression", np.asarray(with_omics.sample_ids))

        assert matrix.shape == (N_SAMPLES, N_FEATURES)

    def test_feature_names_are_exposed(self):
        with_omics = build_synthetic_dataset(omics=["gene_expression"], feature_names=["A", "B"])

        assert with_omics.get_sample_feature_names("gene_expression") == ("A", "B")

    def test_the_feature_width_is_configurable(self):
        with_omics = build_synthetic_dataset(omics=["gene_expression"], n_features=3)

        assert with_omics.mdata.mod["gene_expression"].shape[1] == 3

    def test_a_mapping_sets_per_modality_coverage(self):
        """Partial coverage is what makes the NaN-filtering path fire."""
        partial = build_synthetic_dataset(omics={"gene_expression": 10})

        assert partial.mdata.mod["gene_expression"].shape[0] == 10

    def test_uncovered_samples_come_back_as_nan(self):
        partial = build_synthetic_dataset(omics={"gene_expression": 10})

        matrix = partial.get_sample_features("gene_expression", np.asarray(partial.sample_ids))

        assert np.isnan(matrix[10:]).all()
        assert not np.isnan(matrix[:10]).any()

    def test_the_gistic_alias_resolves_to_the_stored_modality(self):
        """The public name is suffixed; the published files are not."""
        with_cnv = build_synthetic_dataset(omics=["copy_number_variation_gistic"])

        assert "copy_number_variation" in with_cnv.mdata.mod
        assert with_cnv.get_sample_features("copy_number_variation_gistic", np.asarray([])).size == 0


ALL_OMICS = ("gene_expression", "proteomics", "mutations", "methylation", "copy_number_variation_gistic")


class TestOmicsViews:
    """The five views the zoo reads, each with realistic names and a plausible value range."""

    pytestmark = pytest.mark.slow

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

    def test_gene_views_are_named_by_landmark_symbols(self, every_view):
        from drevalpy.components.featurizers.sample.gene_lists import (
            gene_names_from_list_csv,
            resolve_gene_list_path,
        )

        symbols = gene_names_from_list_csv(resolve_gene_list_path("landmark_genes"))

        for view in ("gene_expression", "proteomics", "mutations", "copy_number_variation_gistic"):
            assert every_view.get_sample_feature_names(view) == tuple(symbols[:12])

    def test_methylation_is_named_by_genomic_sites(self, every_view):
        names = every_view.get_sample_feature_names("methylation")

        assert len(set(names)) == 12
        assert all(name.startswith("chr") for name in names)

    def test_mutations_are_binary_calls(self, every_view):
        matrix = self._matrix(every_view, "mutations")

        assert set(np.unique(matrix)) == {0.0, 1.0}
        assert 0.1 < matrix.mean() < 0.4

    def test_methylation_is_a_beta_value(self, every_view):
        matrix = self._matrix(every_view, "methylation")

        assert matrix.min() > 0.0
        assert matrix.max() < 1.0

    def test_copy_number_is_a_gistic_integer(self, every_view):
        matrix = self._matrix(every_view, "copy_number_variation_gistic")

        assert set(np.unique(matrix)) <= {-2.0, -1.0, 0.0, 1.0, 2.0}

    def test_proteomics_intensities_stay_positive(self, every_view):
        assert self._matrix(every_view, "proteomics").min() > 0.0

    def test_a_mapping_of_feature_names_applies_per_view(self):
        dataset = synthetic(omics=["gene_expression", "methylation"], feature_names={"methylation": ["s1", "s2", "s3"]})

        assert dataset.get_sample_feature_names("methylation") == ("s1", "s2", "s3")
        assert len(dataset.get_sample_feature_names("gene_expression")) == N_FEATURES

    def test_the_landmark_featurizer_selects_every_default_gene(self):
        from drevalpy.components.featurizers.sample.landmark import LandmarkGenesFeaturizer
        from drevalpy.types.data.feature_source import SampleFeatureSource

        dataset = synthetic()
        ids = np.asarray(dataset.sample_ids)
        featurizer = LandmarkGenesFeaturizer()

        featurizer.fit(SampleFeatureSource(dataset, ids), entity_ids=ids)

        assert featurizer.output_dim == N_FEATURES

    def test_more_features_than_symbols_asks_for_feature_names(self):
        with pytest.raises(ValueError, match="feature_names"):
            synthetic(n_features=100_000)


class TestExtras:
    """Opt-in drug views, payloads, layers and sample views."""

    pytestmark = pytest.mark.slow

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

    def test_the_fingerprint_is_the_real_featurizers_output(self, full):
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        smiles = full.response.var["canonical_smiles"].iloc[0]
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
        expected = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))

        np.testing.assert_array_equal(full.get_drug_features("morgan_fingerprint", full.drug_ids)[0], expected)

    def test_noise_views_are_seeded(self):
        first = synthetic(drug_views=["chemberta"]).get_drug_features("chemberta", np.asarray(["100000"]))
        second = synthetic(drug_views=["chemberta"]).get_drug_features("chemberta", np.asarray(["100000"]))

        np.testing.assert_array_equal(first, second)

    def test_every_drug_gets_a_graph(self, full):
        graphs = full.get_uns("drug_graphs")

        assert set(graphs) == set(full.drug_ids)
        assert {"x", "edge_index", "edge_attr"} <= set(next(iter(graphs.values())))

    def test_the_pathway_sets_are_large_enough_for_gsva(self, full):
        rows = [line.split("\t") for line in full.get_uns("pathways_gmt").splitlines()]

        assert len(rows) == 2
        assert all(len(row) - 2 >= 5 for row in rows)

    def test_the_ontology_covers_every_gene(self, full):
        genes = set(full.get_sample_feature_names("gene_expression"))
        ontology = full.get_uns("sparsego")

        assert {line.split("\t")[1] for line in ontology["gene2ind"].splitlines()} == genes
        assert "default" in ontology["ontology"]

    def test_learned_bpe_codes_are_stored(self, full):
        assert len(full.get_uns("bpe_codes").splitlines()) > 1

    def test_the_pathway_view_has_one_row_per_sample(self, full):
        assert full.get_sample_features("pathway_features", np.asarray(full.sample_ids)).shape == (24, 2)

    @pytest.mark.parametrize("layer", ["AUC", "IC50"])
    def test_extra_layers_keep_the_unmeasured_pairs(self, full, layer):
        values = full.get_response_layer(layer)

        np.testing.assert_array_equal(np.isnan(values), np.isnan(full.response_matrix))

    def test_the_auc_falls_as_potency_rises(self, full):
        measured = ~np.isnan(full.response_matrix)

        correlation = np.corrcoef(full.response_matrix[measured], full.get_response_layer("AUC")[measured])[0, 1]

        assert correlation < -0.9

    def test_the_ic50_is_in_micromolar_units(self, full):
        measured = ~np.isnan(full.response_matrix)

        np.testing.assert_allclose(
            full.get_response_layer("IC50")[measured], 10.0 ** (6.0 - full.response_matrix[measured]), rtol=1e-4
        )

    def test_extras_do_not_change_the_response(self, full):
        plain = synthetic(n_samples=24, n_drugs=8)

        np.testing.assert_array_equal(plain.response_matrix, full.response_matrix)

    @pytest.mark.parametrize("keyword", ["sample_views", "drug_views", "uns_payloads", "extra_layers"])
    def test_an_unknown_name_is_rejected_with_the_valid_ones(self, keyword):
        with pytest.raises(ValueError, match=keyword):
            synthetic(**{keyword: ["nonsense"]})


class TestSuiteBuilder:
    """``tests.synthetic.build_synthetic_dataset`` is one call into ``synthetic``."""

    pytestmark = pytest.mark.slow

    @pytest.fixture(scope="class")
    def built(self) -> Dataset:
        from tests.synthetic import build_synthetic_dataset as suite_builder

        return suite_builder()

    def test_it_keeps_the_shape_seed_and_name_the_suite_relies_on(self, built):
        assert built.name == "SYNTH"
        assert built.response_matrix.shape == (24, 8)
        assert len(set(built.response.obs[TISSUE_IDENTIFIER])) == 6
        assert int(np.isnan(built.response_matrix).sum()) == 6

    def test_it_carries_all_five_views_and_the_extras(self, built):
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

    def test_it_keeps_the_feature_widths(self, built):
        ids = np.asarray(built.sample_ids)

        assert built.get_sample_features("gene_expression", ids).shape == (24, 16)
        assert built.get_sample_features("methylation", ids).shape == (24, 20)


class TestDeterminism:
    def test_the_same_seed_gives_the_same_matrix(self):
        first = build_synthetic_dataset()
        second = build_synthetic_dataset()

        np.testing.assert_array_equal(first.response_matrix, second.response_matrix)

    def test_a_different_seed_gives_a_different_matrix(self):
        first = build_synthetic_dataset(seed=1)
        second = build_synthetic_dataset(seed=2)

        assert not np.array_equal(first.response_matrix, second.response_matrix)

    def test_the_same_seed_gives_byte_identical_matrices(self):
        first = build_synthetic_dataset(omics=["gene_expression"])
        second = build_synthetic_dataset(omics=["gene_expression"])
        view = np.asarray(first.sample_ids)

        assert first.response_matrix.tobytes() == second.response_matrix.tobytes()
        assert (
            first.get_sample_features("gene_expression", view).tobytes()
            == second.get_sample_features("gene_expression", view).tobytes()
        )


class TestResponseModel:
    """The drug means are spread, and the omics views carry the sample effect."""

    def test_drug_means_differ_by_more_than_the_within_drug_spread(self):
        complete = build_synthetic_dataset(missing_fraction=0.0)
        matrix = complete.response_matrix

        between = matrix.mean(axis=0).std()
        within = matrix.std(axis=0).mean()

        assert between > 1.5 * within

    def test_omics_features_are_centred_on_six(self):
        with_omics = build_synthetic_dataset(omics=["gene_expression"], n_samples=200)

        features = with_omics.get_sample_features("gene_expression", np.asarray(with_omics.sample_ids))

        assert features.mean() == pytest.approx(6.0, abs=0.5)

    def test_every_omics_view_is_driven_by_the_same_hidden_factors(self):
        both = build_synthetic_dataset(omics=["gene_expression", "proteomics"], n_samples=200, n_features=40)
        ids = np.asarray(both.sample_ids)
        expression = both.get_sample_features("gene_expression", ids)
        proteins = both.get_sample_features("proteomics", ids)

        shared = np.linalg.svd(np.hstack([expression, proteins]) - 6.0, compute_uv=False)

        assert shared[3] < 0.5 * shared[2]


class TestSplitterCompatibility:
    """The builder exists so plugin CI can split and train; prove it can."""

    @pytest.mark.parametrize("mode", ["LPO", "LSO", "LDO", "LTO"])
    def test_every_builtin_splitter_accepts_it(self, dataset, mode):
        from drevalpy.registry import splitter

        folds = splitter.get(mode)(dataset, n_splits=2, validation_ratio=0.1)

        assert len(folds) == 2
        for fold in folds:
            assert fold.train.any()
            assert fold.test.any()

    def test_folds_never_select_unmeasured_pairs(self, dataset):
        from drevalpy.registry import splitter

        observed = ~np.isnan(dataset.response_matrix)

        for fold in splitter.get("LPO")(dataset, n_splits=2, validation_ratio=0.1):
            assert not (fold.train.mask & ~observed).any()
            assert not (fold.test.mask & ~observed).any()


class TestSimpsonPattern:
    """On the default seed, drug potency dominates the pooled score but not the per-drug one."""

    pytestmark = pytest.mark.slow

    @pytest.fixture(scope="class")
    def experiment(self) -> ExperimentResult:
        data = build_synthetic_dataset(omics=["gene_expression"])
        models = [
            ModelSpec.from_name(name) for name in ("NaiveDrugMeanPredictor", "NaiveMeanEffectsPredictor", "ElasticNet")
        ]
        return drevalpy.run(models, data, "LPO", hyperparameter_tuning=True, hpo_num_trials=8)

    @staticmethod
    def _pooled(experiment: ExperimentResult, model: str) -> float:
        return experiment.summary_table()[model]["Pearson"]

    @staticmethod
    def _per_drug(experiment: ExperimentResult, model: str) -> float:
        return experiment.summary_table(group_by="drug")[model]["Pearson"]

    def test_the_drug_mean_baseline_scores_high_pooled(self, experiment):
        assert self._pooled(experiment, "NaiveDrugMeanPredictor") >= 0.7

    def test_the_drug_mean_baseline_has_no_per_drug_signal(self, experiment):
        per_drug = self._per_drug(experiment, "NaiveDrugMeanPredictor")

        assert np.isnan(per_drug) or per_drug == 0.0

    def test_the_drug_mean_baseline_is_undefined_for_every_drug(self, experiment):
        assert experiment.summary_group_counts(group_by="drug")["NaiveDrugMeanPredictor"]["Pearson"] == 0
        assert np.isnan(self._per_drug(experiment, "NaiveDrugMeanPredictor"))

    def test_naive_mean_effects_loses_at_least_point_three_per_drug(self, experiment):
        pooled = self._pooled(experiment, "NaiveMeanEffectsPredictor")
        per_drug = self._per_drug(experiment, "NaiveMeanEffectsPredictor")

        assert per_drug <= pooled - 0.3

    def test_elasticnet_beats_naive_mean_effects_per_drug(self, experiment):
        assert self._per_drug(experiment, "ElasticNet") > self._per_drug(experiment, "NaiveMeanEffectsPredictor")
