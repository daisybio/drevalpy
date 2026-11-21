"""Tests for ``Dataset`` against a synthetic dataset written to and read from disk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.data import split, synthetic
from drevalpy.data._replicate_fits import KEY
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.types.data.dataset import Dataset
from tests.data._replicate_fits_table import fits_table
from tests.synthetic import (
    BUILTIN_MEASURE,
    CNV_MODALITY,
    N_DRUGS,
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
        randomized.randomization_provenance = dict(provenance)
        randomized.save(UPath(tmp_path) / "randomized.h5mu")

        reloaded = Dataset.load(UPath(tmp_path) / "randomized.h5mu")

        assert reloaded.randomization == ("SVRC", "gene_expression")
        reloaded_provenance = reloaded.randomization_provenance
        assert reloaded_provenance == provenance
        assert reloaded_provenance is not None
        assert isinstance(reloaded_provenance["views"], list)

    def test_an_unrandomized_dataset_reads_back_without_a_label(self, dataset: Dataset):
        assert dataset.randomization is None
        assert dataset.randomization_provenance is None


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


class TestMetadata:
    def test_sample_meta_and_tissue(self, dataset: Dataset):
        meta = dataset.sample_meta
        assert "tissue" in meta.columns
        assert "cell_line_name" not in meta.columns  # a dataset without the optional cell-line columns
        assert len(dataset.get_tissue(dataset.sample_ids[:5])) == 5
        assert len(dataset.get_tissue(np.array(["FAKE_ID"]))) == 1
        assert len(np.unique(dataset.get_tissue(dataset.sample_ids))) >= 3


class TestReplicateFitsRoundTrip:
    def test_the_table_and_its_dtypes_survive_save_and_load(self, dataset: Dataset, tmp_path: UPath):
        table = fits_table(list(dataset.sample_ids), list(dataset.drug_ids), extra="cell_batch")
        dataset.response.uns[KEY] = table
        path = UPath(tmp_path) / "with_fits.h5mu"

        dataset.save(path)
        loaded = Dataset.load(path).response.uns[KEY]

        pd.testing.assert_frame_equal(loaded, table, check_index_type=False)
        assert loaded["replicate"].dtype == np.int32
        assert loaded["AUC"].dtype == np.float32
        assert isinstance(loaded["cell_batch"].dtype, pd.CategoricalDtype)


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


@pytest.fixture(scope="module")
def annotated() -> Dataset:
    ds = synthetic(n_samples=2, n_drugs=2)
    ds.mdata.obs["study"] = ["A", "B"]
    ds.mdata.obs["score"] = [0.5, 1.5]
    return ds


class TestSampleColumn:
    def test_returns_values_in_the_order_of_the_requested_ids(self, annotated: Dataset) -> None:
        ids = annotated.sample_ids[::-1]

        assert annotated.sample_column("study", ids).tolist() == ["B", "A"]

    def test_unknown_ids_give_nan(self, annotated: Dataset) -> None:
        values = annotated.sample_column("study", np.array([annotated.sample_ids[0], "NOPE"]))

        assert values[0] == "A"
        assert np.isnan(values[1])

    def test_a_numeric_column_stays_numeric_with_unknown_ids(self, annotated: Dataset) -> None:
        values = annotated.sample_column("score", np.array([annotated.sample_ids[1], "NOPE"]))

        assert values.dtype == np.float64
        assert values[0] == 1.5
        assert np.isnan(values[1])

    def test_missing_column_names_the_available_ones(self, annotated: Dataset) -> None:
        with pytest.raises(KeyError, match=r"'nope'.*Available:.*tissue.*study"):
            annotated.sample_column("nope", annotated.sample_ids)

    def test_get_tissue_delegates_to_the_tissue_column(self, annotated: Dataset) -> None:
        ids = np.array([*annotated.sample_ids, "NOPE"])

        expected = annotated.sample_column("tissue", ids)
        actual = annotated.get_tissue(ids)

        assert actual[:-1].tolist() == expected[:-1].tolist()
        assert np.isnan(actual[-1])
