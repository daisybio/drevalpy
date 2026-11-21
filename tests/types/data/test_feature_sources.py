"""Reading features through the sample and drug feature sources."""

from __future__ import annotations

import json

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from drevalpy.errors import InvalidInputError
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import PRECOMPUTED_VIEWS_KEY, FeatureSource, precomputed_views
from tests.synthetic import BPE_LENGTH, CHEMBERTA_DIM, FINGERPRINT_BITS, N_GENES, N_PATHWAYS

SAMPLES = np.array(["cl1", "cl2", "cl3"])
DRUGS = np.array(["d1", "d2"])


def _dataset(
    *,
    varm: dict[str, object] | None = None,
    obsm: dict[str, np.ndarray] | None = None,
    uns: dict[str, object] | None = None,
    cnv_under_stored_name: bool = False,
    tissue: bool = True,
) -> Dataset:
    response = ad.AnnData(
        X=np.arange(6.0, dtype=np.float32).reshape(3, 2),
        obs=pd.DataFrame({"tissue": ["Lung", "Blood", "Skin"]} if tissue else {}, index=SAMPLES),
        var=pd.DataFrame(index=DRUGS),
    )
    for key, value in (varm or {}).items():
        response.varm[key] = value
    for key, value in (obsm or {}).items():
        response.obsm[key] = value

    expression = ad.AnnData(
        X=np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        obs=pd.DataFrame(index=SAMPLES),
        var=pd.DataFrame(index=["GENE1", "GENE2"]),
    )
    mods: dict[str, ad.AnnData] = {"response": response, "gene_expression": expression}
    if cnv_under_stored_name:
        mods["copy_number_variation"] = ad.AnnData(
            X=np.zeros((3, 2), dtype=np.float32),
            obs=pd.DataFrame(index=SAMPLES),
            var=pd.DataFrame(index=["GENE1", "GENE2"]),
        )

    with md.set_options(pull_on_update=False):
        mdata = md.MuData(mods)
    if tissue:
        mdata.obs = response.obs.copy()
    for key, value in (uns or {}).items():
        mdata.uns[key] = value
    return Dataset(mdata, name="feature-access")


def _zeros(*shape: int) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


@pytest.fixture()
def sample_source(synthetic_dataset: Dataset) -> FeatureSource:
    return FeatureSource(synthetic_dataset, "sample")


@pytest.fixture()
def drug_source(synthetic_dataset: Dataset) -> FeatureSource:
    return FeatureSource(synthetic_dataset, "drug")


def test_an_unknown_side_is_rejected(synthetic_dataset: Dataset):
    with pytest.raises(ValueError, match="Unknown feature source side 'cell'"):
        FeatureSource(synthetic_dataset, "cell")  # type: ignore[arg-type]


def test_uns_returns_the_value_or_none(sample_source: FeatureSource):
    assert set(sample_source.uns("sparsego")) == {"gene2ind", "ontology"}
    assert all("\t" in line for line in sample_source.uns("pathways_gmt").strip().splitlines())
    assert sample_source.uns("nonexistent") is None


class TestSampleViews:
    @pytest.mark.parametrize(("view", "dim"), [("gene_expression", N_GENES), ("pathway_features", N_PATHWAYS)])
    def test_view_rows_follow_the_ids_as_float32(self, synthetic_dataset: Dataset, view: str, dim: int):
        source = FeatureSource(synthetic_dataset, "sample")
        ids = synthetic_dataset.sample_ids[[3, 0, 1]]

        matrix = source.view(view, ids)

        assert matrix.shape == (3, dim)
        assert matrix.dtype == np.float32
        np.testing.assert_array_equal(matrix[1:2], source.view(view, ids[1:2]))

    def test_unknown_ids_get_nan_rows(self, sample_source: FeatureSource):
        matrix = sample_source.view("gene_expression", np.array(["FAKE_ID_1", "FAKE_ID_2"]))

        assert matrix.shape == (2, N_GENES)
        assert np.isnan(matrix).all()

    def test_a_view_is_read_from_its_stored_modality(self):
        source = FeatureSource(_dataset(cnv_under_stored_name=True), "sample")

        assert source.view("copy_number_variation_gistic", SAMPLES).shape == (3, 2)
        with pytest.raises(KeyError, match="copy_number_variation_gistic"):
            source.view("absent", SAMPLES)

    def test_feature_names(self):
        source = FeatureSource(_dataset(obsm={"pathway_features": _zeros(3, 4)}), "sample")

        assert source.feature_names("gene_expression") == ("GENE1", "GENE2")
        assert source.feature_names("absent") is None
        assert source.feature_names("pathway_features") is None

    def test_an_obsm_view_is_a_dense_sample_view(self):
        source = FeatureSource(_dataset(obsm={"pathway_features": np.ones((3, 4), dtype=np.float32)}), "sample")

        assert source.view("pathway_features", SAMPLES[:2]).shape == (2, 4)
        with pytest.raises(KeyError, match="Sample view 'pathway_features' not found"):
            FeatureSource(_dataset(), "sample").view("pathway_features", SAMPLES)

    def test_tissue(self, synthetic_dataset: Dataset, sample_source: FeatureSource):
        ids = np.array([*synthetic_dataset.sample_ids[:2], "NOT_A_SAMPLE"])

        tissue = sample_source.tissue(ids)

        assert tissue is not None
        assert tissue[:2].tolist() == synthetic_dataset.get_tissue(ids[:2]).tolist()
        assert np.isnan(tissue[2])
        assert FeatureSource(_dataset(tissue=False), "sample").tissue(SAMPLES) is None

    def test_expression_unit(self):
        dataset = _dataset()
        source = FeatureSource(dataset, "sample")
        assert source.expression_unit("gene_expression") is None

        dataset.mdata.mod["gene_expression"].uns["expression_unit"] = "tpm"

        assert source.expression_unit("gene_expression") == "tpm"
        assert source.expression_unit("absent") is None


class TestDrugViews:
    @pytest.mark.parametrize(
        ("view", "dim"),
        [("chemberta", CHEMBERTA_DIM), ("morgan_fingerprint", FINGERPRINT_BITS), ("bpe_smiles", BPE_LENGTH)],
    )
    def test_varm_views(self, synthetic_dataset: Dataset, drug_source: FeatureSource, view: str, dim: int):
        matrix = drug_source.view(view, synthetic_dataset.drug_ids[:4])

        assert matrix.shape == (4, dim)
        assert matrix.dtype == np.float32
        assert len(drug_source.feature_names(view) or ()) == dim

    def test_feature_names_and_the_unknown_view_message(self):
        frame = pd.DataFrame(_zeros(2, 2), columns=["bit0", "bit1"], index=DRUGS)
        source = FeatureSource(
            _dataset(varm={"zeta": _zeros(2, 1), "alpha": _zeros(2, 1), "pca": _zeros(2, 3), "fp": frame}), "drug"
        )

        assert source.feature_names("pca") == ("0", "1", "2")
        assert source.feature_names("fp") == ("bit0", "bit1")
        assert source.feature_names("absent") is None
        with pytest.raises(KeyError, match=r"Available: \['alpha', 'fp', 'pca', 'zeta'\]"):
            source.view("absent", DRUGS)

    def test_a_view_is_read_by_its_exact_name_only(self):
        with pytest.raises(KeyError, match="Drug view 'pca' not found"):
            FeatureSource(_dataset(varm={"pca:0": _zeros(2, 3)}), "drug").view("pca", DRUGS)

    def test_smiles(self, synthetic_dataset: Dataset, drug_source: FeatureSource):
        ids = synthetic_dataset.drug_ids[:3]

        smiles = drug_source.smiles(ids)

        assert smiles is not None
        assert list(smiles.index) == list(ids)
        assert all(isinstance(value, str) and value for value in smiles)
        unknown = drug_source.smiles(np.array(["not-a-drug"]))
        assert unknown is not None and unknown.isna().all()
        assert FeatureSource(_dataset(), "drug").smiles(DRUGS) is None

    def test_graphs(self, synthetic_dataset: Dataset, drug_source: FeatureSource):
        graphs = drug_source.graphs(np.array([synthetic_dataset.drug_ids[0], "NOT_A_DRUG"]))

        assert graphs[0] is not None
        assert set(graphs[0]) >= {"x", "edge_index", "edge_attr"}
        assert graphs[1] is None
        assert FeatureSource(_dataset(), "drug").graphs(DRUGS) == [None, None]


class TestRaggedDrugView:
    @staticmethod
    def _ragged() -> np.ndarray:
        ragged = np.empty((2, 1), dtype=object)
        ragged[0, 0] = np.ones((2, 3), dtype=np.float32)
        ragged[1, 0] = np.ones((3, 3), dtype=np.float32)
        return ragged

    def test_a_ragged_view_is_a_clear_input_error_not_a_bare_value_error(self):
        source = FeatureSource(_dataset(varm={"molgnet_features": self._ragged()}), "drug")

        with pytest.raises(InvalidInputError, match="molgnet_features.*object array.*computes it from the SMILES"):
            source.view("molgnet_features", DRUGS)

    def test_the_molgnet_featurizer_surfaces_the_error_instead_of_computing_over_it(self):
        from drevalpy.components.featurizers.drug.molgnet import MolGNetDrugFeaturizer
        from drevalpy.models._feature_preparation import FitIds, prepare_features

        source = FeatureSource(_dataset(varm={"molgnet_features": self._ragged()}), "drug")

        with pytest.raises(InvalidInputError, match="object array"):
            prepare_features(MolGNetDrugFeaturizer(), source, DRUGS, fit=FitIds(DRUGS))


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

    def test_sides(self, sample_source: FeatureSource, drug_source: FeatureSource):
        assert sample_source.side == "sample"
        assert drug_source.side == "drug"


_SAMPLES = np.array(["s1", "s2"])
_DRUGS = np.array(["d1", "d2"])


def _recorded_dataset(record: object) -> Dataset:
    response = ad.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame(index=_SAMPLES),
        var=pd.DataFrame(index=_DRUGS),
    )
    response.varm["morgan_fingerprint"] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mdata = md.MuData({"response": response})
    if record is not None:
        mdata.uns[PRECOMPUTED_VIEWS_KEY] = record
    return Dataset(mdata, name="recorded")


class TestViewSettings:
    @pytest.mark.parametrize(
        "record",
        [
            json.dumps({"drug": {"morgan_fingerprint": {"n_bits": 2}}}),
            {"drug": {"morgan_fingerprint": {"n_bits": 2}}},
        ],
        ids=["json", "dict"],
    )
    def test_recorded_settings_are_read_per_side_and_view(self, record):
        dataset = _recorded_dataset(record)

        assert FeatureSource(dataset, "drug").view_settings("morgan_fingerprint") == {"n_bits": 2}
        assert FeatureSource(dataset, "drug").view_settings("chemberta") is None
        assert FeatureSource(dataset, "sample").view_settings("morgan_fingerprint") is None
        assert precomputed_views(dataset.mdata) == {"drug": {"morgan_fingerprint": {"n_bits": 2}}}

    def test_an_unrecorded_dataset_gives_none(self, sample_source: FeatureSource):
        assert FeatureSource(_recorded_dataset(None), "drug").view_settings("morgan_fingerprint") is None
        assert precomputed_views(_recorded_dataset(None).mdata) == {}
        assert sample_source.view_settings("gene_expression") is None
