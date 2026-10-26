"""Dataset feature access: sample and drug views."""

from __future__ import annotations

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from drevalpy.types.data.dataset import Dataset

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
    def test_a_view_is_read_from_its_stored_modality(self):
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

        assert dataset.get_drug_feature_names("pca") == ("0", "1", "2")
        assert dataset.get_drug_feature_names("fp") == ("bit0", "bit1")
        assert dataset.get_drug_feature_names("absent") is None
        with pytest.raises(KeyError, match="drug IDs not found"):
            dataset.get_drug_features("pca", np.array(["absent"]), strict=True)
        with pytest.raises(KeyError, match=r"Available varm keys: \['alpha', 'fp', 'pca', 'zeta'\]"):
            dataset.get_drug_features("absent", DRUGS)

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
