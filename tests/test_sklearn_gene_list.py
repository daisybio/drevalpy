"""Tests for the configurable gene list of single-view sklearn models."""

import pytest

from drevalpy.models import MODEL_FACTORY
from drevalpy.models.baselines.sklearn_models import SklearnModel
from drevalpy.models.utils import load_single_cell_line_view

_GENE_EXPRESSION = (
    "cellosaurus_id,cell_line_name,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
    "CVCL_1104,CAL-120,7.63,2.96,10.38,3.61,3.38,7.09,3.02\n"
    "CVCL_1174,DMS 114,7.55,2.78,11.81,4.07,3.73,2.80,6.08\n"
    "CVCL_1110,CAL-51,8.71,2.64,9.88,3.96,3.24,11.39,4.22\n"
)
#: Symbols of the two gene lists written by the gene_list_data_dir fixture.
_LANDMARK_GENES = ["BRCA1", "SCYL3", "INSIG1", "FOXO3"]
_TARGET_GENES = ["TSPAN6", "SCYL3", "BRCA1"]
_ALL_GENES = ["TSPAN6", "TNMD", "BRCA1", "SCYL3", "HDAC1", "INSIG1", "FOXO3"]


@pytest.fixture
def gene_list_data_dir(tmp_path) -> str:
    """
    Build a minimal data directory with a gene expression matrix and two gene lists.

    :param tmp_path: pytest temporary path fixture
    :returns: path to the data directory
    """
    dataset_dir = tmp_path / "GDSC1_small"
    dataset_dir.mkdir()
    (dataset_dir / "gene_expression.csv").write_text(_GENE_EXPRESSION, encoding="utf-8")

    gene_lists = tmp_path / "meta" / "gene_lists"
    gene_lists.mkdir(parents=True)
    (gene_lists / "landmark_genes_reduced.csv").write_text(
        "Entrez ID,Symbol,Name,Gene Family,Type,RNA-Seq Correlation,RNA-Seq Correlation Self-Rank\n"
        "3638,INSIG1,insulin induced gene 1,,landmark,,\n"
        "2309,FOXO3,forkhead box O3,Forkhead boxes,landmark,,\n"
        "672,BRCA1,BRCA1 DNA repair associated,,landmark,,\n"
        "57147,SCYL3,SCY1 like pseudokinase 3,SCY1 like pseudokinases,landmark,,\n",
        encoding="utf-8",
    )
    (gene_lists / "drug_target_genes_all_drugs.csv").write_text("Symbol\nTSPAN6\nSCYL3\nBRCA1\n", encoding="utf-8")
    return str(tmp_path)


def _gene_names(features) -> list[str]:
    """
    Extract the gene symbols of the gene_expression view.

    :param features: FeatureDataset returned by a loader
    :returns: gene symbols
    """
    assert features.meta_info is not None
    return list(features.meta_info["gene_expression"])


def test_load_single_cell_line_view_defaults_to_landmark_genes(gene_list_data_dir) -> None:
    """
    Omitting gene_list must reproduce the previously hard-coded landmark_genes_reduced behaviour.

    :param gene_list_data_dir: path to the temporary data directory
    """
    features = load_single_cell_line_view(["gene_expression"], gene_list_data_dir, "GDSC1_small", "ElasticNet")
    assert sorted(_gene_names(features)) == sorted(_LANDMARK_GENES)


def test_load_single_cell_line_view_honours_explicit_gene_list(gene_list_data_dir) -> None:
    """
    A different gene list has to change which genes are loaded.

    :param gene_list_data_dir: path to the temporary data directory
    """
    features = load_single_cell_line_view(
        ["gene_expression"],
        gene_list_data_dir,
        "GDSC1_small",
        "ElasticNet",
        gene_list="drug_target_genes_all_drugs",
    )
    assert sorted(_gene_names(features)) == sorted(_TARGET_GENES)


def test_load_single_cell_line_view_gene_list_none_loads_all_genes(gene_list_data_dir) -> None:
    """
    gene_list=None has to load the full expression matrix.

    :param gene_list_data_dir: path to the temporary data directory
    """
    features = load_single_cell_line_view(
        ["gene_expression"], gene_list_data_dir, "GDSC1_small", "ElasticNet", gene_list=None
    )
    assert sorted(_gene_names(features)) == sorted(_ALL_GENES)


def test_sklearn_model_gene_list_default() -> None:
    """Without the hyperparameter the model keeps the class default."""
    model = MODEL_FACTORY["ElasticNet"]()
    model.build_model(hyperparameters={"alpha": 1.0, "l1_ratio": 0.2})
    assert model.gene_list == SklearnModel.gene_list == "landmark_genes_reduced"


def test_sklearn_model_gene_list_hyperparameter(gene_list_data_dir) -> None:
    """
    The "gene_list" hyperparameter has to reach load_cell_line_features.

    :param gene_list_data_dir: path to the temporary data directory
    """
    model = MODEL_FACTORY["ElasticNet"]()
    model.build_model(hyperparameters={"alpha": 1.0, "l1_ratio": 0.2, "gene_list": "drug_target_genes_all_drugs"})
    assert model.gene_list == "drug_target_genes_all_drugs"
    # It stays in the hyperparameters, so save()/load() carry it over to predict().
    assert model.hyperparameters["gene_list"] == "drug_target_genes_all_drugs"

    features = model.load_cell_line_features(data_path=gene_list_data_dir, dataset_name="GDSC1_small")
    assert sorted(_gene_names(features)) == sorted(_TARGET_GENES)
    # The class default is untouched, only the instance was reconfigured.
    assert SklearnModel.gene_list == "landmark_genes_reduced"


def test_sklearn_model_gene_list_does_not_leak_between_instances(gene_list_data_dir) -> None:
    """
    A second model built without the hyperparameter must fall back to the default again.

    :param gene_list_data_dir: path to the temporary data directory
    """
    configured = MODEL_FACTORY["ElasticNet"]()
    configured.build_model(hyperparameters={"alpha": 1.0, "l1_ratio": 0.2, "gene_list": None})
    assert configured.gene_list is None
    assert sorted(_gene_names(configured.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _ALL_GENES
    )

    default = MODEL_FACTORY["ElasticNet"]()
    default.build_model(hyperparameters={"alpha": 1.0, "l1_ratio": 0.2})
    assert sorted(_gene_names(default.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _LANDMARK_GENES
    )
