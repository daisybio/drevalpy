"""Tests for the configurable gene list of the DIPK model."""

import json

import pytest

from drevalpy.models import MODEL_FACTORY
from drevalpy.models.DIPK.dipk import DIPKModel

_GENE_EXPRESSION = (
    "cellosaurus_id,cell_line_name,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
    "CVCL_1104,CAL-120,7.63,2.96,10.38,3.61,3.38,7.09,3.02\n"
    "CVCL_1174,DMS 114,7.55,2.78,11.81,4.07,3.73,2.80,6.08\n"
    "CVCL_1110,CAL-51,8.71,2.64,9.88,3.96,3.24,11.39,4.22\n"
)
#: Symbols of the two gene lists written by the gene_list_data_dir fixture.
_INTERSECTION_GENES = ["BRCA1", "SCYL3", "HDAC1", "INSIG1"]
_LANDMARK_GENES = ["TSPAN6", "BRCA1", "FOXO3"]
_ALL_GENES = ["TSPAN6", "TNMD", "BRCA1", "SCYL3", "HDAC1", "INSIG1", "FOXO3"]

#: Minimal hyperparameters that build a DIPK Predictor, the gene list is independent of them.
_HPAMS = {"heads": 1, "fc_layer_num": 3, "fc_layer_dim": [256, 128, 64, 32, 16, 1], "dropout_rate": 0.0}


@pytest.fixture
def gene_list_data_dir(tmp_path) -> str:
    """
    Build a minimal data directory with everything DIPK's cell line loader reads.

    :param tmp_path: pytest temporary path fixture
    :returns: path to the data directory
    """
    dataset_dir = tmp_path / "GDSC1_small"
    dataset_dir.mkdir()
    (dataset_dir / "gene_expression.csv").write_text(_GENE_EXPRESSION, encoding="utf-8")

    dipk_features = dataset_dir / "DIPK_features"
    dipk_features.mkdir()
    (dipk_features / "gene_list_sel.txt").write_text("\n".join(_ALL_GENES) + "\n", encoding="utf-8")
    (dipk_features / "human_ppi_features.tsv").write_text(
        "gene\tbionic_0\tbionic_1\n" + "".join(f"{gene}\t{i / 10}\t{i / 5}\n" for i, gene in enumerate(_ALL_GENES)),
        encoding="utf-8",
    )

    gene_lists = tmp_path / "meta" / "gene_lists"
    gene_lists.mkdir(parents=True)
    (gene_lists / "gene_expression_intersection.csv").write_text(
        "Symbol\n" + "\n".join(_INTERSECTION_GENES) + "\n", encoding="utf-8"
    )
    (gene_lists / "landmark_genes_reduced.csv").write_text(
        "Symbol\n" + "\n".join(_LANDMARK_GENES) + "\n", encoding="utf-8"
    )
    return str(tmp_path)


def _gene_names(features) -> list[str]:
    """
    Extract the gene symbols of the gene_expression view.

    :param features: FeatureDataset returned by a loader
    :returns: gene symbols
    """
    assert features.meta_info is not None
    return list(features.meta_info["gene_expression"])


def _build(hyperparameters: dict | None = None) -> DIPKModel:
    """
    Instantiate and build a DIPK model.

    :param hyperparameters: extra hyperparameters merged into the minimal set
    :returns: the built model
    """
    model = DIPKModel()
    model.build_model(hyperparameters={**_HPAMS, **(hyperparameters or {})})
    return model


def test_dipk_gene_list_default() -> None:
    """Without the hyperparameter the model keeps the previously hard-coded gene list."""
    assert MODEL_FACTORY["DIPK"] is DIPKModel
    assert DIPKModel.gene_list == "gene_expression_intersection"
    assert _build().gene_list == "gene_expression_intersection"


def test_dipk_default_gene_list_is_used_when_loading(gene_list_data_dir) -> None:
    """
    The default has to reproduce the behaviour before the gene list became configurable.

    :param gene_list_data_dir: path to the temporary data directory
    """
    features = _build().load_cell_line_features(data_path=gene_list_data_dir, dataset_name="GDSC1_small")
    assert sorted(_gene_names(features)) == sorted(_INTERSECTION_GENES)
    # The bionic view is still there, the gene list only touches gene_expression.
    assert "bionic_features" in features.view_names


def test_dipk_gene_list_hyperparameter(gene_list_data_dir) -> None:
    """
    The "gene_list" hyperparameter has to reach load_cell_line_features.

    :param gene_list_data_dir: path to the temporary data directory
    """
    model = _build({"gene_list": "landmark_genes_reduced"})
    assert model.gene_list == "landmark_genes_reduced"
    # It stays in the hyperparameters, so save()/load() carry it over to predict().
    assert model.hyperparameters["gene_list"] == "landmark_genes_reduced"

    features = model.load_cell_line_features(data_path=gene_list_data_dir, dataset_name="GDSC1_small")
    assert sorted(_gene_names(features)) == sorted(_LANDMARK_GENES)
    # The class default is untouched, only the instance was reconfigured.
    assert DIPKModel.gene_list == "gene_expression_intersection"


def test_dipk_gene_list_none_loads_all_genes(gene_list_data_dir) -> None:
    """
    gene_list=None has to load the full expression matrix.

    :param gene_list_data_dir: path to the temporary data directory
    """
    features = _build({"gene_list": None}).load_cell_line_features(
        data_path=gene_list_data_dir, dataset_name="GDSC1_small"
    )
    assert sorted(_gene_names(features)) == sorted(_ALL_GENES)


def test_dipk_gene_list_does_not_leak_between_instances(gene_list_data_dir) -> None:
    """
    A second model built without the hyperparameter must fall back to the default again.

    :param gene_list_data_dir: path to the temporary data directory
    """
    configured = _build({"gene_list": "landmark_genes_reduced"})
    assert sorted(_gene_names(configured.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _LANDMARK_GENES
    )

    default = _build()
    assert sorted(_gene_names(default.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _INTERSECTION_GENES
    )


def test_dipk_gene_list_survives_the_save_load_round_trip(gene_list_data_dir) -> None:
    """
    load() rebuilds from the saved hyperparameters, so the restored model must use the same genes.

    :param gene_list_data_dir: path to the temporary data directory
    """
    trained = _build({"gene_list": "landmark_genes_reduced"})
    # This is what save() writes and load() reads back before calling build_model().
    restored = DIPKModel()
    restored.build_model(json.loads(json.dumps(trained.hyperparameters)))

    assert restored.gene_list == "landmark_genes_reduced"
    assert sorted(_gene_names(restored.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _LANDMARK_GENES
    )
