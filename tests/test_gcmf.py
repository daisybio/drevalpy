"""Smoke tests for the GCMF model family (GCMF, PGCMF, RGCMF, PRGCMF)."""

import tempfile

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import MODEL_FACTORY

GCMF_FAMILY = ["GCMF", "PGCMF", "RGCMF", "PRGCMF"]


def test_gcmf_family_in_factory() -> None:
    """All four GCMF-family models are registered in the model factory."""
    for name in GCMF_FAMILY:
        assert name in MODEL_FACTORY


def test_default_relations_ship_with_meta_bundle(data_dir, sample_dataset) -> None:
    """
    Every resource the default RGCMF relations read is present in the downloaded meta bundle.

    Guards the data dependency directly, on both sides: the drug relations under
    ``meta/gcmf_drug_relations/`` and the per-omics gene lists the cell relations reduce with.
    If any is ever dropped from the bundle this fails here with a clear message instead of
    surfacing as a training error downstream.

    :param data_dir: path to the test data directory (session fixture from conftest)
    :param sample_dataset: session fixture; ensures the toy data and meta bundle are downloaded
    """
    from drevalpy.models.GCMF.gcmf import RGCMF

    model = RGCMF()
    for view in model.drug_relation_views:
        assert RGCMF._drug_resource_path(view, data_path=str(data_dir)) is not None, (
            f"drug relation '{view}' is missing from <data>/meta/{RGCMF._DRUG_SIM_DIR}/; "
            "it must ship with the meta bundle"
        )
    # the gene_expression relation is built from the node features (hp gene_list), so only the
    # omics views loaded from disk need their gene list to be present
    for view in model.cell_relation_views:
        if view == "gene_expression":
            continue
        gene_list = RGCMF._CELL_GENE_LISTS[view]
        path = data_dir / "meta" / "gene_lists" / f"{gene_list}.csv"
        assert path.is_file(), (
            f"gene list '{gene_list}' for cell relation '{view}' is missing from "
            "<data>/meta/gene_lists/; it must ship with the meta bundle"
        )


def _tiny_hpams(model_cls) -> dict:
    """
    First hyperparameter set with the expensive knobs shrunk so a smoke test runs fast.

    :param model_cls: the GCMF-family model class to take the hyperparameter set from
    :returns: the shrunken hyperparameter dictionary
    """
    hp = dict(model_cls.get_hyperparameter_set()[0])
    hp.update(
        n_ensemble=1,
        max_epochs=2,
        patience=2,
        hidden_dim=32,
        emb_dim=16,
        mlp_hidden=16,
        batch_size=64,
        k_cell=4,
        k_drug=4,
        n_bits=128,  # toy data ships only 128-bit fingerprints
    )
    # The relational models keep their real relation defaults on both sides (four cell relations;
    # drug_pathways + drug_bioassay) and their real gene list, so the test exercises every
    # resource they read from the meta bundle and fails if any of them goes missing.
    # The base models default to landmark_genes, which the toy meta bundle does not ship, so only
    # they need the gene list relaxed to all available genes.
    if "cell_relation_views" not in hp:
        hp["gene_list"] = None
    return hp


@pytest.mark.parametrize("model_name", GCMF_FAMILY)
def test_gcmf_family_train_predict_save_load(model_name: str, data_dir) -> None:
    """
    Each model trains on TOYv1, predicts finite values, and round-trips through save/load.

    :param model_name: name of the GCMF-family model under test
    :param data_dir: path to the test data directory (session fixture from conftest)
    """
    data_path = str(data_dir)
    model = MODEL_FACTORY[model_name]()
    model.build_model(_tiny_hpams(type(model)))
    cell_input = model.load_cell_line_features(data_path=data_path, dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path=data_path, dataset_name="TOYv1")

    cells = list(cell_input.identifiers)[:8]
    drugs = list(drug_input.identifiers)[:5]
    cl_ids = np.array([c for c in cells for _ in drugs])
    dr_ids = np.array([d for _ in cells for d in drugs])
    responses = np.random.default_rng(0).normal(size=len(cl_ids)).astype(float)
    train = DrugResponseDataset(response=responses, cell_line_ids=cl_ids, drug_ids=dr_ids, dataset_name="TOYv1")

    model.train(output=train, cell_line_input=cell_input, drug_input=drug_input)
    preds = model.predict(cell_line_ids=cl_ids, drug_ids=dr_ids, cell_line_input=cell_input, drug_input=drug_input)
    assert preds.shape == cl_ids.shape
    assert np.isfinite(preds).all()

    # save -> load -> predict must reproduce the same predictions
    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)
        reloaded = type(model).load(directory)
    preds_reloaded = reloaded.predict(
        cell_line_ids=cl_ids, drug_ids=dr_ids, cell_line_input=cell_input, drug_input=drug_input
    )
    assert np.allclose(preds, preds_reloaded, atol=1e-4)


def test_rgcmf_drug_relation_resource_resolution(tmp_path) -> None:
    """A drug relation resolves from ``<data_path>/meta/<dir>/`` and is None when absent.

    :param tmp_path: pytest-provided temporary directory
    """
    from drevalpy.models.GCMF.gcmf import RGCMF

    assert RGCMF._drug_resource_path("drug_pathways", data_path=str(tmp_path)) is None

    meta = tmp_path / "meta" / RGCMF._DRUG_SIM_DIR
    meta.mkdir(parents=True)
    (meta / "drug_pathways.csv.gz").write_bytes(b"")  # presence is enough for path resolution
    resolved = RGCMF._drug_resource_path("drug_pathways", data_path=str(tmp_path))
    assert resolved is not None and resolved.endswith("drug_pathways.csv.gz")


def test_rgcmf_raises_on_relation_without_pubchem_id(tmp_path, data_dir) -> None:
    """A relation table lacking the pubchem_id join key is rejected instead of guessed at.

    :param tmp_path: pytest-provided temporary directory
    :param data_dir: path to the test data directory (source of the toy dataset)
    """
    import shutil

    import pandas as pd

    from drevalpy.models.GCMF.gcmf import RGCMF

    shutil.copytree(data_dir / "TOYv1", tmp_path / "TOYv1")
    shutil.copytree(data_dir / "meta", tmp_path / "meta")
    rel_dir = tmp_path / "meta" / RGCMF._DRUG_SIM_DIR
    for stale in rel_dir.glob("drug_pathways.csv*"):
        stale.unlink()
    # an old-style table keyed only by drug name
    pd.DataFrame({"drug_name": ["Daporinad", "Axitinib"], "pathway": ["PW0", "PW0"]}).to_csv(
        rel_dir / "drug_pathways.csv", index=False
    )

    model = RGCMF()
    model.build_model(_tiny_hpams(RGCMF))
    with pytest.raises(ValueError, match="pubchem_id"):
        model.load_drug_features(data_path=str(tmp_path), dataset_name="TOYv1")


def test_rgcmf_raises_on_missing_drug_relation(tmp_path, data_dir) -> None:
    """RGCMF raises (no silent fallback) when a configured drug-relation resource is absent.

    :param tmp_path: pytest-provided temporary directory (its meta has no relation resource)
    :param data_dir: path to the test data directory (source of the toy dataset)
    """
    import shutil

    from drevalpy.models.GCMF.gcmf import RGCMF

    shutil.copytree(data_dir / "TOYv1", tmp_path / "TOYv1")
    shutil.copytree(data_dir / "meta", tmp_path / "meta")
    rel_dir = tmp_path / "meta" / RGCMF._DRUG_SIM_DIR
    if rel_dir.exists():
        shutil.rmtree(rel_dir)

    model = RGCMF()
    model.build_model(_tiny_hpams(RGCMF))
    model.load_cell_line_features(data_path=str(tmp_path), dataset_name="TOYv1")
    with pytest.raises(FileNotFoundError):
        model.load_drug_features(data_path=str(tmp_path), dataset_name="TOYv1")
