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
        gene_list=None,  # toy data lacks most landmark genes; use all available genes
        n_bits=128,  # toy data ships only 128-bit fingerprints
    )
    # For the relational models, keep a single (fast) cell relation built from the node features
    # but the bundled drug relations, so the relational encoder + the gzipped package resources
    # are both exercised. (The full multi-omics cell relations are validated in the benchmark.)
    if "cell_relation_views" in hp:
        hp["cell_relation_views"] = ["gene_expression"]
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


def test_rgcmf_loads_bundled_drug_relations() -> None:
    """The default RGCMF drug relations resolve to the gzipped resources bundled in the package."""
    from drevalpy.models.GCMF.gcmf import RGCMF

    for view in ["drug_pathways", "drug_bioassay"]:
        path = RGCMF._drug_resource_path(view, data_path="/nonexistent")
        assert path is not None and path.endswith(".csv.gz")
