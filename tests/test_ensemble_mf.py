"""Tests for EnsembleMF."""

import tempfile

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import MODEL_FACTORY


def _tiny_hpams() -> dict:
    """
    Shrink the expensive knobs so a smoke test runs in seconds.

    :returns: the shrunken hyperparameter dictionary
    """
    hp = dict(MODEL_FACTORY["EnsembleMF"].get_hyperparameter_set()[0])
    hp.update(
        n_ensemble=2,
        max_epochs=3,
        patience=2,
        hidden_dim=32,
        emb_dim=16,
        mlp_hidden=16,
        batch_size=64,
        n_bits=128,  # the toy data ships only 128-bit fingerprints
        gene_list=None,  # gene_expression_intersection is not in the toy meta bundle
    )
    return hp


def test_ensemble_mf_in_factory() -> None:
    """The model is registered in the factory."""
    assert "EnsembleMF" in MODEL_FACTORY


def test_ensemble_mf_train_predict_save_load(data_dir, sample_dataset) -> None:
    """
    The model fits the measured TOYv1 responses, predicts, and round-trips through save/load.

    :param data_dir: path to the test data directory (session fixture from conftest)
    :param sample_dataset: measured TOYv1 responses (session fixture from conftest)
    """
    model = MODEL_FACTORY["EnsembleMF"]()
    model.build_model(_tiny_hpams())
    cell_input = model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path=str(data_dir), dataset_name="TOYv1")

    cl_ids = np.asarray(sample_dataset.cell_line_ids)
    dr_ids = np.asarray(sample_dataset.drug_ids)
    responses = np.asarray(sample_dataset.response, dtype=float)
    train = DrugResponseDataset(response=responses, cell_line_ids=cl_ids, drug_ids=dr_ids, dataset_name="TOYv1")

    model.train(output=train, cell_line_input=cell_input, drug_input=drug_input)
    preds = model.predict(cell_line_ids=cl_ids, drug_ids=dr_ids, cell_line_input=cell_input, drug_input=drug_input)
    assert preds.shape == cl_ids.shape
    assert np.isfinite(preds).all()
    # trained on real responses, so the predictions have to track them; a collapsed or untrained
    # model would not clear this
    assert np.corrcoef(preds, responses)[0, 1] > 0.2

    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)
        reloaded = type(model).load(directory)
    preds_reloaded = reloaded.predict(
        cell_line_ids=cl_ids, drug_ids=dr_ids, cell_line_input=cell_input, drug_input=drug_input
    )
    assert np.allclose(preds, preds_reloaded, atol=1e-4)


def test_unknown_ids_fall_back_to_the_training_mean(data_dir, sample_dataset) -> None:
    """
    Cell lines and drugs absent from training are scored with the training mean, not an error.

    :param data_dir: path to the test data directory (session fixture from conftest)
    :param sample_dataset: measured TOYv1 responses (session fixture from conftest)
    """
    model = MODEL_FACTORY["EnsembleMF"]()
    model.build_model(_tiny_hpams())
    cell_input = model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path=str(data_dir), dataset_name="TOYv1")
    train = DrugResponseDataset(
        response=np.asarray(sample_dataset.response, dtype=float),
        cell_line_ids=np.asarray(sample_dataset.cell_line_ids),
        drug_ids=np.asarray(sample_dataset.drug_ids),
        dataset_name="TOYv1",
    )
    model.train(output=train, cell_line_input=cell_input, drug_input=drug_input)

    preds = model.predict(
        cell_line_ids=np.array(["not-a-cell-line"]),
        drug_ids=np.array(["not-a-drug"]),
        cell_line_input=cell_input,
        drug_input=drug_input,
    )
    assert preds.shape == (1,)
    assert np.isclose(preds[0], model.training_mean)
