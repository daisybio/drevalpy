"""Tests for RidgeMLPEnsemble."""

import tempfile

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.RidgeMLPEnsemble import RidgeMLPEnsemble


def _tiny_model() -> RidgeMLPEnsemble:
    """
    Build the ensemble with a small MLP so a smoke test runs in seconds.

    :returns: the built model
    """
    model = RidgeMLPEnsemble()
    hp = dict(RidgeMLPEnsemble.get_hyperparameter_set()[0])
    hp["gene_list"] = None  # the toy meta bundle has no landmark gene list
    hp.update(mlp_n_ensemble=2, mlp_max_epochs=3, mlp_patience=2, mlp_hidden=16)
    model.build_model(hp)
    return model


def _toy_responses(sample_dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack the measured TOYv1 responses.

    :param sample_dataset: measured TOYv1 responses
    :returns: cell line ids, drug ids, responses
    """
    return (
        np.asarray(sample_dataset.cell_line_ids),
        np.asarray(sample_dataset.drug_ids),
        np.asarray(sample_dataset.response, dtype=float),
    )


def test_ridge_mlp_ensemble_in_factory() -> None:
    """The model is registered in the factory and has no drug views."""
    assert MODEL_FACTORY["RidgeMLPEnsemble"] is RidgeMLPEnsemble
    assert RidgeMLPEnsemble.drug_views == []
    assert RidgeMLPEnsemble.early_stopping


def test_ridge_mlp_ensemble_train_predict_save_load(data_dir, sample_dataset) -> None:
    """
    The ensemble fits TOYv1, predicts the mean of its components, and round-trips through save/load.

    :param data_dir: path to the test data directory (session fixture from conftest)
    :param sample_dataset: measured TOYv1 responses (session fixture from conftest)
    """
    model = _tiny_model()
    cell_input = model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
    assert model.load_drug_features(data_path=str(data_dir), dataset_name="TOYv1") is None

    cl_ids, dr_ids, responses = _toy_responses(sample_dataset)
    keep = np.isin(cl_ids, cell_input.identifiers)
    cl_ids, dr_ids, responses = cl_ids[keep], dr_ids[keep], responses[keep]
    cells = np.unique(cl_ids)
    stop = np.isin(cl_ids, cells[: max(2, len(cells) // 5)])  # a few cell lines for early stopping
    train = DrugResponseDataset(
        response=responses[~stop], cell_line_ids=cl_ids[~stop], drug_ids=dr_ids[~stop], dataset_name="TOYv1"
    )
    early_stopping = DrugResponseDataset(
        response=responses[stop], cell_line_ids=cl_ids[stop], drug_ids=dr_ids[stop], dataset_name="TOYv1"
    )

    model.train(output=train, cell_line_input=cell_input, output_earlystopping=early_stopping)
    preds = model.predict(cell_line_ids=cl_ids, drug_ids=dr_ids, cell_line_input=cell_input)
    assert preds.shape == cl_ids.shape
    assert np.isfinite(preds).all()
    parts = [part.predict(cl_ids, dr_ids, cell_input) for part in model.parts]
    assert np.allclose(preds, np.mean(parts, axis=0))
    # trained on real responses, so the in-sample predictions have to track them
    assert np.corrcoef(preds[~stop], responses[~stop])[0, 1] > 0.2

    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)
        reloaded = RidgeMLPEnsemble.load(directory)
    assert reloaded.hyperparameters["gene_list"] is None
    preds_reloaded = reloaded.predict(cell_line_ids=cl_ids, drug_ids=dr_ids, cell_line_input=cell_input)
    assert np.allclose(preds, preds_reloaded, atol=1e-4)


def test_unknown_drugs_fall_back_to_the_training_mean(data_dir, sample_dataset) -> None:
    """
    Drugs absent from training are predicted with the training mean instead of raising an error.

    :param data_dir: path to the test data directory (session fixture from conftest)
    :param sample_dataset: measured TOYv1 responses (session fixture from conftest)
    """
    model = _tiny_model()
    cell_input = model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
    cl_ids, dr_ids, responses = _toy_responses(sample_dataset)
    keep = np.isin(cl_ids, cell_input.identifiers)
    cl_ids, dr_ids, responses = cl_ids[keep], dr_ids[keep], responses[keep]
    train = DrugResponseDataset(response=responses, cell_line_ids=cl_ids, drug_ids=dr_ids, dataset_name="TOYv1")
    model.train(output=train, cell_line_input=cell_input)
    preds = model.predict(cell_line_ids=cl_ids[:3], drug_ids=np.array(["not_a_drug"] * 3), cell_line_input=cell_input)
    # each component falls back to the mean of the responses it was trained on
    expected = np.mean([part.state["training_mean"] for part in model.parts])
    assert np.isfinite(preds).all()
    assert np.allclose(preds, expected)


def _parts(model: RidgeMLPEnsemble) -> dict:
    """
    Name the components of a built ensemble.

    :param model: the ensemble
    :returns: component name -> component
    """
    return {part.get_model_name(): part for part in model.parts}


def test_components_get_their_prefixed_settings() -> None:
    """Each component receives the ensemble settings with its prefix, without the prefix, plus the seed."""
    model = RidgeMLPEnsemble()
    hp = dict(RidgeMLPEnsemble.get_hyperparameter_set()[0])
    hp.update(mlp_dropout=0.4, seed=3)
    model.build_model(hp)
    parts = _parts(model)
    assert parts["MultiTaskMLP"].hyperparameters["dropout"] == 0.4
    assert parts["MultiTaskMLP"].hyperparameters["hidden"] == 512
    assert parts["MultiTaskMLP"].hyperparameters["seed"] == 3
    assert "mlp_dropout" not in parts["MultiTaskMLP"].hyperparameters
    assert parts["PerDrugRidge"].hyperparameters == {"seed": 3}


def test_first_grid_combination_is_the_untuned_model() -> None:
    """The tuning grid has 27 combinations, and the first is the untuned (submitted) MLP configuration."""
    grid = RidgeMLPEnsemble.get_hyperparameter_set()
    assert len(grid) == 27
    model = RidgeMLPEnsemble()
    model.build_model(grid[0])
    assert _parts(model)["MultiTaskMLP"].hyperparameters == {
        "hidden": 512,
        "depth": 2,
        "dropout": 0.2,
        "lr": 0.001,
        "batch_size": 64,
        "max_epochs": 200,
        "patience": 10,
        "n_ensemble": 10,
        "seed": 0,
    }
