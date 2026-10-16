"""Tests of the mRMR feature selection and of :class:`SingleDrugKNN`."""

import tempfile

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import MODEL_FACTORY, SINGLE_DRUG_MODEL_FACTORY
from drevalpy.models.baselines.singledrug_baselines import SingleDrugKNN
from drevalpy.models.utils import MRMRSelector

_HPAMS = {"cell_line_views": ["gene_expression"], "gene_list": None, "n_neighbors": 3, "weights": "distance"}


def _data(n: int = 40, p: int = 12, seed: int = 1):
    """
    Generate features and responses; gene g0 drives the response.

    :param n: number of cell lines
    :param p: number of genes
    :param seed: random seed
    :returns: responses and features
    """
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, p))
    y = 2.0 * x[:, 0] + rng.normal(scale=0.3, size=n)
    cells = np.array([f"cl{i}" for i in range(n)])
    features = FeatureDataset(
        features={cl: {"gene_expression": x[i]} for i, cl in enumerate(cells)},
        meta_info={"gene_expression": [f"g{j}" for j in range(p)]},
    )
    return DrugResponseDataset(response=y, cell_line_ids=cells, drug_ids=np.array(["d1"] * n)), features


def _model(**hpams) -> SingleDrugKNN:
    """
    Build a model with small defaults.

    :param hpams: hyperparameters, override the defaults
    :returns: built model
    """
    model = SingleDrugKNN()
    model.build_model({**_HPAMS, **hpams})
    model.cell_line_views = ["gene_expression"]
    return model


def test_mrmr_selector_finds_driver_and_skips_constant_columns():
    """The driving column comes first, columns without variance are never chosen."""
    rng = np.random.RandomState(0)
    x = rng.normal(size=(50, 6))
    x[:, 3] = 1.0
    y = 3.0 * x[:, 2] + rng.normal(scale=0.1, size=50)
    selector = MRMRSelector(k=4).fit(x, y)
    assert selector.selected_[0] == 2
    assert 3 not in selector.selected_
    assert len(selector.selected_) == 4
    assert selector.transform(x).shape == (50, 4)


def test_mrmr_selector_penalizes_redundancy():
    """A copy of the best column ranks behind an independent informative column."""
    rng = np.random.RandomState(0)
    a, b = rng.normal(size=(2, 200))
    x = np.column_stack([a, a + 1e-3 * rng.normal(size=200), b, rng.normal(size=200)])
    y = a + 0.8 * b
    selected = MRMRSelector(k=2).fit(x, y).selected_
    assert 2 in selected
    assert len({0, 1} & set(selected)) == 1


def test_mrmr_selector_edge_cases():
    """Non-finite input and k < 1 fail loudly, k is capped, no relevance keeps all columns."""
    x = np.random.RandomState(0).normal(size=(10, 3))
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="finite"):
        MRMRSelector(k=2).fit(x, np.append(y[:-1], np.nan))
    with pytest.raises(ValueError, match="at least 1"):
        MRMRSelector(k=0).fit(x, y)
    assert len(MRMRSelector(k=50).fit(x, y).selected_) == 3
    assert list(MRMRSelector(k=2).fit(x, np.ones(10)).selected_) == [0, 1, 2]


def test_registered_as_single_drug_model():
    """The model is in both factories and reports its name."""
    assert MODEL_FACTORY["SingleDrugKNN"] is SingleDrugKNN
    assert SINGLE_DRUG_MODEL_FACTORY["SingleDrugKNN"] is SingleDrugKNN
    assert SingleDrugKNN.get_model_name() == "SingleDrugKNN"
    assert SingleDrugKNN.is_single_drug_model
    assert _model().drug_views == []
    assert _model().load_drug_features("unused", "unused") is None


def test_trains_on_selected_genes_and_predicts():
    """With mRMR the driver gene is selected and predictions follow the response."""
    output, features = _data()
    model = _model(mrmr_k=3)
    model.train(output=output, cell_line_input=features)
    selected = model.model.named_steps["mrmr"].selected_
    assert len(selected) == 3 and selected[0] == 0
    pred = model.predict(output.cell_line_ids, output.drug_ids, features)
    assert pred.shape == (len(output),)
    assert np.corrcoef(pred, output.response)[0, 1] > 0.9


def test_without_selection_uses_all_genes():
    """mrmr_k = None keeps the plain kNN on all genes."""
    output, features = _data()
    model = _model(mrmr_k=None)
    model.train(output=output, cell_line_input=features)
    assert model.model.named_steps["mrmr"] == "passthrough"
    assert model.model.named_steps["knn"].n_features_in_ == 12


def test_fewer_cell_lines_than_neighbors():
    """The number of neighbors is reduced instead of failing."""
    output, features = _data(n=40)
    output.reduce_to(cell_line_ids=output.cell_line_ids[:2], drug_ids=None)
    model = _model(n_neighbors=10, mrmr_k=3)
    model.train(output=output, cell_line_input=features)
    assert model.model.named_steps["knn"].n_neighbors == 2
    assert len(model.predict(output.cell_line_ids, output.drug_ids, features)) == 2


def test_empty_training_data_predicts_nan():
    """No training rows: the prediction is NaN, as for the other sklearn models."""
    output, features = _data()
    empty = DrugResponseDataset(response=np.array([]), cell_line_ids=np.array([]), drug_ids=np.array([]))
    model = _model()
    model.train(output=empty, cell_line_input=features)
    assert np.isnan(model.predict(output.cell_line_ids, output.drug_ids, features)).all()


def test_save_and_load_give_same_predictions():
    """The selection is part of the saved model."""
    output, features = _data()
    model = _model(mrmr_k=4)
    model.train(output=output, cell_line_input=features)
    expected = model.predict(output.cell_line_ids, output.drug_ids, features)
    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)
        loaded = SingleDrugKNN.load(directory)
    assert isinstance(loaded, SingleDrugKNN)
    loaded.cell_line_views = ["gene_expression"]
    np.testing.assert_allclose(loaded.predict(output.cell_line_ids, output.drug_ids, features), expected)
