"""Tests of :class:`SingleDrugMorgoth`, the wrapper around the ``morgoth`` package.

What is tested is the wrapper, not the package: that the settings arrive correctly, that fit and
prediction run on drevalpy data, that the target (response value) and not the class is returned,
that the fallback kicks in for an empty class, that ``save``/``load`` yield the same predictions
and that the package's hard-wired ``Pool`` is back in its initial state afterwards. Without the
optional package the tests are skipped.
"""

import numpy as np
import pytest

pytest.importorskip("morgoth")

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset  # noqa: E402
from drevalpy.models import MODEL_FACTORY  # noqa: E402
from drevalpy.models.baselines import morgoth as module  # noqa: E402
from drevalpy.models.baselines.morgoth import SingleDrugMorgoth  # noqa: E402

_SMALL = dict(n_estimators=6, min_samples_leaf=3, max_depth=6, n_jobs=2, mrmr_k=5, gene_list=None)


def _model(**hpams) -> SingleDrugMorgoth:
    """Build a model with small defaults for fast tests.

    :param hpams: hyperparameters, override the small defaults
    :returns: built model
    """
    model = SingleDrugMorgoth()
    model.build_model({**_SMALL, "cell_line_views": ["gene_expression"], **hpams})
    model.cell_line_views = ["gene_expression"]
    return model


def _data(n=60, p=12, seed=3, constant=False):
    """Generate features and responses; gene g0 drives the response.

    :param n: number of cell lines
    :param p: number of genes
    :param seed: random seed
    :param constant: True = response without variance (only one class)
    :returns: responses and features
    """
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, p))
    y = np.zeros(n) if constant else 2.0 * x[:, 0] + rng.normal(scale=0.3, size=n)
    cells = np.array([f"cl{i}" for i in range(n)])
    features = FeatureDataset(
        features={cl: {"gene_expression": x[i]} for i, cl in enumerate(cells)},
        meta_info={"gene_expression": [f"g{j}" for j in range(p)]},
    )
    output = DrugResponseDataset(response=y, cell_line_ids=cells, drug_ids=np.array(["d1"] * n))
    return output, features


def test_registered_and_name():
    """The model is in the factory and reports its name."""
    assert MODEL_FACTORY["SingleDrugMorgoth"] is SingleDrugMorgoth
    assert SingleDrugMorgoth.get_model_name() == "SingleDrugMorgoth"
    assert SingleDrugMorgoth.is_single_drug_model


@pytest.mark.parametrize(
    "hpams, word",
    [
        ({"upsampling": True}, "upsampling"),
        ({"sample_weights": "quadratic"}, "sample_weights"),
        ({"criterion_class": "x"}, "criterion_class"),
        ({"criterion_reg": "mae"}, "criterion_reg"),
        ({"impact_classification": 1.5}, "impact_classification"),
        ({"max_features": "all"}, "max_features"),
        ({"max_features": 0.0}, "max_features"),
    ],
)
def test_invalid_settings(hpams, word):
    """Unusable settings abort during build with a named message.

    :param hpams: hyperparameters
    :param word: word that must appear in the message
    """
    with pytest.raises(ValueError, match=word):
        _model(**hpams)


def test_breiman_default_is_mapped_to_sqrt():
    """The sklearn value ``breiman_default`` does not exist in the package; it becomes ``sqrt``."""
    assert _model(max_features="breiman_default").max_features == "sqrt"


def test_missing_package_names_the_installation(monkeypatch):
    """Without the package, installation instructions are raised instead of a bare ImportError.

    :param monkeypatch: pytest helper
    """
    import builtins

    real = builtins.__import__

    def blocker(name, *args, **kwargs):
        if name.startswith("morgoth"):
            raise ImportError("gone")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocker)
    with pytest.raises(ImportError, match="pip install morgoth"):
        SingleDrugMorgoth().build_model({"cell_line_views": ["gene_expression"]})


def test_fit_and_predict_return_the_response_value(tmp_path):
    """The response value (not the class) is returned, and it follows the driving gene.

    :param tmp_path: pytest path
    """
    output, features = _data()
    model = _model()
    model.train(output=output, cell_line_input=features, model_checkpoint_dir=str(tmp_path))
    prediction = model.predict(cell_line_ids=output.cell_line_ids, drug_ids=output.drug_ids, cell_line_input=features)
    assert prediction.shape == (len(output),)
    assert np.isfinite(prediction).all()
    assert len(np.unique(prediction)) > 2  # not the 0/1 of the class
    assert np.corrcoef(prediction, output.response)[0, 1] > 0.5
    assert set(np.unique(model.class_predictions_)) <= {0.0, 1.0}
    assert model.threshold_ is not None
    assert len(model.mrmr_genes_) == 5 and "g0" in model.mrmr_genes_
    assert model.fallback_ is None


def test_package_pool_is_restored(tmp_path):
    """The process count is only set during the call.

    :param tmp_path: pytest path
    """
    import morgoth.morgoth as package

    before = package.Pool
    output, features = _data(n=30)
    model = _model(n_estimators=2)
    model.train(output=output, cell_line_input=features, model_checkpoint_dir=str(tmp_path))
    assert package.Pool is before


def test_pool_is_restored_on_error_too():
    """An error in the package does not leave the pool set.

    :raises RuntimeError: deliberately, inside the block
    """
    import morgoth.morgoth as package

    before = package.Pool
    with pytest.raises(RuntimeError):
        with module._pool_with(package, 2):
            assert package.Pool is not before
            raise RuntimeError("x")
    assert package.Pool is before


def test_fallback_on_empty_class(tmp_path):
    """A fold without variance has no second class; the model carries on without weights.

    :param tmp_path: pytest path
    """
    output, features = _data(n=30, constant=True)
    model = _model(n_estimators=2, mrmr_k=None)
    model.train(output=output, cell_line_input=features, model_checkpoint_dir=str(tmp_path))
    assert model.fallback_ is not None
    prediction = model.predict(cell_line_ids=output.cell_line_ids, drug_ids=output.drug_ids, cell_line_input=features)
    assert np.allclose(prediction, 0.0)


def test_duplicate_cell_lines_are_no_problem(tmp_path):
    """The same cell line several times in the rows (replicates) does not break the package's bookkeeping.

    :param tmp_path: pytest path
    """
    output, features = _data(n=40)
    duplicated = DrugResponseDataset(
        response=np.concatenate([output.response, output.response[:10]]),
        cell_line_ids=np.concatenate([output.cell_line_ids, output.cell_line_ids[:10]]),
        drug_ids=np.array(["d1"] * 50),
    )
    model = _model(n_estimators=2)
    model.train(output=duplicated, cell_line_input=features, model_checkpoint_dir=str(tmp_path))
    prediction = model.predict(
        cell_line_ids=duplicated.cell_line_ids, drug_ids=duplicated.drug_ids, cell_line_input=features
    )
    assert prediction.shape == (50,)


def test_save_and_load_give_the_same_predictions(tmp_path):
    """After ``save``/``load``, predictions, threshold and feature selection are unchanged.

    :param tmp_path: pytest path
    """
    output, features = _data()
    model = _model(n_estimators=4)
    model.train(output=output, cell_line_input=features, model_checkpoint_dir=str(tmp_path))
    expected = model.predict(cell_line_ids=output.cell_line_ids, drug_ids=output.drug_ids, cell_line_input=features)
    model.save(str(tmp_path / "m"))
    loaded = SingleDrugMorgoth.load(str(tmp_path / "m"))
    assert loaded.threshold_ == model.threshold_
    assert loaded.mrmr_genes_ == model.mrmr_genes_
    same = loaded.predict(cell_line_ids=output.cell_line_ids, drug_ids=output.drug_ids, cell_line_input=features)
    assert np.allclose(same, expected)


def _node(n, seed, integer_valued, weighted, output_format="multioutput"):
    """Build a node of the package with random data (optionally with many identical values).

    :param n: number of cases
    :param seed: random seed
    :param integer_valued: True = features with few distinct values (ties)
    :param weighted: True = unequal case weights
    :param output_format: ``multioutput``, ``classification`` or ``regression``
    :returns: node and auxiliary quantities (classes, normalizations)
    """
    import morgoth.multivariate_dt as dt

    rng = np.random.RandomState(seed)
    x = rng.randint(0, 4, size=(n, 3)).astype(float) if integer_valued else rng.normal(size=(n, 3))
    cls = (x[:, 0] + rng.normal(scale=0.7, size=n) > 0.5).astype(float)
    reg = x[:, 1] + 0.5 * cls + rng.normal(scale=0.4, size=n)
    w = rng.uniform(0.5, 3.0, size=n) if weighted else np.ones(n)
    frame = dt.pd.DataFrame(x, columns=["a", "b", "c"])
    y = {"multioutput": np.column_stack([reg, cls]), "classification": cls, "regression": reg}[output_format]
    node = dt.BinaryTreeNode(
        None,
        0,
        True,
        X_train=frame,
        y_train=y,
        sample_weights=w,
        sample_names_train=np.arange(n),
        output_format=output_format,
    )
    if output_format == "regression":
        node.y_train_class = np.zeros(n)
    if output_format == "classification":
        node.y_train_reg = np.zeros(n)
    node.class_names = np.array([0.0, 1.0]) if output_format != "regression" else [0]
    root_class = dt.gini_impurity(node.class_names, node.y_train_class, w) if output_format != "regression" else 1
    root_reg = dt.MSE(node.y_train_reg, w) if output_format != "classification" else 1
    return dt, node, root_class, root_reg


@pytest.mark.parametrize("output_format", ["multioutput", "classification", "regression"])
@pytest.mark.parametrize("integer_valued", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("min_leaf", [1, 5])
def test_fast_split_search_equals_original(output_format, integer_valued, weighted, min_leaf):
    """The vectorized search picks the same split (threshold) per feature as the original loop.

    :param output_format: output format of the node
    :param integer_valued: features with ties
    :param weighted: unequal case weights
    :param min_leaf: minimum number per leaf
    """
    for seed in range(4):
        dt, node, rk, rr = _node(40, seed, integer_valued, weighted, output_format)
        ic = {"multioutput": 0.5, "classification": 1.0, "regression": 0.0}[output_format]
        original = dt.BinaryTreeNode.calculate_best_split_for_feature
        with module._fast_search():
            fast = dt.BinaryTreeNode.calculate_best_split_for_feature
        assert fast is not original
        assert dt.BinaryTreeNode.calculate_best_split_for_feature is original  # restored
        node_score = 1.0 if output_format != "multioutput" else 1.0
        for feature in ["a", "b", "c"]:
            res = []
            for func in (original, fast):
                res.append(
                    func(
                        node,
                        root_sample_weights=node.sample_weights,
                        min_number_of_samples_per_leaf=min_leaf,
                        class_criterion_function=dt.gini_impurity,
                        regression_criterion_function=dt.MSE,
                        root_reg_criterion=rr,
                        root_class_criterion=rk,
                        impact_classification=ic,
                        feature_name=feature,
                        best_split=None,
                        best_score=node_score,
                        valid_split_found=False,
                        node_score=node_score,
                    )
                )
            (s0, b0, v0), (s1, b1, v1) = res
            assert v0 == v1
            assert (b0 is None) == (b1 is None)
            if b0 is not None:
                assert b0.threshold == b1.threshold
                assert np.isclose(s0, s1, rtol=1e-9, atol=1e-12)
                assert np.isclose(b0.score, b1.score, rtol=1e-9, atol=1e-12)


def test_whole_forest_with_fast_search_equals_original(tmp_path):
    """Same random sequence -> same predictions with and without the fast split search.

    :param tmp_path: pytest path
    """
    output, features = _data(n=60)
    results = []
    for fast in (False, True):
        model = _model(fast_split=fast, n_estimators=4, n_jobs=1)
        model.train(output=output, cell_line_input=features, model_checkpoint_dir=str(tmp_path))
        results.append(
            model.predict(cell_line_ids=output.cell_line_ids, drug_ids=output.drug_ids, cell_line_input=features)
        )
    assert np.allclose(results[0], results[1])


def test_model_is_self_contained():
    """The model derives from the generic sklearn base only, not from another forest model."""
    from drevalpy.models.baselines.sklearn_models import SklearnModel

    assert SklearnModel in SingleDrugMorgoth.__mro__
    assert not any("sauron" in c.__module__ for c in SingleDrugMorgoth.__mro__)


def test_mrmr_select_finds_driver_and_skips_constant_columns():
    """The driving column comes first; constant columns are never chosen; bad input raises."""
    rng = np.random.RandomState(0)
    x = rng.normal(size=(80, 6))
    x[:, 3] = 1.0
    y = 3.0 * x[:, 2] + rng.normal(scale=0.1, size=80)
    chosen = module.mrmr_select(x, y, 4)
    assert chosen[0] == 2 and 3 not in chosen and len(set(chosen)) == len(chosen)
    assert module.mrmr_select(x, np.zeros(80), 3) == []
    with pytest.raises(ValueError, match="finite"):
        module.mrmr_select(x, np.full(80, np.nan), 3)


def test_absolute_threshold_and_side():
    """Absolute threshold is taken as given; the quantile follows the sensitive side."""
    y = np.arange(100, dtype=float)
    absolute = _model(sensitivity_threshold_mode="absolute", sensitivity_threshold=12.5)
    assert absolute._derive_threshold(y) == 12.5
    low = _model(sensitivity_quantile=0.2)
    high = _model(sensitivity_quantile=0.2, sensitive_side="high")
    assert np.isclose(low._derive_threshold(y), np.quantile(y, 0.2))
    assert np.isclose(high._derive_threshold(y), np.quantile(y, 0.8))
    with pytest.raises(ValueError, match="sensitivity_threshold"):
        _model(sensitivity_threshold_mode="absolute")
