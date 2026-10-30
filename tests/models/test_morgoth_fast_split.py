"""Tests of the vectorized split search for MORGOTH (``drevalpy.models.Morgoth.fast_split``)."""

import numpy as np
import pytest

pytest.importorskip("morgoth")

from drevalpy.models.Morgoth.fast_split import fast_split  # noqa: E402
from drevalpy.models.Morgoth.morgoth import MorgothRegressor  # noqa: E402


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
        with fast_split():
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


def _split(func, node, rk, rr, feature, min_leaf=1, best_score=1.0):
    """Call a split-search function of the package on one feature with the common arguments.

    :param func: ``calculate_best_split_for_feature`` (original or fast)
    :param node: node with data
    :param rk: root classification criterion
    :param rr: root regression criterion
    :param feature: feature name
    :param min_leaf: minimum number per leaf
    :param best_score: score to beat
    :returns: (score, split, valid_split_found)
    """
    import morgoth.multivariate_dt as dt

    return func(
        node,
        root_sample_weights=node.sample_weights,
        min_number_of_samples_per_leaf=min_leaf,
        class_criterion_function=dt.gini_impurity,
        regression_criterion_function=dt.MSE,
        root_reg_criterion=rr,
        root_class_criterion=rk,
        impact_classification=0.5,
        feature_name=feature,
        best_split=None,
        best_score=best_score,
        valid_split_found=False,
        node_score=1.0,
    )


def test_fast_split_edge_cases_behave_like_the_original():
    """Constant feature, too large leaf size, nothing better than the given score, unsupported weights or NaN."""
    dt, node, rk, rr = _node(40, 0, False, False)
    original = dt.BinaryTreeNode.calculate_best_split_for_feature
    with fast_split():
        fast = dt.BinaryTreeNode.calculate_best_split_for_feature

    node.X_train["const"] = 1.0
    for feature, kwargs in [("const", {}), ("a", {"min_leaf": 30}), ("a", {"best_score": -1.0})]:
        a = _split(original, node, rk, rr, feature, **kwargs)
        b = _split(fast, node, rk, rr, feature, **kwargs)
        assert a[1] is None and b[1] is None and a[0] == b[0] and a[2] == b[2]

    node.sample_weights = node.sample_weights.copy()
    node.sample_weights[0] = 0.0  # unsupported by the fast search -> falls back to the package
    a = _split(original, node, rk, rr, "a")
    b = _split(fast, node, rk, rr, "a")
    assert a[2] == b[2] and a[1].threshold == b[1].threshold


def test_whole_forest_with_fast_split_equals_original():
    """Same random sequence -> same predictions with and without the fast split search."""
    rng = np.random.RandomState(3)
    x = rng.normal(size=(60, 8))
    y = x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.3, size=60)
    predictions = []
    for fast in (False, True):
        regressor = MorgothRegressor(n_estimators=4, max_depth=5, n_jobs=1, fast_split=fast).fit(x, y)
        predictions.append(regressor.predict(x))
    assert np.allclose(predictions[0], predictions[1])


def test_package_split_search_is_restored():
    """The patch is only active inside the context, also after an error."""
    import morgoth.multivariate_dt as dt

    original = dt.BinaryTreeNode.calculate_best_split_for_feature
    with fast_split():
        assert dt.BinaryTreeNode.calculate_best_split_for_feature is not original
    assert dt.BinaryTreeNode.calculate_best_split_for_feature is original
    with pytest.raises(ZeroDivisionError), fast_split():
        1 / 0
    assert dt.BinaryTreeNode.calculate_best_split_for_feature is original
    with fast_split(False):
        assert dt.BinaryTreeNode.calculate_best_split_for_feature is original
