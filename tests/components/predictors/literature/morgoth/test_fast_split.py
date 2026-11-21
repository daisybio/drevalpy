"""Tests of the vectorized split search for MORGOTH (``literature/morgoth/_fast_split.py``)."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from drevalpy.components.predictors.literature.morgoth._fast_split import alias_fireducks, fast_split
from drevalpy.components.predictors.literature.morgoth._forest import MorgothRegressor


def _dt_module() -> Any:
    """Import ``morgoth.multivariate_dt`` behind the fireducks alias.

    :returns: The module ``morgoth.multivariate_dt``.
    """
    alias_fireducks()
    return importlib.import_module("morgoth.multivariate_dt")


def _node(
    n: int, seed: int, integer_valued: bool, weighted: bool, output_format: str = "multioutput"
) -> tuple[Any, Any, float, float]:
    """Build a node of the package with random data (optionally with many identical values).

    :param n: Number of samples.
    :param seed: Random seed.
    :param integer_valued: True gives features with few distinct values (ties).
    :param weighted: True gives unequal sample weights.
    :param output_format: ``multioutput``, ``classification`` or ``regression``.
    :returns: The module, the node and the root classification and regression criteria.
    """
    dt = _dt_module()
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


def _split(
    func: Any,
    node: Any,
    rk: float,
    rr: float,
    feature: str,
    min_leaf: int = 1,
    best_score: float = 1.0,
    impact_classification: float = 0.5,
) -> tuple:
    """Call a split-search function of the package on one feature with the common arguments.

    :param func: ``calculate_best_split_for_feature``, original or fast.
    :param node: Node with data.
    :param rk: Root classification criterion.
    :param rr: Root regression criterion.
    :param feature: Feature name.
    :param min_leaf: Minimum number of samples per leaf.
    :param best_score: Score to beat.
    :param impact_classification: Share of the classification in the score.
    :returns: ``(score, split, valid_split_found)``.
    """
    dt = _dt_module()
    return func(
        node,
        root_sample_weights=node.sample_weights,
        min_number_of_samples_per_leaf=min_leaf,
        class_criterion_function=dt.gini_impurity,
        regression_criterion_function=dt.MSE,
        root_reg_criterion=rr,
        root_class_criterion=rk,
        impact_classification=impact_classification,
        feature_name=feature,
        best_split=None,
        best_score=best_score,
        valid_split_found=False,
        node_score=1.0,
    )


@pytest.mark.slow
@pytest.mark.parametrize("output_format", ["multioutput", "classification", "regression"])
@pytest.mark.parametrize("integer_valued", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("min_leaf", [1, 5])
def test_fast_split_search_equals_original(
    output_format: str, integer_valued: bool, weighted: bool, min_leaf: int
) -> None:
    """The vectorized search picks the same split (threshold) per feature as the original loop.

    :param output_format: Output format of the node.
    :param integer_valued: Features with ties.
    :param weighted: Unequal sample weights.
    :param min_leaf: Minimum number of samples per leaf.
    """
    impact = {"multioutput": 0.5, "classification": 1.0, "regression": 0.0}[output_format]
    for seed in range(4):
        dt, node, rk, rr = _node(40, seed, integer_valued, weighted, output_format)
        original = dt.BinaryTreeNode.calculate_best_split_for_feature
        with fast_split():
            fast = dt.BinaryTreeNode.calculate_best_split_for_feature
        assert fast is not original
        for feature in ["a", "b", "c"]:
            s0, b0, v0 = _split(original, node, rk, rr, feature, min_leaf=min_leaf, impact_classification=impact)
            s1, b1, v1 = _split(fast, node, rk, rr, feature, min_leaf=min_leaf, impact_classification=impact)
            assert v0 == v1
            assert (b0 is None) == (b1 is None)
            if b0 is not None:
                assert b0.threshold == b1.threshold
                assert np.isclose(s0, s1, rtol=1e-9, atol=1e-12)
                assert np.isclose(b0.score, b1.score, rtol=1e-9, atol=1e-12)


def test_fast_split_edge_cases_behave_like_the_original() -> None:
    """Constant feature, too large leaf size, nothing better than the given score, unsupported weights."""
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
    node.sample_weights[0] = 0.0  # unsupported by the fast search, which falls back to the package
    a = _split(original, node, rk, rr, "a")
    b = _split(fast, node, rk, rr, "a")
    assert a[2] == b[2] and a[1].threshold == b[1].threshold


def test_package_split_search_is_restored() -> None:
    """The patch is only active inside the context, also after an error."""
    dt = _dt_module()
    original = dt.BinaryTreeNode.calculate_best_split_for_feature
    with fast_split():
        assert dt.BinaryTreeNode.calculate_best_split_for_feature is not original
    assert dt.BinaryTreeNode.calculate_best_split_for_feature is original
    with pytest.raises(ZeroDivisionError), fast_split():
        _ = 1 / 0
    assert dt.BinaryTreeNode.calculate_best_split_for_feature is original
    with fast_split(False):
        assert dt.BinaryTreeNode.calculate_best_split_for_feature is original


@pytest.mark.slow
def test_whole_forest_with_fast_split_equals_original() -> None:
    """Same random sequence gives the same predictions with and without the fast split search."""
    rng = np.random.RandomState(3)
    x = rng.normal(size=(60, 8))
    y = x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.3, size=60)
    predictions = [
        MorgothRegressor(n_estimators=4, max_depth=5, n_jobs=1, fast_split=fast).fit(x, y).predict(x)
        for fast in (False, True)
    ]
    assert np.allclose(predictions[0], predictions[1])
