"""Vectorized split search for the trees of the ``morgoth`` package.

The package searches the best split of a feature in pure Python and recomputes the Gini impurity and the sum of
squares for every candidate threshold (roughly O(n^2) per node and feature). :func:`fast_split` swaps this method for
an equivalent one that sorts the values once and scores all candidates via cumulative sums.

:func:`alias_fireducks` lives here rather than in ``_forest.py`` so this module needs nothing from it (no import
cycle); ``_forest.import_morgoth`` calls it before importing the package.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np


def alias_fireducks() -> None:
    """Make ``import fireducks.pandas`` resolve to pandas when fireducks is not installed.

    An installed fireducks is left untouched; the reason is in the module docstring of ``_forest.py``.
    """
    try:
        importlib.import_module("fireducks.pandas")
    except ImportError:
        import pandas

        stub = sys.modules.setdefault("fireducks", types.ModuleType("fireducks"))
        sys.modules.setdefault("fireducks.pandas", pandas)
        stub.pandas = sys.modules["fireducks.pandas"]


def build_fast_search(dt_module: Any) -> Any:
    """Build the vectorized replacement for ``BinaryTreeNode.calculate_best_split_for_feature``.

    For each feature, the package tries every midpoint of two adjacent values and recomputes Gini and
    weighted sum of squares in Python loops each time (roughly O(n^2) per node and feature). Here the
    values are sorted once and all candidates are scored via cumulative sums (O(n log n)).
    The result is mathematically the same: same thresholds (midpoints), same partition
    (``x < threshold``), same scoring and the same tie rule (the **last** best candidate wins,
    ``act_score <= best_score``). Differences only arise from rounding noise due to the summation
    order. For anything other than Gini + MSE (or weights <= 0, NaN) the original function is used.

    :param dt_module: The module ``morgoth.multivariate_dt``.
    :returns: Replacement function with the signature of the original method.
    """
    original = dt_module.BinaryTreeNode.calculate_best_split_for_feature

    def fast(
        self: Any,
        root_sample_weights: Any,
        min_number_of_samples_per_leaf: int,
        class_criterion_function: Any,
        regression_criterion_function: Any,
        root_reg_criterion: float,
        root_class_criterion: float,
        impact_classification: float,
        feature_name: str,
        best_split: Any,
        best_score: float,
        valid_split_found: bool,
        node_score: float,
    ) -> tuple:
        """Best split of a feature via sorted cumulative sums.

        :param self: The node.
        :param root_sample_weights: Weights of the root's bootstrap sample.
        :param min_number_of_samples_per_leaf: Minimum number of samples per leaf.
        :param class_criterion_function: Classification criterion of the package.
        :param regression_criterion_function: Regression criterion of the package.
        :param root_reg_criterion: Regression normalization.
        :param root_class_criterion: Class normalization.
        :param impact_classification: Share of the classification in the score.
        :param feature_name: Feature under examination.
        :param best_split: Best split so far.
        :param best_score: Score of the best split so far.
        :param valid_split_found: Whether a valid split has already been found.
        :param node_score: Score of the node without a split.
        :returns: ``(best_score, best_split, valid_split_found)`` as in the original.
        """
        class_on = self.output_format != "regression"
        reg_on = self.output_format != "classification"
        w = np.asarray(self.sample_weights, dtype=float)
        x = np.asarray(self.X_train[feature_name].to_numpy(), dtype=float)
        criteria = (class_criterion_function, regression_criterion_function)
        if not _vectorizable(dt_module, self.output_format, criteria, w, x):
            return original(
                self,
                root_sample_weights=root_sample_weights,
                min_number_of_samples_per_leaf=min_number_of_samples_per_leaf,
                class_criterion_function=class_criterion_function,
                regression_criterion_function=regression_criterion_function,
                root_reg_criterion=root_reg_criterion,
                root_class_criterion=root_class_criterion,
                impact_classification=impact_classification,
                feature_name=feature_name,
                best_split=best_split,
                best_score=best_score,
                valid_split_found=valid_split_found,
                node_score=node_score,
            )
        n = len(x)
        values = np.unique(x)
        if len(values) < 2:
            return best_score, best_split, valid_split_found
        thresholds = (values[:-1] + values[1:]) / 2.0  # like np.mean([a, b])
        order = np.argsort(x, kind="stable")
        ws = w[order]
        n_left = np.searchsorted(x[order], thresholds, side="left")  # number of samples with x < threshold
        valid = (n_left >= min_number_of_samples_per_leaf) & (n - n_left >= min_number_of_samples_per_leaf)
        if not valid.any():
            return best_score, best_split, valid_split_found

        w_l, w_r = _sides(ws, n_left)
        w_total = np.sum(w)
        shares = (w_l / w_total, w_r / w_total)
        class_term = np.zeros(len(thresholds))
        reg_term = np.zeros(len(thresholds))
        if class_on and root_class_criterion != 0:
            ys = np.asarray(self.y_train_class)[order]
            class_term = _gini_term(ys, ws, n_left, self.class_names, shares) / root_class_criterion
        if reg_on:
            yr = np.asarray(self.y_train_reg, dtype=float)[order]
            reg_term = _rss_term(yr, ws, n_left, shares) / root_reg_criterion
        act = impact_classification * class_term + (1 - impact_classification) * reg_term
        act = np.where(valid, act, np.inf)
        valid_split_found = True
        smallest = act.min()
        if not smallest <= best_score:
            return best_score, best_split, valid_split_found
        i = int(np.nonzero(act == smallest)[0][-1])  # on a tie the last one wins, as in the original
        split = dt_module.Split(feature_name=feature_name, threshold=thresholds[i])
        split.score = (node_score - smallest) * np.sum(w) / np.sum(root_sample_weights)
        return float(smallest), split, valid_split_found

    return fast


def _vectorizable(dt_module: Any, output_format: str, criteria: tuple[Any, Any], w: np.ndarray, x: np.ndarray) -> bool:
    """Whether the vectorized search applies: Gini and MSE, one positive weight per sample, no NaN values.

    :param dt_module: The module ``morgoth.multivariate_dt``.
    :param output_format: Output format of the tree.
    :param criteria: Classification and regression criterion of the package.
    :param w: Sample weights of the node.
    :param x: Values of the feature.
    :returns: True when the vectorized search gives the package's result.
    """
    simple_class = output_format == "regression" or criteria[0] is dt_module.gini_impurity
    simple_reg = output_format == "classification" or criteria[1] is dt_module.MSE
    return simple_class and simple_reg and len(w) == len(x) and bool(np.all(w > 0)) and not np.isnan(x).any()


def _sides(field: np.ndarray, n_left: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum left and right of each candidate (prefix and suffix sums, without subtraction).

    :param field: Values sorted by the feature.
    :param n_left: Number of samples left of each candidate.
    :returns: Sums left and right per candidate.
    """
    prefix = np.concatenate([[0.0], np.cumsum(field)])
    suffix = np.concatenate([np.cumsum(field[::-1])[::-1], [0.0]])
    return prefix[n_left], suffix[n_left]


def _gini_term(
    ys: np.ndarray, ws: np.ndarray, n_left: np.ndarray, class_names: Any, shares: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Weighted Gini impurity of both sides of each candidate, before the root normalization.

    :param ys: Classes sorted by the feature.
    :param ws: Sample weights sorted by the feature.
    :param n_left: Number of samples left of each candidate.
    :param class_names: Classes of the tree.
    :param shares: Weight shares of the left and right side per candidate.
    :returns: One impurity per candidate.
    """
    w_l, w_r = _sides(ws, n_left)
    gini_l = np.zeros(len(n_left))
    gini_r = np.zeros(len(n_left))
    for class_name in class_names:
        c_l, c_r = _sides(ws * (ys == class_name), n_left)
        pk_l, pk_r = c_l / w_l, c_r / w_r
        gini_l += pk_l * (1 - pk_l)
        gini_r += pk_r * (1 - pk_r)
    return shares[0] * gini_l + shares[1] * gini_r


def _rss_term(yr: np.ndarray, ws: np.ndarray, n_left: np.ndarray, shares: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Weighted mean sum of squares of both sides of each candidate, before the root normalization.

    :param yr: Responses sorted by the feature.
    :param ws: Sample weights sorted by the feature.
    :param n_left: Number of samples left of each candidate.
    :param shares: Weight shares of the left and right side per candidate.
    :returns: One score per candidate.
    """
    n = len(yr)
    yr = yr - yr.mean()  # RSS is shift-invariant; centering dampens the cancellation
    w_l, w_r = _sides(ws, n_left)
    wy_l, wy_r = _sides(ws * yr, n_left)
    wyy_l, wyy_r = _sides(ws * yr * yr, n_left)
    ny_l, ny_r = _sides(yr, n_left)  # unweighted sum: the package centers with the unweighted mean
    m_l, m_r = ny_l / n_left, ny_r / (n - n_left)
    rss_l = wyy_l - 2 * m_l * wy_l + m_l * m_l * w_l
    rss_r = wyy_r - 2 * m_r * wy_r + m_r * m_r * w_r
    return shares[0] * (rss_l / w_l) + shares[1] * (rss_r / w_r)


@contextlib.contextmanager
def fast_split(enabled: bool = True) -> Iterator[None]:
    """Replace the package's split search at runtime and restore it afterwards.

    The child processes of the ``Pool`` inherit the replacement only under fork.

    :param enabled: False keeps the original split search.
    :yields: Nothing; the replacement is active inside the block.
    """
    if not enabled:
        yield
        return
    alias_fireducks()
    dt_module = importlib.import_module("morgoth.multivariate_dt")

    node_class = dt_module.BinaryTreeNode
    original = node_class.calculate_best_split_for_feature
    node_class.calculate_best_split_for_feature = build_fast_search(dt_module)
    try:
        yield
    finally:
        node_class.calculate_best_split_for_feature = original
