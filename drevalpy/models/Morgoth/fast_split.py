"""
Vectorized split search for the trees of the ``morgoth`` package.

The package searches the best split of a feature in pure Python and recomputes the Gini impurity and the sum of
squares for every candidate threshold (roughly O(n^2) per node and feature). :func:`fast_split` swaps this method for
an equivalent one that sorts the values once and scores all candidates via cumulative sums.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np


def build_fast_search(dt_module: Any) -> Any:
    """
    Build the vectorized replacement for ``BinaryTreeNode.calculate_best_split_for_feature``.

    For each feature, the package tries every midpoint of two adjacent values and recomputes Gini and
    weighted sum of squares in Python loops each time (roughly O(n^2) per node and feature). Here the
    values are sorted once and all candidates are scored via cumulative sums (O(n log n)).
    The result is mathematically the same: same thresholds (midpoints), same partition
    (``x < threshold``), same scoring and the same tie rule (the **last** best candidate wins,
    ``act_score <= best_score``). Differences only arise from rounding noise due to the summation
    order. For anything other than Gini + MSE (or weights <= 0, NaN) the original function is used.

    :param dt_module: the module ``morgoth.multivariate_dt``
    :returns: replacement function with the signature of the original method
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
        """
        Best split of a feature via sorted cumulative sums.

        :param self: the node
        :param root_sample_weights: weights of the root's bootstrap sample
        :param min_number_of_samples_per_leaf: minimum number per leaf
        :param class_criterion_function: classification criterion of the package
        :param regression_criterion_function: regression criterion of the package
        :param root_reg_criterion: regression normalization
        :param root_class_criterion: class normalization
        :param impact_classification: share of the classification
        :param feature_name: feature under examination
        :param best_split: best split so far
        :param best_score: score of the best split so far
        :param valid_split_found: whether a valid split has already been found
        :param node_score: score of the node without a split
        :returns: ``(best_score, best_split, valid_split_found)`` as in the original
        """
        class_on = self.output_format != "regression"
        reg_on = self.output_format != "classification"
        w = np.asarray(self.sample_weights, dtype=float)
        x = np.asarray(self.X_train[feature_name].to_numpy(), dtype=float)
        is_simple = (
            (not class_on or class_criterion_function is dt_module.gini_impurity)
            and (not reg_on or regression_criterion_function is dt_module.MSE)
            and len(w) == len(x)
            and bool(np.all(w > 0))
            and not np.isnan(x).any()
        )
        if not is_simple:
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
        xs, ws = x[order], w[order]
        n_left = np.searchsorted(xs, thresholds, side="left")  # number of cases with x < threshold
        valid = (n_left >= min_number_of_samples_per_leaf) & (n - n_left >= min_number_of_samples_per_leaf)
        if not valid.any():
            return best_score, best_split, valid_split_found

        def sides(field: np.ndarray) -> tuple:
            """Sums left/right per candidate (prefix and suffix respectively, without subtraction).

            :param field: values sorted by ``x``
            :returns: sums left and right per candidate
            """
            prefix = np.concatenate([[0.0], np.cumsum(field)])
            suffix = np.concatenate([np.cumsum(field[::-1])[::-1], [0.0]])
            return prefix[n_left], suffix[n_left]

        w_l, w_r = sides(ws)
        w_total = np.sum(w)
        left_share, right_share = w_l / w_total, w_r / w_total
        class_term = np.zeros(len(thresholds))
        reg_term = np.zeros(len(thresholds))
        if class_on and root_class_criterion != 0:
            ys = np.asarray(self.y_train_class)[order]
            gini_l = np.zeros(len(thresholds))
            gini_r = np.zeros(len(thresholds))
            for class_name in self.class_names:
                c_l, c_r = sides(ws * (ys == class_name))
                pk_l, pk_r = c_l / w_l, c_r / w_r
                gini_l += pk_l * (1 - pk_l)
                gini_r += pk_r * (1 - pk_r)
            class_term = (left_share * gini_l + right_share * gini_r) / root_class_criterion
        if reg_on:
            yr = np.asarray(self.y_train_reg, dtype=float)[order]
            yr = yr - yr.mean()  # RSS is shift-invariant; centering dampens the cancellation
            wy_l, wy_r = sides(ws * yr)
            wyy_l, wyy_r = sides(ws * yr * yr)
            ny_l, ny_r = sides(yr)  # unweighted sum: the package centers with the unweighted mean
            m_l, m_r = ny_l / n_left, ny_r / (n - n_left)
            rss_l = wyy_l - 2 * m_l * wy_l + m_l * m_l * w_l
            rss_r = wyy_r - 2 * m_r * wy_r + m_r * m_r * w_r
            reg_term = (left_share * (rss_l / w_l) + right_share * (rss_r / w_r)) / root_reg_criterion
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


@contextlib.contextmanager
def fast_split(enabled: bool = True) -> Iterator[None]:
    """
    Replace the package's split search at runtime and restore it afterwards.

    The child processes of the ``Pool`` inherit the replacement (fork).

    :param enabled: False = keep the original
    :yields: nothing
    """
    if not enabled:
        yield
        return
    import morgoth.multivariate_dt as dt_module

    node_class = dt_module.BinaryTreeNode
    original = node_class.calculate_best_split_for_feature
    node_class.calculate_best_split_for_feature = build_fast_search(dt_module)
    try:
        yield
    finally:
        node_class.calculate_best_split_for_feature = original
