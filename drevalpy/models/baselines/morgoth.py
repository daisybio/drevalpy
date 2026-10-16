"""MORGOTH as a single-drug model (``volkamerlab/MORGOTH``).

MORGOTH (Multivariate classificatiOn and Regression increasinG trustwOrTHiness) is a random
forest whose trees learn the class (sensitive/resistant) and the response value
**simultaneously**. The split criterion is a linear combination of classification and
regression error; ``impact_classification`` sets the share of the classification (0 = regression
only, 1 = classification only). The trees are classes of the ``morgoth`` package and cannot be
mapped onto sklearn, so this model **wraps the package** instead of re-implementing it.

Source and citation:

*  Code: https://github.com/volkamerlab/MORGOTH
*  Paper: Rolli, L.-M.; Eckhart, L.; Herrmann, L.; Volkamer, A.; Lenhof, H.-P.; Lenhof, K.
   *Increasing trustworthiness of machine learning-based drug sensitivity prediction with a
   multivariate random forest approach.* Digital Discovery **2026**, 5 (4), 1746-1764.
   doi:10.1039/D5DD00284B.

The model is self-contained: it does not depend on any other forest model of this repository. Around
the package it provides the per-drug sensitivity threshold derived from the training fold only
(``quantile`` or ``absolute``), an optional mRMR feature selection on the training fold only (with
named storage), and ``save``/``load``. The package itself does the trees, the weighting of the
minority class (``simple``/``linear``) and the class-aware aggregation (``tree_weights``).

The response value is predicted (regression); the predicted class is then available in
``class_predictions_``. Conformal prediction, leaf graphs and silhouettes of the package are
**not** used -- they are file outputs for the analysis, not part of the point prediction.

Practical points worth knowing:

*  **Optional dependency.** ``pip install morgoth`` (PyPI 1.0 to 1.6; tested with version 1.5 from
   GitHub). On Linux it pulls in ``fireducks``. Without the package, ``build_model`` aborts with
   installation instructions.
*  **Runtime.** The trees of the package are pure Python and its split search is quadratic. This
   model replaces the search at run time by an equivalent vectorized one (``fast_split``, on by
   default; verified against the original loop in the tests), which makes a tree about 50 times
   faster. Without it, a tree takes about 21 s for 493 rows x 100 features. ``mrmr_k`` defaults
   to 100 as in the paper (50 to 100 genes); without selection, 11,883 genes would be infeasible.
*  **Cores.** The package hard-wires ``multiprocessing.Pool()`` with all cores. Here the number
   of processes is set via ``n_jobs`` (``-1`` = all).
*  **Files.** ``MORGOTH.fit`` necessarily writes four log files. They end up in a temporary
   directory that is removed after the fit.
*  **Fallbacks.** If a fold has fewer than two cases in one of the two classes, the model runs
   without class weights and without ``tree_weights`` (noted in ``fallback_``) instead of
   failing in the package.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import tempfile
from collections.abc import Iterator
from functools import partial
from typing import Any

import joblib
import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..utils import scale_gene_expression
from .sklearn_models import SklearnModel

#: Weighting modes known to the package (``none`` = no weights).
MORGOTH_WEIGHT_MODES = ("none", "simple", "linear")

#: Split criteria for classification and regression according to the package.
CRITERIA_CLASS = ("gini", "hellinger", "entropy")
CRITERIA_REG = ("mse",)

#: Installation hint for the optional dependency.
INSTALL_HINT = "SingleDrugMorgoth needs the package 'morgoth' (volkamerlab/MORGOTH). Installation: pip install morgoth"


def _import_morgoth() -> Any:
    """
    Import the ``morgoth`` package lazily.

    :returns: the module ``morgoth.morgoth`` that holds the class ``MORGOTH`` and the ``Pool``
    :raises ImportError: if the package is not installed
    """
    try:
        import morgoth.morgoth as morgoth_module
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    return morgoth_module


@contextlib.contextmanager
def _pool_with(morgoth_module: Any, processes: int) -> Iterator[None]:
    """
    Set the process count of the hard-wired ``Pool()`` in the package and restore it afterwards.

    :param morgoth_module: the module ``morgoth.morgoth``
    :param processes: number of processes
    :yields: nothing
    """
    original = morgoth_module.Pool
    morgoth_module.Pool = partial(multiprocessing.Pool, processes=processes)
    try:
        yield
    finally:
        morgoth_module.Pool = original


def _build_fast_search(dt_module: Any) -> Any:
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
def _fast_search(enabled: bool = True) -> Iterator[None]:
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
    node_class.calculate_best_split_for_feature = _build_fast_search(dt_module)
    try:
        yield
    finally:
        node_class.calculate_best_split_for_feature = original


#: Allowed ways to derive the sensitivity threshold from the training responses.
THRESHOLD_MODES = ("quantile", "absolute")

#: Lower bound of the redundancy term in the mRMR score (avoids dividing by ~0).
MRMR_FLOOR = 0.001

#: Name of the file with the fitted extra state next to ``model.pkl``.
STATE_FILE = "morgoth_state.pkl"


def mrmr_select(x: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    """
    Select ``k`` columns by greedy mRMR (F-statistic relevance over mean absolute correlation).

    Relevance is the F-statistic of the univariate linear fit of ``y`` on a column; redundancy is
    the mean absolute Pearson correlation to the columns chosen so far (floored at
    :data:`MRMR_FLOOR`). Columns without variance have no relevance and are never chosen.

    :param x: design matrix (rows x columns), finite
    :param y: response vector of the same rows, finite
    :param k: number of columns to select
    :returns: column indices in selection order; empty if no column has any relevance
    :raises ValueError: if ``x`` or ``y`` contains non-finite values
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("mRMR needs finite values in x and y; filter missing responses first.")
    n, p = x.shape
    centered = x - x.mean(axis=0)
    norm = np.sqrt((centered**2).sum(axis=0))
    varies = norm > 0
    y_centered = y - y.mean()
    y_norm = np.sqrt((y_centered**2).sum())

    r = np.zeros(p)
    if y_norm > 0:
        np.divide(centered.T @ y_centered, norm * y_norm, out=r, where=varies)
    r2 = np.clip(r**2, 0.0, 1.0 - 1e-15)
    relevance = np.where(varies, r2 / (1.0 - r2) * (n - 2), 0.0)

    unit = np.zeros_like(centered)
    np.divide(centered, norm, out=unit, where=varies)  # unit columns: correlation = dot product
    candidates = [j for j in range(p) if relevance[j] > 0]
    chosen: list[int] = []
    redundancy_sum = np.zeros(p)
    for _ in range(min(k, len(candidates))):
        cand = np.array(candidates, dtype=int)
        if chosen:
            denominator = redundancy_sum[cand] / len(chosen)
            denominator = np.where(denominator == 1.0, np.inf, denominator)
            score = relevance[cand] / denominator
        else:
            score = relevance[cand]
        best = int(cand[int(np.argmax(score))])
        chosen.append(best)
        candidates.remove(best)
        redundancy_sum += np.maximum(np.abs(unit.T @ unit[:, best]), MRMR_FLOOR)
    return chosen


class SingleDrugMorgoth(SklearnModel):
    """MORGOTH: multivariate random forest (class + response) per drug."""

    is_single_drug_model = True
    drug_views: list = []
    early_stopping = False
    gene_list: str | None = "landmark_genes"

    @classmethod
    def get_model_name(cls) -> str:
        """
        Return the model name.

        :returns: SingleDrugMorgoth
        """
        return "SingleDrugMorgoth"

    def __init__(self):
        """Initialize the model; the forest is only created during the fit."""
        super().__init__()
        #: sensitivity threshold derived from the training responses
        self.threshold_: float | None = None
        #: predicted class (1 = sensitive) of the last prediction, per case
        self.class_predictions_: np.ndarray | None = None
        #: note if the fit fell back to the simple path
        self.fallback_: str | None = None
        #: number of features to select by mRMR; None = no selection
        self.mrmr_k: int | None = None
        #: selected feature names in selection order (fitted state)
        self.mrmr_genes_: list[str] | None = None
        #: column indices of the selection in the full feature space
        self.mrmr_idx_: np.ndarray | None = None
        #: names of the full feature space at fit time; None if there were no names
        self.mrmr_input_names_: list[str] | None = None
        #: number of columns of the full feature space at fit time
        self.n_features_full_: int | None = None
        #: feature names of the most recently passed feature dataset (transient)
        self._feature_names: list[str] | None = None

    def build_model(self, hyperparameters: dict) -> None:
        """
        Read the MORGOTH settings; the forest itself is created in :meth:`_fit_estimator`.

        :param hyperparameters: settings, see ``hyperparameters.yaml``
        :raises ValueError: on an unknown weighting, threshold or split criterion, an invalid
            ``impact_classification``, ``max_features`` or ``mrmr_k``, or an absolute threshold
            without a value
        """
        _import_morgoth()
        super().build_model(hyperparameters)
        self.drug_views = []
        hp = self.hyperparameters
        self.sample_weights = hp.get("sample_weights", "simple")
        if self.sample_weights not in MORGOTH_WEIGHT_MODES:
            raise ValueError(f"sample_weights must be one of {MORGOTH_WEIGHT_MODES}, not {self.sample_weights!r}.")
        if hp.get("upsampling", False):
            raise ValueError("SingleDrugMorgoth does not support upsampling; the minority class is weighted.")
        self.sensitive_side = hp.get("sensitive_side", "low")
        if self.sensitive_side not in ("low", "high"):
            raise ValueError(f"sensitive_side must be 'low' or 'high', not {self.sensitive_side!r}.")
        self.threshold_mode = hp.get("sensitivity_threshold_mode", "quantile")
        if self.threshold_mode not in THRESHOLD_MODES:
            raise ValueError(
                f"sensitivity_threshold_mode must be one of {THRESHOLD_MODES}, not {self.threshold_mode!r}."
            )
        self.sensitivity_quantile = float(hp.get("sensitivity_quantile", 0.2))
        self.sensitivity_threshold = hp.get("sensitivity_threshold", None)
        if self.threshold_mode == "absolute" and self.sensitivity_threshold is None:
            raise ValueError("sensitivity_threshold_mode='absolute' needs sensitivity_threshold.")
        self.mrmr_k = self._read_mrmr_k(hp.get("mrmr_k", None))
        self.criterion_class = hp.get("criterion_class", "gini")
        if self.criterion_class not in CRITERIA_CLASS:
            raise ValueError(f"criterion_class must be one of {CRITERIA_CLASS}, not {self.criterion_class!r}.")
        self.criterion_reg = hp.get("criterion_reg", "mse")
        if self.criterion_reg not in CRITERIA_REG:
            raise ValueError(f"criterion_reg must be one of {CRITERIA_REG}, not {self.criterion_reg!r}.")
        self.impact_classification = float(hp.get("impact_classification", 0.5))
        if not 0.0 <= self.impact_classification <= 1.0:
            raise ValueError(f"impact_classification must be between 0 and 1, not {self.impact_classification}.")
        self.tree_weights = bool(hp.get("tree_weights", True))
        # The package knows 'sqrt', 'log2' or a fraction in (0, 1]; 'breiman_default' (sklearn) does not exist there.
        self.max_features = hp.get("max_features", "sqrt")
        if self.max_features == "breiman_default":
            self.max_features = "sqrt"
        if isinstance(self.max_features, str):
            if self.max_features not in ("sqrt", "log2"):
                raise ValueError(f"max_features must be 'sqrt', 'log2' or a fraction, not {self.max_features!r}.")
        elif not 0.0 < float(self.max_features) <= 1.0:
            raise ValueError(f"max_features as a number must be in (0, 1], not {self.max_features!r}.")
        self.min_samples_leaf = int(hp.get("min_samples_leaf", 10))
        self.max_depth_morgoth = int(hp.get("max_depth", 20))
        self.n_trees = int(hp.get("n_estimators", 500))
        n_jobs = int(hp.get("n_jobs", -1))
        self.n_processes = (os.cpu_count() or 1) if n_jobs < 1 else n_jobs
        self.random_state_morgoth = int(hp.get("random_state", 0))
        #: vectorized split search (Gini + MSE); False = original loop of the package
        self.fast_split = bool(hp.get("fast_split", True))

    @staticmethod
    def _read_mrmr_k(value: object) -> int | None:
        """
        Read ``mrmr_k``; any empty value switches the selection off.

        :param value: setting from the hyperparameters
        :returns: positive number or None
        :raises ValueError: if the value is not a positive integer
        """
        if value in (None, 0, False, "None", "none", ""):
            return None
        try:
            k = int(value)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            raise ValueError(f"mrmr_k must be a positive integer or None, not {value!r}.") from None
        if k < 1:
            raise ValueError(f"mrmr_k must be at least 1, not {k}.")
        return k

    def load_drug_features(self, data_path: str, dataset_name: str) -> None:
        """
        Single-drug model -- no drug features.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None

    # ------------------------------------------------------------------ Design matrix

    def _design_matrix(
        self,
        output_cell_line_ids: np.ndarray,
        output_drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        training: bool,
    ) -> np.ndarray:
        """
        Build the design matrix of one cell line view (gene expression is standardized).

        Also remembers the feature names of the view; they name the mRMR selection.

        :param output_cell_line_ids: cell line ids of the rows
        :param output_drug_ids: drug ids of the rows
        :param cell_line_input: feature dataset of the cell lines
        :param training: True fits the scaler, False applies the fitted one
        :returns: design matrix, one row per case
        """
        view = self.cell_line_views[0]
        names = cell_line_input.meta_info.get(view) if cell_line_input is not None else None
        self._feature_names = None if names is None else [str(n) for n in names]
        if view == "gene_expression":
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output_cell_line_ids),
                training=training,
                gene_expression_scaler=self.gene_expression_scaler,
            )
        return self.get_concatenated_features(
            cell_line_view=view,
            drug_view=None,
            cell_line_ids_output=output_cell_line_ids,
            drug_ids_output=output_drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Train the model on the cases of one drug.

        :param output: training dataset
        :param cell_line_input: features of the cell lines
        :param drug_input: unused (single-drug model)
        :param output_earlystopping: unused
        :param model_checkpoint_dir: unused
        """
        if len(output) == 0:
            print("No training data provided, will predict NA.")
            self.model = None
            return
        x = self._design_matrix(output.cell_line_ids, output.drug_ids, cell_line_input, training=True)
        self._fit_estimator(x, output)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predict the response value.

        :param cell_line_ids: cell line ids
        :param drug_ids: drug ids
        :param cell_line_input: features of the cell lines
        :param drug_input: unused (single-drug model)
        :returns: predictions of the response value (NA without a trained model)
        """
        if self.model is None:
            print("No training data was available, predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))
        x = self._design_matrix(cell_line_ids, drug_ids, cell_line_input, training=False)
        return self._predict_estimator(x)

    # ------------------------------------------------------------------ Threshold and mRMR

    def _derive_threshold(self, y: np.ndarray) -> float:
        """
        Derive the sensitivity threshold from the training responses only.

        :param y: responses of the training fold
        :returns: threshold; cases below it (``sensitive_side='low'``) or above it are sensitive
        """
        if self.threshold_mode == "absolute":
            return float(self.sensitivity_threshold)  # type: ignore[arg-type]
        q = self.sensitivity_quantile if self.sensitive_side == "low" else 1.0 - self.sensitivity_quantile
        return float(np.quantile(np.asarray(y, dtype=float), q))

    def _sensitive_mask(self, y: np.ndarray) -> np.ndarray:
        """
        Split the responses at the threshold into sensitive (True) and resistant (False).

        :param y: responses
        :returns: boolean mask, True = sensitive
        """
        if self.sensitive_side == "low":
            return np.asarray(y) < self.threshold_
        return np.asarray(y) > self.threshold_

    def _column_names(self, n_features: int) -> tuple[list[str], bool]:
        """
        Return column names for the design matrix and whether they are real feature names.

        :param n_features: number of columns of the design matrix
        :returns: (names, True if real feature names)
        :raises ValueError: on duplicate gene names, since a named selection would be ambiguous
        """
        names = self._feature_names
        if names is None or len(names) != n_features:
            return [f"column_{i}" for i in range(n_features)], False
        if len(set(names)) != len(names):
            raise ValueError("The feature space contains duplicate gene names; a named mRMR selection is ambiguous.")
        return list(names), True

    def _fit_mrmr(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """
        Select ``mrmr_k`` features on the training fold and keep the selection.

        :param x: design matrix of the training rows
        :param y: responses of the same rows
        :returns: column indices of the selection, or None if ``mrmr_k`` is off or mRMR selects
            nothing (the full feature space is kept then)
        """
        self.mrmr_genes_ = None
        self.mrmr_idx_ = None
        self.mrmr_input_names_ = None
        self.n_features_full_ = int(x.shape[1])
        if not self.mrmr_k:
            return None
        names, named = self._column_names(x.shape[1])
        chosen = mrmr_select(x, y, min(int(self.mrmr_k), x.shape[1]))
        if not chosen:
            # Relevance is not estimable (e.g. constant response in the fold): keep all features
            # instead of losing the whole drug, and say so.
            print(f"SingleDrugMorgoth: mRMR selected no feature for {x.shape[0]} rows x {x.shape[1]} columns.")
            return None
        self.mrmr_genes_ = [names[i] for i in chosen]
        self.mrmr_idx_ = np.array(chosen, dtype=int)
        self.mrmr_input_names_ = names if named else None
        return self.mrmr_idx_

    def _apply_mrmr(self, x: np.ndarray) -> np.ndarray:
        """
        Cut the design matrix down to the features selected at fit time.

        With feature names the columns are matched by name (robust to a different column order),
        otherwise by index, which requires the same number of columns as at fit time.

        :param x: design matrix in the full feature space
        :returns: design matrix with the selected columns
        :raises ValueError: if selected genes are missing or the feature space does not match
        """
        if self.mrmr_idx_ is None:
            return x
        names = self._feature_names
        if self.mrmr_input_names_ is not None and names is not None:
            position = {name: i for i, name in enumerate(names)}
            genes = self.mrmr_genes_ or []
            missing = [g for g in genes if g not in position]
            if missing:
                raise ValueError(
                    f"{len(missing)} genes selected by mRMR are missing in the feature space of the prediction "
                    f"(e.g. {missing[:5]}). The model is not applicable to these data."
                )
            return x[:, np.array([position[g] for g in genes], dtype=int)]
        if self.n_features_full_ is not None and x.shape[1] != self.n_features_full_:
            raise ValueError(
                f"The design matrix has {x.shape[1]} columns, the fit used {self.n_features_full_}; "
                "without feature names there is no safe mapping."
            )
        return x[:, self.mrmr_idx_]

    def selected_features(self) -> dict[str, list[str]] | None:
        """
        Return the mRMR selection of this fit, for the run log.

        :returns: ``{"gene_expression": [...]}`` in selection order, None without a selection
        """
        if not self.mrmr_genes_:
            return None
        return {"gene_expression": list(self.mrmr_genes_)}

    # ------------------------------------------------------------------ Fit

    @staticmethod
    def _frame(x: np.ndarray) -> Any:
        """
        Turn the design matrix into the DataFrame the package expects.

        Row names are running numbers: the package keeps track of sample names, and cell lines
        may occur twice in a dataset.

        :param x: design matrix
        :returns: DataFrame with columns ``f0 ...`` and rows ``r0 ...``
        """
        import pandas as pd

        return pd.DataFrame(
            np.asarray(x, dtype=float),
            columns=[f"f{j}" for j in range(x.shape[1])],
            index=[f"r{i}" for i in range(x.shape[0])],
        )

    def _fit_estimator(self, x: np.ndarray, output: DrugResponseDataset) -> None:
        """
        Fit the MORGOTH forest: threshold, mRMR selection, classes, forest.

        :param x: design matrix of the training rows
        :param output: training dataset (already transformed)
        """
        morgoth_module = _import_morgoth()
        y = np.asarray(output.response, dtype=float)
        self.threshold_ = self._derive_threshold(y)
        sensitive = self._sensitive_mask(y)

        selected = self._fit_mrmr(x, y)
        if selected is not None:
            x = x[:, selected]

        weights, tree_weights = self.sample_weights, self.tree_weights
        self.fallback_ = None
        if min(int(sensitive.sum()), int((~sensitive).sum())) < 2:
            weights, tree_weights = "none", False
            self.fallback_ = "fewer than two cases in one class: without class weights and tree_weights"
            print(f"SingleDrugMorgoth: {self.fallback_}.")

        frame = self._frame(x)
        y_train = np.column_stack([y, sensitive.astype(float)])
        with tempfile.TemporaryDirectory(prefix="morgoth_") as tmp:
            path = partial(os.path.join, tmp)
            forest = morgoth_module.MORGOTH(
                X_train=frame,
                y_train=y_train,
                sample_names_train=frame.index,
                criterion_class=self.criterion_class,
                criterion_reg=self.criterion_reg,
                min_number_of_samples_per_leaf=self.min_samples_leaf,
                number_of_trees_in_forest=self.n_trees,
                number_of_features_per_split=self.max_features,
                class_names=[0, 1],
                output_format="multioutput",
                threshold=[float(self.threshold_)],
                time_file=path("time.txt"),
                sample_weights_included=weights,
                random_state=self.random_state_morgoth,
                max_depth=self.max_depth_morgoth,
                impact_classification=self.impact_classification,
                sample_info_file=path("sample_info.txt"),
                analysis_name="drevalpy",
                leaf_assignment_file_train=path("leaf_assignment.txt"),
                feature_imp_output_file=path("feature_importance.txt"),
                tree_weights=tree_weights,
                distance_measure="",
                draw_graph=False,
            )
            with _pool_with(morgoth_module, self.n_processes), _fast_search(self.fast_split):
                forest.fit()
        self.model = forest

    # ------------------------------------------------------------------ Prediction

    def _predict_estimator(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the response value with the MORGOTH forest and remember the class.

        :param x: design matrix of the prediction rows
        :returns: predictions of the response value
        """
        morgoth_module = _import_morgoth()
        x = self._apply_mrmr(x)
        # With tree_weights the package appends a column to the DataFrame it is given; hence a copy.
        with _pool_with(morgoth_module, self.n_processes):
            result = np.asarray(self.model.predict(X_test=self._frame(x)), dtype=float)
        self.class_predictions_ = result[:, 1]
        return result[:, 0]

    # ------------------------------------------------------------------ Saving

    def save(self, directory: str) -> None:
        """
        Save the forest, scaler, hyperparameters, threshold and mRMR selection.

        :param directory: target directory
        """
        super().save(directory)
        joblib.dump(
            {
                "threshold_": self.threshold_,
                "fallback_": self.fallback_,
                "mrmr_genes_": self.mrmr_genes_,
                "mrmr_idx_": self.mrmr_idx_,
                "mrmr_input_names_": self.mrmr_input_names_,
                "n_features_full_": self.n_features_full_,
            },
            os.path.join(directory, STATE_FILE),
        )

    @classmethod
    def load(cls, directory: str) -> "SingleDrugMorgoth":
        """
        Load the forest, scaler and the fitted extra state.

        :param directory: source directory
        :returns: loaded model
        """
        instance: SingleDrugMorgoth = super().load(directory)  # type: ignore[assignment]
        state_path = os.path.join(directory, STATE_FILE)
        if not os.path.exists(state_path):
            state_path = os.path.join(directory, "sauron_state.pkl")  # models saved by earlier development versions
        if os.path.exists(state_path):
            state = joblib.load(state_path)
            instance.threshold_ = state.get("threshold_")
            instance.fallback_ = state.get("fallback_")
            instance.mrmr_genes_ = state.get("mrmr_genes_")
            instance.mrmr_idx_ = state.get("mrmr_idx_")
            instance.mrmr_input_names_ = state.get("mrmr_input_names_")
            instance.n_features_full_ = state.get("n_features_full_")
        return instance
