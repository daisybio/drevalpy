"""sklearn interface to the MORGOTH forest of the ``morgoth`` package.

MORGOTH is a random forest whose trees learn the response value and a class (sensitive or resistant) at once;
``impact_classification`` (lambda in the paper) weights the classification in the split criterion. drevalpy uses
the regression output as the prediction.

The package imports ``fireducks.pandas`` on Linux, but uses it only like pandas (for annotations and plain frames).
drevalpy's ``pyproject.toml`` excludes fireducks, because it pins pyarrow 21 and pandas<2.4, so
:func:`import_morgoth` aliases ``fireducks.pandas`` to pandas when fireducks is absent (see
``_fast_split.alias_fireducks``). Every import of the package goes through :func:`import_morgoth`.

:class:`MorgothRegressor` derives from scikit-learn bases, so this module imports ``sklearn`` at module scope.
Nothing on ``import drevalpy`` imports it: the predictor imports it inside the functions that fit. See
``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import tempfile
from collections.abc import Iterator
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from upath import UPath

from ._fast_split import alias_fireducks, fast_split


def import_morgoth() -> Any:
    """Import the ``morgoth`` package, with ``fireducks.pandas`` aliased to pandas if fireducks is absent.

    :returns: The class ``morgoth.morgoth.MORGOTH``.
    :raises ImportError: If the optional ``morgoth`` extra is not installed.
    """
    alias_fireducks()
    try:
        from morgoth.morgoth import MORGOTH
    except ImportError as error:
        raise ImportError(
            "SingleDrugMorgoth needs the optional 'morgoth' extra: pip install 'drevalpy[morgoth]' "
            "(with uv: uv sync --extra morgoth)"
        ) from error
    return MORGOTH


@contextlib.contextmanager
def _pool(n_jobs: int) -> Iterator[None]:
    """Set the number of processes of the ``Pool()`` hard-wired in the package and restore it afterwards.

    ``joblib.effective_n_jobs`` resolves -1 to all cores, but to 1 inside the workers of a per-drug parallel
    run, where the drugs already run in parallel. With one process, the trees are grown in this process
    instead of in a child.

    :param n_jobs: Number of processes, -1 for all cores.
    :yields: Nothing; the package's ``Pool`` is swapped inside the block.
    """
    import_morgoth()
    import morgoth.morgoth as morgoth_module

    processes = joblib.effective_n_jobs(n_jobs)
    original = morgoth_module.Pool
    morgoth_module.Pool = partial(ThreadPool, 1) if processes == 1 else partial(multiprocessing.Pool, processes)
    try:
        yield
    finally:
        morgoth_module.Pool = original


class MorgothRegressor(RegressorMixin, BaseEstimator):
    """MORGOTH forest of the ``morgoth`` package, predicting the response value."""

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 10,
        max_features: str | float = "sqrt",
        max_depth: int = 10,
        criterion_class: str = "gini",
        criterion_reg: str = "mse",
        impact_classification: float = 0.5,
        sample_weights: str = "simple",
        tree_weights: bool = True,
        threshold_quantile: float = 0.5,
        random_state: int = 42,
        n_jobs: int = -1,
        fast_split: bool = True,
    ) -> None:
        """Store the parameters; they are passed on to ``morgoth.MORGOTH`` in :meth:`fit`.

        :param n_estimators: Number of trees.
        :param min_samples_leaf: Minimal number of samples per leaf.
        :param max_features: Features per split: ``"sqrt"``, ``"log2"`` or a fraction in [0, 1].
        :param max_depth: Maximal depth of the trees.
        :param criterion_class: Split criterion of the classification: ``"gini"``, ``"entropy"`` or
            ``"hellinger"``.
        :param criterion_reg: Split criterion of the regression, ``"mse"``.
        :param impact_classification: Share of the classification in the split criterion (lambda in the paper).
        :param sample_weights: Sample weighting of the package: ``"simple"``, ``"linear"`` or ``"none"``.
        :param tree_weights: Whether to use the class-aware tree weights of SAURON-RF.
        :param threshold_quantile: Quantile of the training responses that separates the two classes; only used
            without class labels.
        :param random_state: Seed of the forest.
        :param n_jobs: Number of processes, -1 for all cores; with ``fast_split`` the trees are grown in this
            process.
        :param fast_split: Replace the package's split search by the equivalent vectorized one; False keeps the
            original.
        """
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth
        self.criterion_class = criterion_class
        self.criterion_reg = criterion_reg
        self.impact_classification = impact_classification
        self.sample_weights = sample_weights
        self.tree_weights = tree_weights
        self.threshold_quantile = threshold_quantile
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fast_split = fast_split

    def fit(self, X: Any, y: Any, y_class: Any = None) -> MorgothRegressor:  # noqa: N803
        """Fit the forest.

        With ``y_class``, the threshold for the sample weights is the quantile of the responses at the fraction
        of sensitive samples, so the weights balance the classes; that fraction is flipped when the sensitive
        samples have the higher responses (e.g. pEC50). Without it, the threshold is the ``threshold_quantile``
        and the class is 1 below it (sensitive for e.g. ln IC50). Calls :func:`import_morgoth`, which raises
        ``ImportError`` without the ``morgoth`` extra.

        :param X: Feature matrix, one row per sample.
        :param y: Response vector.
        :param y_class: Class labels (1 = sensitive, 0 = resistant), or None to derive them from the threshold.
        :returns: The fitted regressor.
        """
        morgoth_class = import_morgoth()
        y = np.asarray(y, dtype=float)
        if y_class is not None:
            y_class = np.asarray(y_class)
            q = float(np.mean(y_class))
            # sensitive samples can be the low (e.g. ln IC50) or the high (e.g. pEC50) responses
            if 0 < q < 1 and y[y_class == 1].mean() > y[y_class == 0].mean():
                q = 1 - q
            self.threshold_ = float(np.quantile(y, q))
        else:
            self.threshold_ = float(np.quantile(y, self.threshold_quantile))
            y_class = (y < self.threshold_).astype(int)
        # the patched split search only exists in this process: spawned pool workers import the package unpatched
        fit_jobs = 1 if self.fast_split else self.n_jobs
        # the package writes these files during the fit; drevalpy does not need them
        with tempfile.TemporaryDirectory() as tmp, _pool(fit_jobs), fast_split(self.fast_split):
            directory = UPath(tmp)
            self.forest_ = morgoth_class(
                X_train=pd.DataFrame(X),
                y_train=np.column_stack([y, y_class]),
                criterion_class=self.criterion_class,
                criterion_reg=self.criterion_reg,
                sample_names_train=np.arange(len(y)).astype(str),
                min_number_of_samples_per_leaf=self.min_samples_leaf,
                number_of_trees_in_forest=self.n_estimators,
                number_of_features_per_split=self.max_features,
                class_names=[0, 1],
                output_format="multioutput",
                threshold=[self.threshold_],
                time_file=str(directory / "time.txt"),
                sample_weights_included=self.sample_weights,
                random_state=self.random_state,
                max_depth=self.max_depth,
                impact_classification=self.impact_classification,
                sample_info_file=str(directory / "sample_info.txt"),
                analysis_name="drevalpy",
                leaf_assignment_file_train=str(directory / "leaf_assignment.txt"),
                feature_imp_output_file=str(directory / "feature_importance.txt"),
                tree_weights=self.tree_weights,
            )
            self.forest_.fit()
        return self

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        """Predict the response value (the regression output of the forest).

        :param X: Feature matrix, one row per sample.
        :returns: Predicted responses as a float array.
        """
        with _pool(self.n_jobs):
            prediction = self.forest_.predict(X_test=pd.DataFrame(X))
        return np.asarray(prediction)[:, 0].astype(float)
