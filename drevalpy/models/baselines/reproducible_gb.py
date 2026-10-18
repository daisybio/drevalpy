"""Deterministic variant of the GradientBoosting baseline.

``GradientBoosting`` wraps ``HistGradientBoostingRegressor`` without a ``random_state``. With the
default ``early_stopping="auto"``, scikit-learn reserves a *random* 10 % of the training rows for
early stopping whenever ``n_samples > 10_000``. Two consequences at full CTRPv2 scale:

* the number of boosting iterations (and therefore runtime) varies between runs and between folds;
* two models that should differ only in their target cannot be compared cleanly, because they may
  stop at different iterations by chance.

This class keeps the upstream feature handling, hyperparameter grid and prediction path, and only
fixes the stochastic parts: ``random_state=42`` and an explicit early-stopping configuration. It is
intended for reproducible benchmark comparisons, not as a new modelling idea.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from .sklearn_models import GradientBoosting


class GradientBoostingReproducible(GradientBoosting):
    """``GradientBoosting`` with a fixed random state and deterministic early stopping."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: GradientBoostingReproducible
        """
        return "GradientBoostingReproducible"

    def build_model(self, hyperparameters: dict):
        """
        Build the histogram gradient-boosting regressor deterministically.

        :param hyperparameters: as for :class:`GradientBoosting` (``max_iter``, ``learning_rate``,
            ``max_depth``, plus the feature-view keys)
        """
        super().build_model(hyperparameters)
        self.model = HistGradientBoostingRegressor(
            max_iter=self.hyperparameters.get("max_iter", 100),
            learning_rate=self.hyperparameters.get("learning_rate", 0.1),
            max_depth=self.hyperparameters.get("max_depth", 3),
            early_stopping=self.hyperparameters.get("early_stopping", True),
            validation_fraction=self.hyperparameters.get("validation_fraction", 0.1),
            n_iter_no_change=self.hyperparameters.get("n_iter_no_change", 10),
            random_state=self.hyperparameters.get("random_state", 42),
        )
