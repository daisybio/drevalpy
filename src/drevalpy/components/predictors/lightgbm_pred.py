"""LightGBM tabular predictor.

``lightgbm`` is imported inside ``_make_estimator``: ``drevalpy.registry`` imports
this module to register the ``lightgbm`` predictor on ``import drevalpy``, and
``lightgbm.compat`` pulls in ``sklearn`` (and through it ``scipy.stats``), which
costs ~0.39s. See ``tests/test_import_cost_policy.py``.

Everything this shares with ``xgboost_pred.py`` lives in ``_boosted_trees.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors._boosted_trees import BoostedTreesPredictor
from drevalpy.registry.predictor import register


@register(
    "lightgbm",
    description="LightGBM regressor on concatenated dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class LightGBMPredictor(BoostedTreesPredictor):
    """LightGBM regressor for dense tabular pair features."""

    boosting_default_overrides: ClassVar[dict[str, Any]] = {
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    boosting_extra_defaults: ClassVar[dict[str, Any]] = {
        "num_leaves": 63,
        "reg_lambda": 0.0,
        "n_jobs": -1,
    }

    boosting_space_overrides: ClassVar[dict[str, dict[str, Any]]] = {
        "learning_rate": {"log": True},
    }

    tuned_hyperparameters: ClassVar[tuple[str, ...]] = (
        "n_estimators",
        "learning_rate",
        "max_depth",
        "num_leaves",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    )

    def _make_estimator(self):
        """Return an unfitted LightGBM regressor.

        :returns: Unfitted ``LGBMRegressor`` configured from hyperparameters.
        """
        import lightgbm as lgb

        return lgb.LGBMRegressor(**self._estimator_params(), verbosity=-1)
