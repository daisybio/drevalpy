"""Shared base for the two third-party gradient-boosting predictors.

``lightgbm_pred.py`` and ``xgboost_pred.py`` wrap different libraries but read the
same tree/shrinkage/subsampling knobs out of ``self._hyperparameters`` with the same coercions,
and tune overlapping slices of one search space. That common part lives here so a
change to a default or a bound is made once.

The module is ``_``-prefixed on purpose: ``registry/_builtins.py`` registers
predictors by scanning the directory, and a public module here would be imported
as a component. It also imports nothing outside ``drevalpy``, so the boosting
libraries stay off the ``import drevalpy`` path (see
``tests/test_import_cost_policy.py``) - only the concrete subclasses reach for
them, inside ``_make_estimator``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy.components.contracts.hyperparameter_space import is_fixed
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor

#: Search-space specs for every knob either predictor takes. A subclass selects
#: the ones it exposes and adjusts individual fields; it never restates a spec.
#: ``learning_rate`` carries no ``log`` flag here because the two predictors
#: disagree on it - LightGBM draws it log-uniformly, XGBoost uniformly. The type of
#: each ``default`` also fixes the coercion applied to an override, so a
#: hyperparameter arriving as a string from a config file still reaches the
#: estimator as a number.
SHARED_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "low": 50, "high": 300, "default": 100},
    "max_depth": {"type": "int", "low": 3, "high": 12, "default": 6},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "default": 0.1},
    "num_leaves": {"type": "int", "low": 15, "high": 255, "default": 63},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0, "default": 0.8},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "default": 0.8},
    "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0, "default": 0.0},
    "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0, "default": 0.0},
    "random_state": {"type": "fixed", "default": 42},
    "n_jobs": {"type": "fixed", "default": -1},
}


class BoostedTreesPredictor(SklearnTabularPredictor):
    """Hyperparameter plumbing shared by the LightGBM and XGBoost regressors."""

    #: Names from :data:`SHARED_SPACE` this predictor passes to its estimator, in declaration order.
    estimator_hyperparameters: ClassVar[tuple[str, ...]] = ()

    #: Per-name adjustments to the shared spec, merged over it.
    boosting_space_overrides: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the space assembled from the shared specs.

        :returns: One spec per name in :attr:`estimator_hyperparameters`.
        """
        return {
            name: {**SHARED_SPACE[name], **cls.boosting_space_overrides.get(name, {})}
            for name in cls.estimator_hyperparameters
        }

    def _estimator_params(self) -> dict[str, Any]:
        """Resolve every constructor argument this library takes from hyperparameters.

        :returns: Hyperparameter values, tuned ones coerced to the type of their default.
        """
        space = self.get_hyperparameter_space()
        return {
            name: self._hyperparameters[name] if is_fixed(spec) else type(spec["default"])(self._hyperparameters[name])
            for name, spec in space.items()
        }
