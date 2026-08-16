"""Shared base for the two third-party gradient-boosting predictors.

``lightgbm_pred.py`` and ``xgboost_pred.py`` wrap different libraries but read the
same tree/shrinkage/subsampling knobs out of ``self._h`` with the same coercions,
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

from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor

#: Constructor arguments both libraries accept, with the defaults they agree on.
#: The type of each value also fixes the coercion applied to an override, so a
#: hyperparameter arriving as a string from a config file still reaches the
#: estimator as a number.
SHARED_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "random_state": 42,
}

#: Search-space specs for every knob either predictor tunes. A subclass selects
#: the ones it exposes and adjusts individual fields; it never restates a spec.
#: ``learning_rate`` carries no ``log`` flag here because the two predictors
#: disagree on it - LightGBM samples it log-uniformly, XGBoost uniformly.
SHARED_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "low": 50, "high": 300, "default": 100},
    "max_depth": {"type": "int", "low": 3, "high": 12, "default": 6},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "default": 0.1},
    "num_leaves": {"type": "int", "low": 15, "high": 255, "default": 63},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0, "default": 0.8},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "default": 0.8},
    "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0, "default": 0.0},
    "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0, "default": 0.0},
}


class BoostedTreesPredictor(SklearnTabularPredictor):
    """Hyperparameter plumbing shared by the LightGBM and XGBoost regressors."""

    #: Entries of :data:`SHARED_DEFAULTS` this library defaults differently.
    boosting_default_overrides: ClassVar[dict[str, Any]] = {}

    #: Constructor arguments only this library accepts, with their defaults.
    #: Resolved from ``self._h`` exactly like the shared ones.
    boosting_extra_defaults: ClassVar[dict[str, Any]] = {}

    #: Names from :data:`SHARED_SPACE` this predictor tunes, in declaration order.
    tuned_hyperparameters: ClassVar[tuple[str, ...]] = ()

    #: Per-name adjustments to the shared spec, merged over it.
    boosting_space_overrides: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable space assembled from the shared specs.

        :returns: One spec per name in :attr:`tuned_hyperparameters`.
        """
        return {
            name: {**SHARED_SPACE[name], **cls.boosting_space_overrides.get(name, {})}
            for name in cls.tuned_hyperparameters
        }

    def _estimator_params(self) -> dict[str, Any]:
        """Resolve every constructor argument this library takes from hyperparameters.

        :returns: Hyperparameter values coerced to the type of their default.
        """
        defaults = {
            **SHARED_DEFAULTS,
            **self.boosting_default_overrides,
            **self.boosting_extra_defaults,
        }
        return {name: type(default)(self._h.get(name, default)) for name, default in defaults.items()}
