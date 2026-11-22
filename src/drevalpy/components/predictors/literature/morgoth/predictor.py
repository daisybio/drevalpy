"""SingleDrugMorgoth: a MORGOTH forest per drug on the mRMR-selected sample features.

``_forest`` derives from scikit-learn bases and imports the ``morgoth`` package, so it is
imported inside the methods that fit: ``drevalpy.registry`` imports this module on
``import drevalpy``. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor, _sample_matrix
from drevalpy.log import get_logger
from drevalpy.registry._component import LiteratureReference
from drevalpy.registry.predictor import register
from drevalpy.types.data.predictor_input import PredictorInput

logger = get_logger(__name__)

#: Response layer with the CurveCurator verdict of each pair: -1 down, 0 not, 1 up, NaN none.
REGULATION = "regulation"

#: Hyperparameters passed on to ``MorgothRegressor`` unchanged.
_FOREST_PARAMETERS = (
    "n_estimators",
    "min_samples_leaf",
    "max_features",
    "max_depth",
    "criterion_class",
    "criterion_reg",
    "impact_classification",
    "sample_weights",
    "tree_weights",
    "threshold_quantile",
    "random_state",
    "fast_split",
)

_REFERENCE = LiteratureReference(
    repo_url="https://github.com/volkamerlab/MORGOTH",
    citation_doi="10.1039/D5DD00284B",
    citation_text=(
        "Rolli et al., Increasing trustworthiness of machine learning-based drug sensitivity prediction with a "
        "multivariate random forest approach, Digital Discovery 2026."
    ),
    deviations=(
        "Port of the DrEvalPy 1.6.1 SingleDrugMorgoth wrapper around the morgoth package, with its threshold rule, "
        "pool handling and vectorized split search. The expression scaler is fitted on all training samples rather "
        "than per drug, and the class comes from the regulation layer coded -1/0/1 instead of strings."
    ),
)


@register(
    "singleDrugMorgoth",
    description="MORGOTH multivariate random forest fitted per drug on mRMR-selected sample features.",
    reference=_REFERENCE,
    license="MIT",
)
class SingleDrugMorgothPredictor(SingleDrugSklearnPredictor):
    """MORGOTH, fitted for each drug on the ``mrmr_k`` sample features most relevant for its response.

    Each tree learns the response and a class at once. The class is 1 (sensitive) for pairs whose
    ``regulation`` is -1 (down) and 0 for 0 (not) and 1 (up); pairs without a verdict (NaN) are not
    trained on, and a drug left without pairs predicts NaN. Without a ``regulation`` layer the class is
    1 for responses below the ``threshold_quantile`` quantile of the drug, as in DrEvalPy 1.6.1; for
    pEC50, where high is sensitive, that class marks the resistant pairs.
    """

    response_layers: ClassVar[tuple[str, ...]] = (REGULATION,)

    def _make_estimator(self) -> Any:
        from sklearn.pipeline import Pipeline

        from drevalpy.components.predictors._mrmr import MRMRSelector
        from drevalpy.components.predictors.literature.morgoth._forest import MorgothRegressor

        forest = MorgothRegressor(**{name: self._hyperparameters[name] for name in _FOREST_PARAMETERS})
        return Pipeline([("mrmr", MRMRSelector(k=int(self._hyperparameters["mrmr_k"]))), ("morgoth", forest)])

    def _fit(self, data: PredictorInput) -> None:
        """Warn once when the dataset has no ``regulation`` layer, then fit one forest per drug.

        :param data: Training batch.
        """
        if REGULATION not in data.response_layers:
            logger.warning(
                "SingleDrugMorgoth: the dataset has no %r response layer; the class of each pair is whether its "
                "response is below the threshold_quantile quantile of its drug.",
                REGULATION,
            )
        super()._fit(data)

    def _fit_drug(self, drug_id: str, data: PredictorInput) -> Any | None:
        """Fit the pipeline of one drug on its pairs with a regulation verdict.

        :param drug_id: Drug whose pairs *data* holds; the pipeline does not depend on it.
        :param data: The drug's training pairs.
        :returns: The fitted pipeline, or ``None`` when no pair has a verdict.
        """
        keep, y_class = class_labels(data.response_layers.get(REGULATION), len(data))
        if not keep.any():
            return None
        estimator = self._make_estimator()
        estimator.fit(_sample_matrix(data)[keep], data.response[keep], morgoth__y_class=y_class)
        return estimator

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Search-space specs with a declared default for each parameter; untuned settings are fixed.
        """
        return {
            "impact_classification": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5},
            "max_depth": {"type": "int", "low": 5, "high": 20, "default": 10},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 20, "default": 10},
            "mrmr_k": {"type": "int", "low": 20, "high": 100, "default": 50},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2"], "default": "sqrt"},
            "n_estimators": {"type": "fixed", "default": 100},
            "criterion_class": {"type": "fixed", "default": "gini"},
            "criterion_reg": {"type": "fixed", "default": "mse"},
            "sample_weights": {"type": "fixed", "default": "simple"},
            "tree_weights": {"type": "fixed", "default": True},
            "threshold_quantile": {"type": "fixed", "default": 0.5},
            "fast_split": {"type": "fixed", "default": True},
            "random_state": {"type": "fixed", "default": 42},
        }


def class_labels(regulation: np.ndarray | None, n_pairs: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Turn the regulation verdicts of a drug's pairs into the pairs to train on and their classes.

    :param regulation: One verdict per pair (-1 down, 0 not, 1 up, NaN none), or ``None`` without the layer.
    :param n_pairs: Number of pairs, used when *regulation* is ``None``.
    :returns: A mask of the pairs with a verdict, and their classes (1 for down, else 0); every pair
        and ``None`` without the layer, so the forest falls back to its quantile threshold.
    """
    if regulation is None:
        return np.ones(n_pairs, dtype=bool), None
    regulation = np.asarray(regulation, dtype=np.float64)
    keep = ~np.isnan(regulation)
    return keep, (regulation[keep] == -1).astype(int)
