"""The search space of a model and drawing hyperparameters from it.

:func:`space` merges the spaces of a model's components under their qualified keys
(:mod:`drevalpy.models.hyperparameters`): sample featurizer leaves, drug featurizer leaves,
then the predictor. A component's space is the one its config declares, else its class's
``get_hyperparameter_space()``. :func:`draw` turns a space into values for one Optuna trial.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from drevalpy.models.hyperparameters import PREDICTOR_SLOT, key
from drevalpy.registry.predictor import get as get_predictor

if TYPE_CHECKING:
    from drevalpy.models.config import ModelConfig


def _effective_space(declared: Mapping[str, Any] | None, cls: type[Any]) -> dict[str, Any]:
    if declared is not None:
        return {name: dict(spec) for name, spec in declared.items()}
    return dict(cls.get_hyperparameter_space())


def space(config: ModelConfig) -> dict[str, Any]:
    """Merge the hyperparameter spaces of all components of a model config.

    :param config: Model configuration.
    :returns: Every component's space entries under their qualified keys.
    """
    merged: dict[str, Any] = {}
    for side, root in config.featurizers().items():
        for leaf in root.leaves():
            leaf_space = _effective_space(leaf.hyperparameter_space, leaf.featurizer_class)
            merged.update({key(f"{side}_featurizer", leaf.selector, name): spec for name, spec in leaf_space.items()})
    predictor = config.predictor
    predictor_space = _effective_space(predictor.hyperparameter_space, get_predictor(predictor.name))
    merged.update({key(PREDICTOR_SLOT, predictor.name, name): spec for name, spec in predictor_space.items()})
    return merged


def _suggest_categorical(trial: Any, name: str, choices: list[Any]) -> Any:
    """Draw one of *choices*, which may be lists such as layer widths.

    Optuna only accepts ``None``, booleans, numbers and strings as categorical choices, so a
    choice list holding anything else is drawn by index.

    :param trial: An ``optuna.Trial`` instance.
    :param name: Parameter name.
    :param choices: The values to choose from.
    :returns: The chosen value.
    """
    if any(isinstance(choice, (list, tuple, dict)) for choice in choices):
        return choices[trial.suggest_categorical(name, list(range(len(choices))))]
    return trial.suggest_categorical(name, choices)


def draw(trial: Any, space: Mapping[str, Any]) -> dict[str, Any]:
    """Draw one value per entry of *space* from an Optuna trial.

    A ``fixed`` entry keeps its ``default`` without asking the trial. A ``pow2`` entry is sampled as
    an integer exponent and returned as the power of two, so replaying a trial's ``params`` through
    this function (``optuna.trial.FixedTrial``) gives the values it trained with.

    :param trial: An ``optuna.Trial`` instance.
    :param space: Search space such as
        ``{"alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True, "default": 1.0}}``.
    :returns: Drawn values under the keys of *space*.
    :raises ValueError: If an entry has an unknown type.
    """
    result: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec.get("type", "categorical")
        if kind == "fixed":
            result[name] = spec["default"]
        elif kind == "int":
            result[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=spec.get("log", False))
        elif kind == "float":
            result[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False)
            )
        elif kind == "categorical":
            result[name] = _suggest_categorical(trial, name, list(spec.get("choices", [])))
        elif kind == "pow2":
            result[name] = 2 ** trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        else:
            raise ValueError(f"Unknown hyperparameter type {kind!r} for parameter {name!r}")
    return result
