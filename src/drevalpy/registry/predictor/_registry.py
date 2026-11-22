"""The predictor registry and its ``register`` decorator."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from drevalpy.components.hyperparameter_space import validate_component_hyperparameter_space
from drevalpy.components.predictors.inputs import predictor_inputs
from drevalpy.registry._component import LiteratureReference, component_decorator
from drevalpy.registry._registry import Registry
from drevalpy.types.enums import Accelerator, ModelScope

#: Every registered predictor class, in registration order.
predictor_registry: Registry[type[Any]] = Registry("predictor")


def _validate(name: str, cls: type[Any], accelerator: Accelerator) -> None:
    """Check the input declarations, the hyperparameter space and that a single-drug predictor is CPU only.

    ``predictor_inputs`` raises ``TypeError`` for a side without a declaration.

    :param name: Registry name under which *cls* is being registered.
    :param cls: Predictor class.
    :param accelerator: The accelerator level the class is registered with.
    :raises ValueError: If *cls* is a single-drug predictor and *accelerator* is not ``NONE``: its
        per-drug fits run on the CPU (see :mod:`drevalpy.components.predictors.abstract.single_drug`).
    """
    for side in ("sample", "drug"):
        predictor_inputs(cls, side)
    validate_component_hyperparameter_space(name, cls)
    if cls.scope is ModelScope.SINGLE_DRUG and accelerator is not Accelerator.NONE:
        msg = (
            f"predictor '{name}' fits one model per drug, so it runs on the CPU only: "
            f"register it without accelerator={accelerator.value!r}"
        )
        raise ValueError(msg)


def register(
    name: str,
    *,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
    accelerator: Accelerator | str = Accelerator.NONE,
    license: str | None = None,
    artifact_license: str | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Return a class decorator that registers a predictor under *name*.

    What the predictor reads from each side is its ``sample_inputs`` / ``drug_inputs``
    declaration; registration checks that both are declared.

    :param name: Registry name used in ``ModelConfig`` and recipes.
    :param description: Short human-readable summary for catalog listings.
    :param tags: Optional discovery tags.
    :param reference: Optional literature citation metadata.
    :param accelerator: ``"optional"`` when the predictor uses a torch accelerator if
        there is one, ``"required"`` when it cannot run without one. A single-drug predictor
        (``scope = ModelScope.SINGLE_DRUG``) runs on the CPU and keeps the default ``"none"``.
    :param license: SPDX license of the code the predictor was ported from; required with a
        *reference*. Without one, the predictor has the license of the package that ships it.
    :param artifact_license: SPDX license of the third-party files the predictor downloads.
    :returns: Class decorator that registers and returns the decorated predictor.
    """
    return component_decorator(
        predictor_registry,
        name,
        description=description,
        tags=tags,
        reference=reference,
        accelerator=accelerator,
        license=license,
        artifact_license=artifact_license,
        validate=lambda registered_name, cls: _validate(registered_name, cls, Accelerator(accelerator)),
    )


def get(name: str) -> type[Any]:
    """Return the predictor class registered under *name*.

    An unknown *name* raises :class:`~drevalpy.errors.UnknownNameError`.

    :param name: Registry name of the predictor.
    :returns: Predictor class registered under *name*.
    """
    return predictor_registry.get(name)
