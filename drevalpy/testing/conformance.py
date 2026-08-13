"""Reusable conformance checks for third-party featurizers and predictors.

Registration validates a component's declarations, not its behaviour: a class
missing ``_fit`` registers happily and only fails when instantiated, and a
``get_state``/``set_state`` pair that drops a fitted attribute is invisible until
a checkpoint is reloaded. The checks below close that gap by actually running the
component, and are what a plugin's test suite parametrizes over.

Each ``check_*`` function raises :class:`ConformanceError` with a message naming
the component and the broken expectation, and otherwise returns ``None``.

Every check in a family takes the same arguments - ``(cls, fixture, **kwargs)``,
where the fixture is a dataset for featurizers and a batch for predictors and may
be omitted - so a suite can parametrize over :data:`FEATURIZER_CHECKS` or
:data:`PREDICTOR_CHECKS` directly::

    @pytest.mark.parametrize("check", FEATURIZER_CHECKS)
    def test_my_featurizer_conforms(check):
        check(MyFeaturizer)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import (
    CellLineFeatureSource,
    DrugFeatureSource,
    FeatureSource,
)

from .batch import build_synthetic_batch
from .synthetic import build_synthetic_dataset


class ConformanceError(AssertionError):
    """Raised when a component does not honour the featurizer/predictor contract.

    Subclasses ``AssertionError`` so a failure reads as a test failure when
    raised from inside a test.
    """


def _require(condition: object, message: str) -> None:
    """Raise :class:`ConformanceError` with *message* unless *condition* holds."""
    if not condition:
        raise ConformanceError(message)


def feature_source_for(cls: type[Featurizer], dataset: Dataset) -> FeatureSource:
    """Return the feature source matching the featurizer's registered side.

    Args:
        cls: Featurizer class. Its ``side`` is set by the registry to
            ``"cell_line"`` or ``"drug"``.
        dataset: Dataset to wrap.

    Returns:
        A source for the featurizer's own entity side.
    """
    if getattr(cls, "side", "") == "drug":
        return DrugFeatureSource(dataset, np.asarray(dataset.drug_ids))
    return CellLineFeatureSource(dataset, np.asarray(dataset.cell_line_ids))


def _entity_ids(cls: type[Featurizer], dataset: Dataset) -> np.ndarray:
    return np.asarray(dataset.drug_ids if getattr(cls, "side", "") == "drug" else dataset.cell_line_ids)


def check_featurizer_instantiates(
    cls: type[Featurizer],
    dataset: Dataset | None = None,
    **kwargs: Any,
) -> Featurizer:
    """Check that *cls* can be constructed with its declared defaults.

    This is the check registration cannot make: ``ABCMeta`` only enforces
    abstract methods at instantiation, so a featurizer missing ``_fit`` or
    ``_transform_blocks`` registers cleanly and fails here.

    Args:
        cls: Featurizer class to instantiate.
        dataset: Unused; accepted so every entry of :data:`FEATURIZER_CHECKS`
            takes the same arguments and a suite can parametrize over the tuple.
        kwargs: Constructor keyword arguments. Defaults to none, i.e. the
            featurizer must be constructible with no arguments.

    Returns:
        The constructed instance.

    Raises:
        ConformanceError: If construction fails, or the result is not a
            :class:`~drevalpy.plugin.Featurizer`.
    """
    _ = dataset
    try:
        instance = cls(**kwargs)
    except TypeError as exc:
        msg = f"{cls.__name__} could not be constructed with {kwargs!r}: {exc}"
        raise ConformanceError(msg) from exc
    _require(isinstance(instance, Featurizer), f"{cls.__name__} is not a Featurizer subclass instance")
    return instance


def check_featurizer_fit_transform(
    cls: type[Featurizer],
    dataset: Dataset | None = None,
    **kwargs: Any,
) -> None:
    """Check that *cls* fits, transforms and reports a matching ``output_dim``.

    Args:
        cls: Featurizer class to exercise.
        dataset: Dataset to featurize. Defaults to
            :func:`~drevalpy.testing.build_synthetic_dataset`, which suffices for
            an ``entity_id_only`` featurizer; pass a dataset carrying the
            required views otherwise.
        kwargs: Constructor keyword arguments.

    Raises:
        ConformanceError: If fitting or transforming fails, the blocks are not
            aligned with the entity ids, or ``output_dim`` disagrees with the
            produced width.
    """
    dataset = dataset if dataset is not None else build_synthetic_dataset()
    instance = check_featurizer_instantiates(cls, **kwargs)
    source = feature_source_for(cls, dataset)
    entity_ids = _entity_ids(cls, dataset)

    instance.fit(source, entity_ids=entity_ids)
    blocks = instance.transform_blocks(source, entity_ids)
    _require(blocks, f"{cls.__name__}.transform_blocks returned no blocks")
    for name, block in blocks.items():
        if not block.entity_aligned:
            continue
        _require(
            len(block.values) == len(entity_ids),
            f"{cls.__name__} block {name!r} has {len(block.values)} rows for {len(entity_ids)} entities",
        )

    matrix = instance.transform(source, entity_ids)
    _require(
        matrix.shape[0] == len(entity_ids),
        f"{cls.__name__}.transform returned {matrix.shape[0]} rows for {len(entity_ids)} entities",
    )
    _require(
        instance.output_dim == matrix.shape[1],
        f"{cls.__name__}.output_dim is {instance.output_dim} but transform produced {matrix.shape[1]} columns",
    )


def check_featurizer_state_round_trip(
    cls: type[Featurizer],
    dataset: Dataset | None = None,
    **kwargs: Any,
) -> None:
    """Check that a fitted featurizer survives ``get_state``/``set_state``.

    A checkpoint stores the state mapping, not the object, so a fitted attribute
    left out of ``get_state`` makes a reloaded model silently produce different
    features. This transforms through a fresh instance restored from the state
    and requires identical output.

    Args:
        cls: Featurizer class to exercise.
        dataset: Dataset to featurize; defaults to the synthetic one.
        kwargs: Constructor keyword arguments.

    Raises:
        ConformanceError: If the restored featurizer produces different features.
    """
    dataset = dataset if dataset is not None else build_synthetic_dataset()
    source = feature_source_for(cls, dataset)
    entity_ids = _entity_ids(cls, dataset)

    fitted = check_featurizer_instantiates(cls, **kwargs)
    fitted.fit(source, entity_ids=entity_ids)
    expected = fitted.transform(source, entity_ids)

    restored = check_featurizer_instantiates(cls, **kwargs)
    restored.set_state(fitted.get_state())
    actual = restored.transform(source, entity_ids)

    _require(
        actual.shape == expected.shape,
        f"{cls.__name__} restored from get_state produced shape {actual.shape}, expected {expected.shape}",
    )
    _require(
        np.allclose(actual, expected, equal_nan=True),
        f"{cls.__name__} restored from get_state produced different features; "
        "check that get_state covers every attribute _transform_blocks reads",
    )


def check_predictor_instantiates(
    cls: type[Predictor],
    batch: ModelInputBatch | None = None,
    **kwargs: Any,
) -> Predictor:
    """Check that *cls* can be constructed with its declared defaults.

    Args:
        cls: Predictor class to instantiate.
        batch: Unused; accepted so every entry of :data:`PREDICTOR_CHECKS` takes
            the same arguments and a suite can parametrize over the tuple.
        kwargs: Constructor keyword arguments, typically ``hyperparameters=``.

    Returns:
        The constructed instance.

    Raises:
        ConformanceError: If construction fails, or the result is not a
            :class:`~drevalpy.plugin.Predictor`.
    """
    _ = batch
    try:
        instance = cls(**kwargs)
    except TypeError as exc:
        msg = f"{cls.__name__} could not be constructed with {kwargs!r}: {exc}"
        raise ConformanceError(msg) from exc
    _require(isinstance(instance, Predictor), f"{cls.__name__} is not a Predictor subclass instance")
    return instance


def check_predictor_fit_predict(
    cls: type[Predictor],
    batch: ModelInputBatch | None = None,
    **kwargs: Any,
) -> None:
    """Check that *cls* trains and returns one finite prediction per pair.

    An unfitted :class:`~drevalpy.plugin.MatrixPredictor` returns all-NaN, so the
    finiteness requirement is what distinguishes a predictor that trained from
    one that merely ran.

    Args:
        cls: Predictor class to exercise.
        batch: Featurized batch. Defaults to
            :func:`~drevalpy.testing.build_synthetic_batch` over the synthetic
            dataset.
        kwargs: Constructor keyword arguments.

    Raises:
        ConformanceError: If fitting or predicting fails, or the predictions are
            the wrong length or not finite.
    """
    batch = batch if batch is not None else build_synthetic_batch(build_synthetic_dataset())
    instance = check_predictor_instantiates(cls, **kwargs)

    instance.fit(batch)
    predictions = np.asarray(instance.predict(batch))

    _require(
        predictions.shape == (batch.n_pairs,),
        f"{cls.__name__}.predict returned shape {predictions.shape}, expected ({batch.n_pairs},)",
    )
    _require(
        np.isfinite(predictions).all(),
        f"{cls.__name__}.predict returned non-finite values; an unfitted predictor returns all-NaN, "
        "so check that _fit stores the trained model",
    )


def check_predictor_state_round_trip(
    cls: type[Predictor],
    batch: ModelInputBatch | None = None,
    **kwargs: Any,
) -> None:
    """Check that a trained predictor survives ``get_state``/``set_state``.

    Args:
        cls: Predictor class to exercise.
        batch: Featurized batch; defaults to the synthetic one.
        kwargs: Constructor keyword arguments.

    Raises:
        ConformanceError: If the restored predictor predicts differently.
    """
    batch = batch if batch is not None else build_synthetic_batch(build_synthetic_dataset())

    trained = check_predictor_instantiates(cls, **kwargs)
    trained.fit(batch)
    expected = np.asarray(trained.predict(batch))
    _require(
        np.isfinite(expected).all(),
        f"{cls.__name__} produced non-finite predictions before the round trip, so it cannot be compared",
    )

    restored = check_predictor_instantiates(cls, **kwargs)
    restored.set_state(trained.get_state())
    actual = np.asarray(restored.predict(batch))

    _require(
        np.allclose(actual, expected, equal_nan=False),
        f"{cls.__name__} restored from get_state predicted differently; "
        "check that get_state covers every attribute _predict reads",
    )


#: Every featurizer check, so a plugin's suite can parametrize over the set
#: rather than list them and drift when a check is added.
FEATURIZER_CHECKS: tuple[Callable[..., Any], ...] = (
    check_featurizer_instantiates,
    check_featurizer_fit_transform,
    check_featurizer_state_round_trip,
)

#: Every predictor check; see :data:`FEATURIZER_CHECKS`.
PREDICTOR_CHECKS: tuple[Callable[..., Any], ...] = (
    check_predictor_instantiates,
    check_predictor_fit_predict,
    check_predictor_state_round_trip,
)
