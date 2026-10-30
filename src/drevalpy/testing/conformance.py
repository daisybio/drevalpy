"""Reusable conformance checks for third-party featurizers and predictors.

Registration validates a component's declarations, not its behaviour: a class
missing ``transform`` registers happily and only fails when instantiated, and a
fitted attribute that does not pickle is invisible until a saved model is reloaded.
The checks below close that gap by actually running the component, and are what a
plugin's test suite parametrizes over.

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

import io
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import FeatureSource, Side

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


def _side(cls: type[Any]) -> Side:
    """Return the side the checks run *cls* on: ``"sample"`` when it supports it, else ``"drug"``.

    :param cls: Featurizer class.
    :returns: A side among the class's ``sides``.
    """
    return "sample" if "sample" in cls.sides else "drug"


def feature_source_for(cls: type[Featurizer], dataset: Dataset) -> FeatureSource:
    """Return a feature source over every entity of a side the featurizer supports.

    :param cls: Featurizer class; the sample side is used when it is among its ``sides``.
    :param dataset: Dataset to wrap.
    :returns: A source for that side.
    """
    side = _side(cls)
    return FeatureSource(dataset, side, _entity_ids(cls, dataset))


def _entity_ids(cls: type[Featurizer], dataset: Dataset) -> np.ndarray:
    return np.asarray(dataset.drug_ids if _side(cls) == "drug" else dataset.sample_ids)


def _features(featurizer: Featurizer, source: FeatureSource, ids: np.ndarray, *, fit: bool) -> FeaturizerResult:
    from drevalpy.models._feature_preparation import FitIds, prepare_features

    return prepare_features(featurizer, source, ids, fit=FitIds(ids) if fit else None)


def check_featurizer_instantiates(
    cls: type[Featurizer],
    dataset: Dataset | None = None,
    **kwargs: Any,
) -> Featurizer:
    r"""Check that *cls* can be constructed with its declared defaults.

    This is the check registration cannot make: ``ABCMeta`` only enforces
    abstract methods at instantiation, so a featurizer missing ``fit`` or
    ``transform`` registers cleanly and fails here.

    :param cls: Featurizer class to instantiate.
    :param dataset: Unused; accepted so every entry of :data:`FEATURIZER_CHECKS`
        takes the same arguments and a suite can parametrize over the tuple.
    :param \*\*kwargs: Constructor keyword arguments. Defaults to none, i.e. the
        featurizer must be constructible with no arguments.
    :returns: The constructed instance.
    :raises ConformanceError: If construction fails, or the result is not a
        :class:`~drevalpy.components.featurizers.featurizer.Featurizer`.
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
    r"""Check that *cls* fits, transforms and emits the blocks ``output_blocks`` declares.

    Raises ``ConformanceError`` if fitting or transforming fails, no block is
    produced, an entity-aligned block is not aligned with the entity IDs, or the
    produced entity-aligned block names differ from the declared ones (global
    metadata blocks such as ``identity_categories`` are not declared).

    :param cls: Featurizer class to exercise.
    :param dataset: Dataset to featurize. Defaults to
        :func:`~drevalpy.testing.build_synthetic_dataset`, which suffices for a
        featurizer that reads no view; pass a dataset carrying the required views otherwise.
    :param \*\*kwargs: Constructor keyword arguments.
    """
    dataset = dataset if dataset is not None else build_synthetic_dataset()
    instance = check_featurizer_instantiates(cls, **kwargs)
    source = feature_source_for(cls, dataset)
    entity_ids = _entity_ids(cls, dataset)

    result = _features(instance, source, entity_ids, fit=True)
    blocks = result.blocks
    _require(blocks, f"{cls.__name__}.transform returned no blocks")
    for name, block in blocks.items():
        if not block.entity_aligned:
            continue
        _require(
            len(block.values) == len(entity_ids),
            f"{cls.__name__} block {name!r} has {len(block.values)} rows for {len(entity_ids)} entities",
        )
    config = dict(kwargs)
    for key, spec in cls.get_hyperparameter_space().items():
        if isinstance(spec, Mapping) and "default" in spec:
            config.setdefault(key, spec["default"])
    declared = {spec.name for spec in cls.output_blocks(config)}
    aligned = {name for name, block in blocks.items() if block.entity_aligned}
    _require(
        not declared or declared == aligned,
        f"{cls.__name__} produced entity-aligned blocks {sorted(aligned)} "
        f"but output_blocks declares {sorted(declared)}",
    )


def check_featurizer_pickle_round_trip(
    cls: type[Featurizer],
    dataset: Dataset | None = None,
    **kwargs: Any,
) -> None:
    r"""Check that a fitted featurizer survives a joblib dump and load.

    A saved model pickles its fitted featurizer, so its fitted state must live in
    picklable attributes. This transforms through the loaded copy and requires
    identical output.

    :param cls: Featurizer class to exercise.
    :param dataset: Dataset to featurize; defaults to the synthetic one.
    :param \*\*kwargs: Constructor keyword arguments.
    :raises ConformanceError: If the featurizer cannot be pickled or the loaded copy
        produces different features.
    """
    import joblib

    dataset = dataset if dataset is not None else build_synthetic_dataset()
    source = feature_source_for(cls, dataset)
    entity_ids = _entity_ids(cls, dataset)

    fitted = check_featurizer_instantiates(cls, **kwargs)
    expected = _features(fitted, source, entity_ids, fit=True)

    buffer = io.BytesIO()
    try:
        joblib.dump(fitted, buffer)
    except Exception as exc:
        raise ConformanceError(f"{cls.__name__} could not be pickled: {exc}") from exc
    buffer.seek(0)
    restored = joblib.load(buffer)
    actual = _features(restored, source, entity_ids, fit=False)

    _require(
        set(actual.blocks) == set(expected.blocks),
        f"{cls.__name__} loaded from a pickle produced blocks {sorted(actual.blocks)}, "
        f"expected {sorted(expected.blocks)}",
    )
    for name, block in expected.blocks.items():
        got = np.asarray(actual.blocks[name].values)
        want = np.asarray(block.values)
        same = got.shape == want.shape and (
            np.allclose(got, want, equal_nan=True) if want.dtype.kind in "biufc" else bool(np.all(got == want))
        )
        _require(
            same,
            f"{cls.__name__} loaded from a pickle produced different features in block {name!r}; "
            "check that every attribute transform reads is picklable",
        )


def check_predictor_instantiates(
    cls: type[Predictor],
    batch: ModelInputBatch | None = None,
    **kwargs: Any,
) -> Predictor:
    r"""Check that *cls* can be constructed with its declared defaults.

    :param cls: Predictor class to instantiate.
    :param batch: Unused; accepted so every entry of :data:`PREDICTOR_CHECKS` takes
        the same arguments and a suite can parametrize over the tuple.
    :param \*\*kwargs: Constructor keyword arguments, typically ``hyperparameters=``.
    :returns: The constructed instance.
    :raises ConformanceError: If construction fails, or the result is not a
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
    r"""Check that *cls* trains and returns one finite prediction per pair.

    An unfitted :class:`~drevalpy.plugin.MatrixPredictor` returns all-NaN, so the
    finiteness requirement is what distinguishes a predictor that trained from
    one that merely ran.

    Raises ``ConformanceError`` if fitting or predicting fails, or the predictions are
    the wrong length or not finite.

    :param cls: Predictor class to exercise.
    :param batch: Featurized batch. Defaults to
        :func:`~drevalpy.testing.build_synthetic_batch` over the synthetic
        dataset.
    :param \*\*kwargs: Constructor keyword arguments.
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


def check_predictor_pickle_round_trip(
    cls: type[Predictor],
    batch: ModelInputBatch | None = None,
    **kwargs: Any,
) -> None:
    r"""Check that a trained predictor survives a joblib dump and load.

    A fitted predictor is persisted by pickling it, so its fitted state must live in
    picklable attributes.

    :param cls: Predictor class to exercise.
    :param batch: Featurized batch; defaults to the synthetic one.
    :param \*\*kwargs: Constructor keyword arguments.
    :raises ConformanceError: If the predictor cannot be pickled or the loaded copy
        predicts differently.
    """
    import joblib

    batch = batch if batch is not None else build_synthetic_batch(build_synthetic_dataset())

    trained = check_predictor_instantiates(cls, **kwargs)
    trained.fit(batch)
    expected = np.asarray(trained.predict(batch))
    _require(
        np.isfinite(expected).all(),
        f"{cls.__name__} produced non-finite predictions before the round trip, so it cannot be compared",
    )

    buffer = io.BytesIO()
    try:
        joblib.dump(trained, buffer)
    except Exception as exc:
        raise ConformanceError(f"{cls.__name__} could not be pickled: {exc}") from exc
    buffer.seek(0)
    restored = joblib.load(buffer)
    actual = np.asarray(restored.predict(batch))

    _require(
        np.allclose(actual, expected, equal_nan=False),
        f"{cls.__name__} loaded from a pickle predicted differently; "
        "check that __getstate__/__setstate__ keep every attribute _predict reads",
    )


#: Every featurizer check, so a plugin's suite can parametrize over the set
#: rather than list them and drift when a check is added.
FEATURIZER_CHECKS: tuple[Callable[..., Any], ...] = (
    check_featurizer_instantiates,
    check_featurizer_fit_transform,
    check_featurizer_pickle_round_trip,
)

#: Every predictor check; see :data:`FEATURIZER_CHECKS`.
PREDICTOR_CHECKS: tuple[Callable[..., Any], ...] = (
    check_predictor_instantiates,
    check_predictor_fit_predict,
    check_predictor_pickle_round_trip,
)
