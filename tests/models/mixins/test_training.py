"""Tests for the ``DRPModel`` train / predict surface.

Mirrors :mod:`drevalpy.models.mixins._training`. The happy paths for a real model
are covered end to end in ``tests/models/test_drp_model.py``; what is asserted
here is the surface's own contract - the two guard clauses, the scope resolution
``predict`` accepts three spellings of, and the ResponseBatch form no call site in
the library exercises.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from drevalpy.models import construct_model
from drevalpy.models.mixins._training import DRPTrainingMixin, _resolve_predict_scope
from drevalpy.types import SplitMask, SplitMasks
from tests.models.synthetic_fixtures import (
    lco_split_masks,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
)


class _Stackless(DRPTrainingMixin):
    """Model-shaped object that was never given a component stack."""

    def __init__(self) -> None:
        self._stack = None
        self._empty_training = False

    @classmethod
    def get_model_name(cls) -> str:
        return "Stackless"


class _RecordingStack:
    """Stack that records which fit entry point the mixin chose."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _fit_featurizers_and_predictor(self, output, cell_line_input, drug_input, **kwargs) -> None:
        self.calls.append(("features", {"output": output, "cl": cell_line_input, "dr": drug_input, **kwargs}))

    def is_fitted(self) -> bool:
        return True


class _RecordingModel(DRPTrainingMixin):
    """Model-shaped object over a recording stack."""

    def __init__(self) -> None:
        self._stack = _RecordingStack()
        self._empty_training = False

    @classmethod
    def get_model_name(cls) -> str:
        return "Recording"


class _Batch(list):
    """Response batch stand-in whose only relevant property is its length."""


class TestTheStackGuard:
    """Neither entry point may silently no-op on an unmaterialized model."""

    def test_train_refuses_a_model_without_a_stack(self) -> None:
        with pytest.raises(RuntimeError, match="has not been constructed with a component stack"):
            _Stackless().train(synthetic_mudataset_identity(), lco_split_masks())

    def test_predict_refuses_a_model_without_a_stack(self) -> None:
        with pytest.raises(RuntimeError, match="has not been constructed with a component stack"):
            _Stackless().predict(synthetic_mudataset_identity(), lco_split_masks())


class TestTrainRejectsAnIncompleteCall:
    """A call matching neither accepted form is a ``TypeError``, not a no-op fit."""

    def test_no_arguments_at_all(self) -> None:
        with pytest.raises(TypeError, match=r"train\(\) requires either"):
            _RecordingModel().train()

    def test_a_dataset_without_a_scope(self) -> None:
        with pytest.raises(TypeError, match=r"train\(\) requires either"):
            _RecordingModel().train(synthetic_mudataset_identity())


class TestTheFeatureSourceForm:
    """``(output, cell_line_input, drug_input)`` is only reachable by hand."""

    def test_it_reaches_the_featurizer_and_predictor_fit(self) -> None:
        model = _RecordingModel()

        model.train(_Batch([1, 2, 3]), "cl_source", "dr_source")

        assert [name for name, _ in model._stack.calls] == ["features"]
        assert model._empty_training is False

    def test_it_forwards_the_early_stopping_batch_and_a_context(self) -> None:
        model = _RecordingModel()

        model.train(_Batch([1]), "cl_source", output_earlystopping="es_batch")

        _, kwargs = model._stack.calls[0]
        assert kwargs["output_earlystopping"] == "es_batch"
        assert kwargs["training_context"].logging_metadata == {"model_name": "Recording"}

    def test_the_checkpoint_directory_reaches_the_context(self) -> None:
        model = _RecordingModel()

        model.train(_Batch([1]), "cl_source", model_checkpoint_dir="/tmp/ckpt")  # noqa: S108

        _, kwargs = model._stack.calls[0]
        assert str(kwargs["training_context"].checkpoint_dir) == "/tmp/ckpt"  # noqa: S108

    def test_an_empty_batch_records_empty_training_without_fitting(self) -> None:
        model = _RecordingModel()

        model.train(_Batch([]), "cl_source")

        assert model._empty_training is True
        assert model._stack.calls == []


class TestResolvePredictScope:
    """``predict`` accepts the mask as a keyword, a positional, or inside a split."""

    def test_an_explicit_scope_wins(self) -> None:
        split = lco_split_masks()

        assert _resolve_predict_scope(split, scope=split.train, split=None) is split.train

    def test_a_positional_mask_is_used_as_is(self) -> None:
        split = lco_split_masks()

        assert _resolve_predict_scope(split.val, scope=None, split=None) is split.val

    def test_positional_split_masks_resolve_to_the_test_mask(self) -> None:
        split = lco_split_masks()

        assert _resolve_predict_scope(split, scope=None, split=None) is split.test

    def test_keyword_split_masks_resolve_to_the_test_mask(self) -> None:
        split = lco_split_masks()

        assert _resolve_predict_scope(None, scope=None, split=split) is split.test

    def test_nothing_at_all_resolves_to_none(self) -> None:
        assert _resolve_predict_scope(None, scope=None, split=None) is None


class TestPredictAgainstARealModel:
    """The remaining branches need a stack that can actually answer."""

    def test_it_refuses_a_missing_scope(self) -> None:
        model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})

        with pytest.raises(TypeError, match=r"predict\(\) requires"):
            model.predict(synthetic_mudataset_gene_expression_fingerprints())

    def test_it_refuses_a_missing_dataset(self) -> None:
        model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})

        with pytest.raises(TypeError, match=r"predict\(\) requires"):
            model.predict(scope=lco_split_masks().test)

    def test_it_refuses_an_untrained_model(self) -> None:
        model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})

        with pytest.raises(RuntimeError, match="has not been trained"):
            model.predict(synthetic_mudataset_gene_expression_fingerprints(), lco_split_masks())

    def test_an_empty_training_scope_answers_nan_instead_of_raising(self) -> None:
        model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})
        mudataset = synthetic_mudataset_gene_expression_fingerprints()
        empty = SplitMasks(
            train=SplitMask(np.zeros((2, 2), dtype=bool)),
            test=lco_split_masks().test,
            val=SplitMask(np.zeros((2, 2), dtype=bool)),
        )

        model.train(mudataset, empty)
        predictions = model.predict(mudataset, empty)

        assert model._empty_training is True
        assert np.isnan(predictions).all()

    def test_early_stopping_takes_the_other_dataset_branch(self) -> None:
        """A non-empty ``val`` mask routes through ``train_with_early_stopping``."""
        model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})
        mudataset = synthetic_mudataset_gene_expression_fingerprints()
        split = SplitMasks(
            train=SplitMask(np.array([[True, False], [False, False]])),
            test=SplitMask(np.array([[False, False], [True, False]])),
            val=SplitMask(np.array([[False, True], [False, False]])),
        )

        model.train(mudataset, split)

        assert model._stack is not None
        assert model._stack.is_fitted()
