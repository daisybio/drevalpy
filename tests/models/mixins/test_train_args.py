"""Tests for the resolution of ``DRPModel.train``'s two accepted call shapes.

Mirrors :mod:`drevalpy.models.mixins._train_args`, whose job is to turn one
``train`` call - positional or keyword, Dataset form or ResponseBatch form - into
a :class:`TrainCallArgs` the caller can dispatch on. Asserted directly rather
than through ``train`` because the interesting cases are precisely the ones no
call site in the library makes: the compat spellings that only exist for
hand-rolled models.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.models.mixins._train_args import TrainCallArgs, resolve_train_args
from drevalpy.types import SplitMask, SplitMasks
from tests.models.synthetic_fixtures import (
    lco_split_masks,
    synthetic_mudataset_identity,
)


@pytest.fixture
def dataset():
    """The cheap 2x2 synthetic dataset."""
    return synthetic_mudataset_identity()


@pytest.fixture
def split() -> SplitMasks:
    """Leave-cell-line-out masks over that dataset, with an empty ``val``."""
    return lco_split_masks()


def _masks_with_val() -> SplitMasks:
    """Split masks whose ``val`` mask selects something."""
    return SplitMasks(
        train=SplitMask(np.array([[True, False], [False, False]])),
        test=SplitMask(np.array([[False, False], [True, False]])),
        val=SplitMask(np.array([[False, True], [False, False]])),
    )


class TestTheDatasetForm:
    """``(mudataset, scope)`` - what every call site inside the library uses."""

    def test_keywords_are_taken_as_given(self, dataset, split) -> None:
        args = resolve_train_args(mudataset=dataset, scope=split.train)

        assert args.is_dataset_form
        assert not args.is_feature_source_form
        assert args.mudataset is dataset
        assert args.scope is split.train

    def test_a_dataset_in_the_first_positional_slot_is_recognised(self, dataset, split) -> None:
        args = resolve_train_args(dataset, split.train)

        assert args.is_dataset_form
        assert args.mudataset is dataset

    def test_a_positional_split_masks_is_narrowed_to_its_train_mask(self, dataset, split) -> None:
        args = resolve_train_args(dataset, split)

        assert args.scope is split.train

    def test_a_keyword_split_masks_is_narrowed_too(self, dataset, split) -> None:
        args = resolve_train_args(mudataset=dataset, split=split)

        assert args.scope is split.train

    def test_a_non_empty_val_mask_becomes_the_early_stopping_scope(self, dataset) -> None:
        masks = _masks_with_val()

        args = resolve_train_args(dataset, masks)

        assert args.early_stopping_scope is masks.val

    def test_an_empty_val_mask_leaves_early_stopping_off(self, dataset, split) -> None:
        args = resolve_train_args(dataset, split)

        assert args.early_stopping_scope is None

    def test_an_explicit_early_stopping_scope_survives_narrowing(self, dataset, split) -> None:
        explicit = split.test

        args = resolve_train_args(dataset, split, early_stopping_scope=explicit)

        assert args.early_stopping_scope is explicit

    def test_an_explicit_scope_wins_over_a_positional_split(self, dataset, split) -> None:
        args = resolve_train_args(dataset, split, scope=split.test)

        assert args.scope is split.test
        assert args.early_stopping_scope is None


class TestTheFeatureSourceForm:
    """``(output, cell_line_input, drug_input)`` - for hand-rolled models."""

    def test_keywords_are_taken_as_given(self) -> None:
        args = resolve_train_args(output="batch", cell_line_input="cl", drug_input="dr")

        assert args.is_feature_source_form
        assert not args.is_dataset_form
        assert (args.output, args.cell_line_input, args.drug_input) == ("batch", "cl", "dr")

    def test_a_non_dataset_first_positional_becomes_output(self) -> None:
        args = resolve_train_args("batch", "cl", "dr")

        assert args.is_feature_source_form
        assert args.output == "batch"
        assert args.cell_line_input == "cl"
        assert args.drug_input == "dr"

    def test_a_non_mask_second_positional_becomes_the_cell_line_input(self) -> None:
        args = resolve_train_args(output="batch", cell_line_input=None)

        assert not args.is_feature_source_form

        args = resolve_train_args("batch", "cl")

        assert args.is_feature_source_form


class TestNeitherForm:
    """An incomplete call resolves to neither form, which is what ``train`` reports."""

    def test_an_empty_call_is_neither(self) -> None:
        args = resolve_train_args()

        assert not args.is_dataset_form
        assert not args.is_feature_source_form

    def test_a_dataset_without_a_scope_is_neither(self, dataset) -> None:
        args = resolve_train_args(dataset)

        assert not args.is_dataset_form
        assert not args.is_feature_source_form

    def test_an_output_without_a_cell_line_input_is_neither(self) -> None:
        args = resolve_train_args("batch")

        assert not args.is_feature_source_form


def test_the_resolved_arguments_are_immutable(dataset, split) -> None:
    """A frozen dataclass keeps ``train``'s branches from editing the call."""
    args = resolve_train_args(dataset, split.train)

    with pytest.raises(AttributeError):
        args.scope = split.test  # type: ignore[misc]


def test_an_empty_result_defaults_to_no_inputs() -> None:
    assert TrainCallArgs() == TrainCallArgs(
        mudataset=None,
        scope=None,
        early_stopping_scope=None,
        output=None,
        cell_line_input=None,
        drug_input=None,
    )
