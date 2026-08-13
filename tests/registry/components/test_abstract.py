"""Tests for the shared abstract-member check run at registration time."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from drevalpy.registry.components._abstract import abstract_members, validate_no_abstract_methods


class _Base(ABC):
    """Two-method abstract base used to build the fixtures below."""

    @abstractmethod
    def alpha(self) -> None:
        """First required member."""

    @abstractmethod
    def beta(self) -> None:
        """Second required member."""


class _Partial(_Base):
    """Implements one of the two required members."""

    def alpha(self) -> None:
        """Implemented."""


class _Complete(_Partial):
    """Implements both required members."""

    def beta(self) -> None:
        """Implemented."""


class _Plain:
    """A class that is not an ABC at all."""


def test_a_plain_class_has_no_abstract_members() -> None:
    assert abstract_members(_Plain) == ()


def test_a_complete_subclass_has_no_abstract_members() -> None:
    assert abstract_members(_Complete) == ()


def test_abstract_members_are_reported_sorted() -> None:
    assert abstract_members(_Base) == ("alpha", "beta")


def test_only_the_unimplemented_members_are_reported() -> None:
    assert abstract_members(_Partial) == ("beta",)


def test_a_plain_class_passes_validation() -> None:
    validate_no_abstract_methods("predictor", "plain", _Plain)


def test_a_complete_subclass_passes_validation() -> None:
    validate_no_abstract_methods("predictor", "complete", _Complete)


def test_an_incomplete_subclass_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"predictor 'partial' \(_Partial\) does not implement beta"):
        validate_no_abstract_methods("predictor", "partial", _Partial)


def test_the_error_names_every_missing_member() -> None:
    with pytest.raises(ValueError, match="does not implement alpha, beta"):
        validate_no_abstract_methods("predictor", "base", _Base)


def test_the_error_suggests_the_fix() -> None:
    with pytest.raises(ValueError, match="register a concrete subclass"):
        validate_no_abstract_methods("cell_line_featurizer", "base", _Base)
