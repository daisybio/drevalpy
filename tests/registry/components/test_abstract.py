"""Tests for the shared abstract-member check run at registration time."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from drevalpy.registry.components._abstract import abstract_members, validate_no_abstract_methods


class _Base(ABC):
    @abstractmethod
    def alpha(self) -> None: ...

    @abstractmethod
    def beta(self) -> None: ...


class _Partial(_Base):
    def alpha(self) -> None: ...


class _Complete(_Partial):
    def beta(self) -> None: ...


class _Plain:
    pass


def test_abstract_members_are_reported_sorted_and_only_when_unimplemented() -> None:
    assert abstract_members(_Plain) == ()
    assert abstract_members(_Complete) == ()
    assert abstract_members(_Base) == ("alpha", "beta")
    assert abstract_members(_Partial) == ("beta",)


def test_complete_classes_pass_validation() -> None:
    validate_no_abstract_methods("predictor", "plain", _Plain)
    validate_no_abstract_methods("predictor", "complete", _Complete)


def test_an_incomplete_subclass_is_rejected_naming_every_missing_member_and_the_fix() -> None:
    with pytest.raises(ValueError, match=r"predictor 'partial' \(_Partial\) does not implement beta"):
        validate_no_abstract_methods("predictor", "partial", _Partial)
    with pytest.raises(ValueError, match="does not implement alpha, beta.*register a concrete subclass"):
        validate_no_abstract_methods("sample_featurizer", "base", _Base)
