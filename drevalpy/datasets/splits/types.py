"""Shared types for test-mode split providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..dataset import DrugResponseDataset

TEST_MODES: frozenset[str] = frozenset({"LPO", "LCO", "LDO", "LTO"})
REQUIRED_ROLES: tuple[str, ...] = ("train", "validation", "test")
OPTIONAL_ROLES: tuple[str, ...] = ("validation_es", "early_stopping")

SplitFold = dict[str, DrugResponseDataset]
SplitResult = tuple[list[SplitFold], list[dict[str, Any]]]


class SplitError(ValueError):
    """Raised when split settings or provider output are invalid."""


@dataclass(frozen=True)
class SplitParams:
    """Pipeline split settings passed to built-in and external split providers."""

    test_mode: str
    n_cv_splits: int
    validation_ratio: float
    random_state: int
    split_early_stopping: bool


ExternalSplitCreator = Callable[[DrugResponseDataset, SplitParams], list[dict[str, Any]]]
SplitCreator = ExternalSplitCreator


def make_split_params(
    *,
    test_mode: str,
    n_cv_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    split_early_stopping: bool = True,
) -> SplitParams:
    """
    Build ``SplitParams`` from pipeline keyword arguments.

    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``
    :param n_cv_splits: requested number of CV splits
    :param validation_ratio: validation fraction of the training set
    :param random_state: random seed for splitting
    :param split_early_stopping: whether to derive early-stopping roles
    :returns: frozen split settings for providers
    """
    return SplitParams(
        test_mode=test_mode,
        n_cv_splits=n_cv_splits,
        validation_ratio=validation_ratio,
        random_state=random_state,
        split_early_stopping=split_early_stopping,
    )
