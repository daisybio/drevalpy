"""A recording stand-in for the ``wandb`` module.

``wandb`` is on the import-cost policy's forbidden list, so the code under test looks
it up in ``sys.modules`` or imports it inside a function. Injecting this fake there
pins what the tracking layer does without a network, an account or the real package.

Sibling of :mod:`tests._import_shims`: an ``_``-prefixed module at the ``tests/`` root,
imported as ``from tests._fake_wandb import install_fake_wandb``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class FakeRun:
    """One recorded run: what it was started with, what it logged, how it ended."""

    project: str
    name: str
    config: dict[str, Any]
    tags: list[str]
    summary: dict[str, Any] = field(default_factory=dict)
    logged: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    exit_code: int | None = None


class FakeWandb:
    """The slice of the ``wandb`` API drevalpy uses, recording every call."""

    def __init__(self) -> None:
        self.run: FakeRun | None = None
        self.runs: list[FakeRun] = []

    def init(self, *, project: str, name: str, config: dict[str, Any], tags: list[str]) -> FakeRun:
        self.run = FakeRun(project=project, name=name, config=config, tags=tags)
        self.runs.append(self.run)
        return self.run

    def finish(self, exit_code: int | None = None) -> None:
        if self.run is not None:
            self.run.finished = True
            self.run.exit_code = exit_code
        self.run = None

    def log(self, metrics: dict[str, Any]) -> None:
        assert self.run is not None, "wandb.log called without an active run"
        self.run.logged.append(dict(metrics))

    def run_names(self) -> list[str]:
        return [run.name for run in self.runs]


def install_fake_wandb(monkeypatch: pytest.MonkeyPatch) -> FakeWandb:
    """Make ``import wandb`` resolve to a fresh :class:`FakeWandb`.

    :param monkeypatch: Pytest fixture; the module is removed again at teardown.
    :returns: The installed fake.
    """
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake
