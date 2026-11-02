"""Tests for the license a component inherits from the package that ships it."""

from __future__ import annotations

import importlib.metadata

import pytest

from drevalpy.registry import _license
from drevalpy.registry._license import distribution_license
from drevalpy.registry.predictor import get as get_predictor


class _Fake:
    """Stands in for a distribution, exposing only ``metadata``."""

    def __init__(self, **metadata: str) -> None:
        self.metadata = metadata


@pytest.fixture
def fake_distribution(monkeypatch: pytest.MonkeyPatch):
    """Let the test choose the metadata of the distribution ``fakepkg``."""
    _license._package_license.cache_clear()
    holder: dict[str, _Fake] = {}

    def distribution(name: str):
        if name == "fakepkg" and "dist" in holder:
            return holder["dist"]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(importlib.metadata, "packages_distributions", dict)
    yield holder
    _license._package_license.cache_clear()


def _class_in(module: str) -> type:
    return type("Component", (), {"__module__": module})


def test_a_builtin_has_drevalpy_s_license() -> None:
    assert distribution_license(get_predictor("ridge")) == "MIT"


def test_a_class_outside_an_installed_package_has_no_assertion() -> None:
    assert distribution_license(_class_in("my_extension_file")) == "NOASSERTION"


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"License-Expression": "apache-2.0"}, "Apache-2.0"),
        ({"License": "BSD-3-Clause"}, "BSD-3-Clause"),
        ({"License": "Some free-text license, see LICENSE"}, "NOASSERTION"),
        ({}, "NOASSERTION"),
    ],
    ids=["expression", "legacy-spdx", "legacy-free-text", "none"],
)
def test_the_package_license_is_read_from_its_metadata(
    fake_distribution: dict[str, _Fake], metadata: dict[str, str], expected: str
) -> None:
    fake_distribution["dist"] = _Fake(**metadata)

    assert distribution_license(_class_in("fakepkg.featurizers")) == expected
