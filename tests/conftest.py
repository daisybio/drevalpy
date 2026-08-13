"""Pytest configuration for the drevalpy test suite.

The suite is deliberately self-contained: no fixture downloads anything and no
test resolves against a developer's local ``data/`` directory. Every dataset the
suite needs is built in memory by
:func:`tests.synthetic.build_synthetic_dataset`.
"""

from __future__ import annotations

import shutil
import zipfile
from typing import Any

import numpy as np
import pytest
from upath import UPath

import drevalpy.registry  # noqa: F401  -- triggers register_builtin_components() + discover_plugins()
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import FeatureSource
from tests._trusted_subprocess import run_trusted_python
from tests.synthetic import build_synthetic_dataset

REPO_ROOT = UPath(__file__).resolve().parents[1]

#: Builds the wheel for :func:`built_wheel_contents` in a fresh interpreter.
#: ``sys.argv[1]`` is the output directory.
_BUILD_WHEEL_SCRIPT = (
    "import subprocess, sys; sys.exit(subprocess.run(['uv', 'build', '--wheel', '--out-dir', sys.argv[1]]).returncode)"
)


@pytest.fixture(autouse=True)
def _ensure_registries_populated():
    """Ensure built-in components are registered before each test.

    Some tests call registry.clear() for isolation. This fixture guarantees
    the next test always starts with populated registries.
    """
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()


@pytest.fixture(scope="session")
def synthetic_dataset() -> Dataset:
    """Session-wide synthetic raw-omics dataset standing in for a real ``.h5mu``.

    Built once because rdkit fingerprints, molecular graphs and the learned BPE
    merges cost more than the tests that consume them. Nothing in the suite
    writes to it, so sharing is safe.

    :returns: Dataset with complete modality coverage; see
        :mod:`tests.synthetic.builders` for its exact shape.
    """
    return build_synthetic_dataset()


@pytest.fixture(scope="session")
def built_wheel_contents(tmp_path_factory: pytest.TempPathFactory) -> frozenset[str]:
    """Namelist of a real ``uv build --wheel`` of this repository, built once.

    Two packaging tests assert against the wheel that consumers actually get:
    ``tests/plugin/test_init.py`` for the PEP 561 marker and
    ``tests/testing/test_init.py`` for the shipped ``drevalpy.testing``
    submodules. The build is identical for both, so it runs once per session and
    the namelist is shared read-only rather than paying for a second hatchling
    run. Skips when ``uv`` is unavailable, which is what makes the tests that
    request it skip too.

    :param tmp_path_factory: Session-scoped temporary directory factory.
    :returns: Every archive member name in the built wheel.
    """
    if shutil.which("uv") is None:
        pytest.skip("needs uv to build a wheel")

    out_dir = UPath(tmp_path_factory.mktemp("built_wheel"))
    result = run_trusted_python(_BUILD_WHEEL_SCRIPT, cwd=str(REPO_ROOT), extra_args=[str(out_dir)])
    if result.returncode != 0:
        pytest.fail(f"uv build --wheel failed:\n{result.stderr}")

    wheels = list(out_dir.glob("*.whl"))
    if not wheels:
        pytest.fail("uv build --wheel produced no wheel")
    with zipfile.ZipFile(str(wheels[0])) as archive:
        return frozenset(archive.namelist())


class MockFeatureSource(FeatureSource):
    """Test helper implementing the FeatureSource ABC."""

    def __init__(self, features: dict[str, dict[str, Any]], meta_info: dict[str, Any] | None = None):
        """Initialize with features dict and optional metadata.

        :param features: Mapping of entity_id -> {view_name -> feature_array}.
        :param meta_info: Optional mapping of view_name -> feature names or metadata.
        """
        self._features = features
        self._meta_info = meta_info or {}

    @property
    def identifiers(self) -> np.ndarray:
        """All available entity IDs."""
        return np.array(list(self._features.keys()))

    @property
    def mdata(self) -> Any:
        """No MuData backing for mocks."""
        return None

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        """Direct access to the backing features dict."""
        return self._features

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Return (len(ids), n_features) float array for a dense numeric view."""
        rows = [np.asarray(self._features[str(eid)][view], dtype=np.float64).ravel() for eid in entity_ids]
        return np.vstack(rows)

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return ordered feature/column names for a view, or None."""
        meta = self._meta_info.get(view)
        return tuple(str(n) for n in meta) if meta is not None else None

    def get_entity_view(self, entity_id: str, view: str) -> Any:
        """Return the raw per-entity object for non-numeric views (graphs, etc.)."""
        entity = self._features.get(str(entity_id))
        if entity is None:
            return None
        return entity.get(view)

    def get_metadata(self, key: str) -> Any:
        """Return arbitrary metadata (e.g. ontology structures)."""
        return self._meta_info.get(key)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest session defaults and a headless Matplotlib backend.

    :param config: Pytest configuration object.
    """
    import matplotlib

    matplotlib.use("Agg")
    config.option.flaky_report = "none"
    config.option.tbstyle = "short"
