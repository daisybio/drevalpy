"""Pytest configuration for the drevalpy test suite.

The suite is deliberately self-contained: no fixture downloads anything and no
test resolves against a developer's local ``data/`` directory. Every dataset the
suite needs is built in memory by
:func:`tests.synthetic.build_synthetic_dataset`.
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from upath import UPath

import drevalpy.registry  # noqa: F401  -- registers the built-ins and discovers plugins
import tests.registry._helpers  # noqa: F401  -- captures the built-in component state before any test module registers
from drevalpy.registry._all import restore_all, snapshot_all
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import FeatureSource, Side
from tests._trusted_subprocess import run_trusted_python
from tests.synthetic import build_synthetic_dataset

REPO_ROOT = UPath(__file__).resolve().parents[1]

#: Builds the wheel for :func:`built_wheel_contents` in a fresh interpreter.
#: ``sys.argv[1]`` is the output directory.
_BUILD_WHEEL_SCRIPT = (
    "import subprocess, sys; sys.exit(subprocess.run(['uv', 'build', '--wheel', '--out-dir', sys.argv[1]]).returncode)"
)


@pytest.fixture(autouse=True)
def registry_snapshot() -> Iterator[None]:
    """Restore every in-memory registry after each test, so registrations never leak between tests.

    :yields: Control to the test.
    """
    snapshots = snapshot_all()
    yield
    restore_all(snapshots)


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
    the namelist is shared read-only rather than paying for a second build. Skips
    when ``uv`` is unavailable, which is what makes the tests that request it skip
    too.

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
    """In-memory feature source: per-entity view values and optional names or ``uns`` metadata."""

    def __init__(
        self,
        features: dict[str, dict[str, Any]],
        meta_info: dict[str, Any] | None = None,
        *,
        side: Side = "sample",
    ):
        """Initialize with features dict and optional metadata.

        :param features: Mapping of entity_id -> {view_name -> feature_array}; ``"tissue"`` holds the label.
        :param meta_info: Optional mapping of view_name -> feature names, or of ``uns`` key -> value.
        :param side: Entity side the source serves.
        """
        self.side = side
        self._features = features
        self._meta_info = meta_info or {}

    @property
    def identifiers(self) -> np.ndarray:
        """All entity IDs of the mock."""
        return np.array(list(self._features.keys()))

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        """Direct access to the backing features dict."""
        return self._features

    def has_view(self, name: str) -> bool:
        """Report the views that have feature names."""
        return self.feature_names(name) is not None

    def view(self, name: str, ids: np.ndarray) -> np.ndarray:
        """Return (len(ids), n_features) float array for a dense numeric view."""
        rows = [np.asarray(self._features[str(eid)][name], dtype=np.float64).ravel() for eid in ids]
        return np.vstack(rows)

    def feature_names(self, name: str) -> tuple[str, ...] | None:
        """Return ordered feature/column names for a view, or None."""
        meta = self._meta_info.get(name)
        return tuple(str(n) for n in meta) if meta is not None else None

    def view_settings(self, name: str) -> dict[str, Any] | None:
        """No view of a mock records settings."""
        return None

    def expression_unit(self, name: str) -> str | None:
        """No view of a mock records a unit."""
        return None

    def smiles(self, ids: np.ndarray) -> Any:
        """A mock has no SMILES."""
        return None

    def tissue(self, ids: np.ndarray) -> np.ndarray | None:
        """Return the ``"tissue"`` value of every entity, NaN where it has none."""
        labels = []
        for eid in ids:
            raw = self._features.get(str(eid), {}).get("tissue")
            labels.append(np.nan if raw is None else np.asarray(raw).reshape(-1)[0])
        return np.array(labels, dtype=object)

    def uns(self, key: str) -> Any:
        """Return arbitrary metadata (e.g. ontology structures)."""
        return self._meta_info.get(key)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest session defaults and a headless Matplotlib backend.

    :param config: Pytest configuration object.
    """
    import matplotlib

    matplotlib.use("Agg")
    config.option.tbstyle = "short"
