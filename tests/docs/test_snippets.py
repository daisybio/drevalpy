"""Run every code block on the tutorial and how-to pages, so a stale snippet fails CI.

The offline variant runs everything except ``.. snippet: skip`` and
``.. snippet: network`` blocks. The network variant re-runs the pages that have
a ``network`` block with those blocks included; it needs read access to the
built-in dataset bucket, and is selected with ``-m network``.

Blocks train real models, so the whole file is extended tier.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"

if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

from _snippets import collect, run_page  # noqa: E402

pytestmark = pytest.mark.slow

PAGES = sorted(path for section in ("tutorials", "how-to") for path in (DOCS / section).rglob("*.rst"))


def _page_id(page: Path) -> str:
    return page.relative_to(DOCS).as_posix()


@pytest.fixture
def isolated_drevalpy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep a page's registrations, config and zoo files from leaking into other tests.

    Pages register components (``-e DIR``, ``register_*``), dataset sources and
    zoo files. The component registries are rolled back to the names they had
    before the page; the dataset registry persists to ``DREVALPY_CONFIG_DIR``,
    so that points into *tmp_path*.
    """
    from drevalpy.models.zoo import clear_external_zoo
    from drevalpy.registry.drug_featurizer import drug_featurizer_registry
    from drevalpy.registry.predictor import predictor_registry
    from drevalpy.registry.sample_featurizer import sample_featurizer_registry
    from drevalpy.registry.splitter import splitter_registry
    from drevalpy.registry.visualization import visualization_registry

    monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
    kept = [
        (predictor_registry, frozenset(predictor_registry.list_names())),
        (sample_featurizer_registry, frozenset(sample_featurizer_registry.list_names())),
        (drug_featurizer_registry, frozenset(drug_featurizer_registry.list_names())),
        (splitter_registry, frozenset(splitter_registry.modes)),
        (visualization_registry, frozenset(visualization_registry.names)),
    ]
    yield
    for registry, names in kept:
        registry.retain_only(names)
    clear_external_zoo()


def test_the_runner_sees_the_tutorial_and_how_to_pages() -> None:
    """An empty parametrization would pass vacuously, so pin that the globs find the pages."""
    sections = {path.relative_to(DOCS).parts[0] for path in PAGES}
    assert sections == {"tutorials", "how-to"}


@pytest.mark.parametrize("page", PAGES, ids=_page_id)
def test_page_snippets_run_offline(page: Path, tmp_path: Path, isolated_drevalpy: None) -> None:
    run_page(page, tmp_path / "work", allow_network=False)


@pytest.mark.network
@pytest.mark.parametrize("page", PAGES, ids=_page_id)
def test_page_snippets_run_with_network(page: Path, tmp_path: Path, isolated_drevalpy: None) -> None:
    if not any(snippet.marker == "network" for snippet in collect(page)):
        pytest.skip("no network snippets on this page; the offline variant covers it")
    run_page(page, tmp_path / "work", allow_network=True)
