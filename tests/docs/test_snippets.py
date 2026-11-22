"""Run every code block on the tutorial and how-to pages, so a stale snippet fails CI.

The offline variant runs everything except ``.. snippet: skip`` and
``.. snippet: network`` blocks. The network variant re-runs the pages that have
a ``network`` block with those blocks included; it needs read access to the
built-in dataset bucket, and is selected with ``-m network``.

Blocks train real models, so the whole file is extended tier.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"

if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

from _snippets import collect, run_page  # noqa: E402

pytestmark = pytest.mark.slow

PAGES = sorted(path for section in ("tutorials", "how-to") for path in (DOCS / section).rglob("*.rst"))


def _page_id(page: Path) -> str:
    return page.relative_to(DOCS).as_posix()


if TYPE_CHECKING:
    import pandas as pd

    from drevalpy.curation._fit import FitOptions
    from drevalpy.curation._plan import Chunk, Plan


@pytest.fixture
def isolated_drevalpy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep a page's dataset sources and environment from leaking into other tests.

    Pages register components (``-e DIR``, ``register_*``), zoo files and dataset
    sources. The autouse ``registry_snapshot`` fixture restores the in-memory
    registries, zoo included; the dataset registry persists to
    ``DREVALPY_CONFIG_DIR``, so that points into *tmp_path*.

    Curve fits are cheap here. The pages curate tiny tables to show the API, not to
    publish pEC50 values, and the default ``fit_speed="exhaustive"`` costs seconds per
    18-curve fit (``fast`` about ten times less), so ``curation._plan.plan`` and
    ``run_chunks`` are wrapped to plan every curation with ``fit_speed="fast"`` and to
    fit its chunks in this process. Both ``curate(...)`` and ``drevalpy curate run`` go
    through them; the chunks carry their options, so nothing else needs patching. The
    snippets assert no fitted values, only that the calls run.
    """
    from drevalpy.curation import _plan

    plan, run_chunks = _plan.plan, _plan.run_chunks

    def cheap_plan(
        df: pd.DataFrame, *, replicate_fits: bool, min_doses: int, metadata: Sequence[str], options: FitOptions
    ) -> Plan:
        fast = dataclasses.replace(options, fit_speed="fast")
        return plan(df, replicate_fits=replicate_fits, min_doses=min_doses, metadata=metadata, options=fast)

    def serial_run_chunks(chunks: Sequence[Chunk], max_workers: int) -> dict[str, pd.DataFrame]:
        return run_chunks(chunks, 1)

    monkeypatch.setattr(_plan, "plan", cheap_plan)
    monkeypatch.setattr(_plan, "run_chunks", serial_run_chunks)
    monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
    yield


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
