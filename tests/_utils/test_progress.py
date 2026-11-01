"""Tests for :mod:`drevalpy._utils._progress`; :func:`rss_gb` takes its ``/proc`` status path as an argument."""

from __future__ import annotations

import logging
import sys

import pytest
from upath import UPath

from drevalpy._utils import _progress
from tests._import_shims import block_imports

#: Windows has no ``resource`` module, so the rusage fallback reports 0 there.
needs_rusage = pytest.mark.skipif(sys.platform == "win32", reason="no resource module on Windows")

_STATUS = "Name:\tpython3\nVmHWM:\t 3145728 kB\nVmRSS:\t 2097152 kB\n"


def _write(tmp_path: UPath, text: str) -> str:
    path = UPath(tmp_path) / "status"
    path.write_text(text)
    return str(path)


class TestRss:
    def test_current_rss_comes_from_vmrss(self, tmp_path: UPath) -> None:
        assert _progress.rss_gb(_write(tmp_path, _STATUS)) == pytest.approx(2.0)

    @needs_rusage
    @pytest.mark.parametrize("text", [None, "VmRSS:\tnot-a-number kB\n", "VmRSS:\n", "VmSize:\t 1 kB\n"])
    def test_rss_falls_back_to_the_peak_without_a_readable_vmrss(self, tmp_path: UPath, text: str | None) -> None:
        path = str(UPath(tmp_path) / "absent") if text is None else _write(tmp_path, text)

        assert _progress.rss_gb(path) == _progress._peak_rss_gb() > 0

    @needs_rusage
    def test_peak_rss_is_normalised_to_kib_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_progress.sys, "platform", "linux")

        linux = _progress._peak_rss_gb()

        monkeypatch.setattr(_progress.sys, "platform", "darwin")
        assert linux == pytest.approx(_progress._peak_rss_gb() * 1024)

    def test_peak_rss_is_zero_without_the_resource_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        block_imports(monkeypatch, "resource")

        assert _progress._peak_rss_gb() == 0.0


def test_log_stage_emits_one_lazily_formatted_record_at_the_requested_level(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(_progress, "rss_gb", lambda: 2.0)
    monkeypatch.setattr(_progress, "_peak_rss_gb", lambda: 3.0)
    logger = logging.getLogger("drevalpy.tests.progress")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        _progress.log_stage(logger, "plot heatmap")
        _progress.log_stage(logger, "load", level=logging.DEBUG)

    assert [r.getMessage() for r in caplog.records] == [
        "plot heatmap | rss=2.00 GB peak=3.00 GB",
        "load | rss=2.00 GB peak=3.00 GB",
    ]
    assert [r.levelno for r in caplog.records] == [logging.INFO, logging.DEBUG]
