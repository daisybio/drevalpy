"""Tests for :mod:`drevalpy.visualization._progress`; the readers take their ``/proc`` and cgroup paths as arguments."""

from __future__ import annotations

import logging
import sys

import pytest
from upath import UPath

from drevalpy.visualization import _progress
from tests._import_shims import block_imports

#: Windows has no ``resource`` module, so the rusage fallback reports 0 there.
needs_rusage = pytest.mark.skipif(sys.platform == "win32", reason="no resource module on Windows")

_STATUS = """Name:\tpython3
VmPeak:\t 8388608 kB
VmSize:\t 8388608 kB
VmHWM:\t 3145728 kB
VmRSS:\t 2097152 kB
Threads:\t1
"""


@pytest.fixture()
def status_file(tmp_path: UPath) -> str:
    path = UPath(tmp_path) / "status"
    path.write_text(_STATUS)
    return str(path)


def _write(tmp_path: UPath, name: str, text: str) -> str:
    path = UPath(tmp_path) / name
    path.write_text(text)
    return str(path)


def _absent(tmp_path: UPath) -> str:
    return str(UPath(tmp_path) / "absent")


class TestReadStatus:
    def test_reads_the_requested_field_in_kib(self, status_file: str) -> None:
        assert _progress._read_status_kib("VmRSS", status_file) == 2097152
        assert _progress._read_status_kib("VmSize", status_file) == 8388608

    def test_missing_field_file_value_or_number_is_none(self, status_file: str, tmp_path: UPath) -> None:
        assert _progress._read_status_kib("VmSwap", status_file) is None
        assert _progress._read_status_kib("VmRSS", _absent(tmp_path)) is None
        assert _progress._read_status_kib("VmRSS", _write(tmp_path, "bad", "VmRSS:\tnot-a-number kB\n")) is None
        assert _progress._read_status_kib("VmRSS", _write(tmp_path, "empty", "VmRSS:\n")) is None


class TestRss:
    def test_current_and_peak_rss_come_from_vmrss_and_vmhwm(self, status_file: str) -> None:
        assert _progress.rss_gb(status_file) == pytest.approx(2.0)
        assert _progress.peak_rss_gb(status_file) == pytest.approx(3.0)

    @needs_rusage
    def test_rss_falls_back_to_rusage_without_proc(self, tmp_path: UPath) -> None:
        assert _progress.rss_gb(_absent(tmp_path)) > 0
        assert _progress.peak_rss_gb(_absent(tmp_path)) > 0
        assert _progress.rss_gb() > 0

    @needs_rusage
    def test_rusage_is_normalised_to_kib_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_progress.sys, "platform", "linux")

        linux = _progress._rusage_max_rss_gb()

        monkeypatch.setattr(_progress.sys, "platform", "darwin")
        assert linux == pytest.approx(_progress._rusage_max_rss_gb() * 1024)

    def test_rusage_reports_zero_without_the_resource_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        block_imports(monkeypatch, "resource")

        assert _progress._rusage_max_rss_gb() == 0.0


class TestMemoryLimit:
    def test_reads_cgroup_v2_bytes_then_falls_back_to_v1(self, tmp_path: UPath) -> None:
        v2 = _write(tmp_path, "memory.max", str(36 * 1024**3))
        v1 = _write(tmp_path, "limit_in_bytes", str(8 * 1024**3))

        assert _progress.memory_limit_gb(v2, _absent(tmp_path)) == pytest.approx(36.0)
        assert _progress.memory_limit_gb(_absent(tmp_path), v1) == pytest.approx(8.0)

    @pytest.mark.parametrize("content", ["max\n", str(2**63 - 1), "\n", "unlimited\n"])
    def test_unbounded_empty_or_unparseable_values_are_not_a_limit(self, tmp_path: UPath, content: str) -> None:
        v2 = _write(tmp_path, "memory.max", content)
        v1 = _write(tmp_path, "limit_in_bytes", str(4 * 1024**3))

        assert _progress.memory_limit_gb(v2, _absent(tmp_path)) is None
        if content == "max\n":
            assert _progress.memory_limit_gb(v2, v1) == pytest.approx(4.0)

    def test_both_paths_absent_is_none(self, tmp_path: UPath) -> None:
        assert _progress.memory_limit_gb(_absent(tmp_path), _absent(tmp_path)) is None


class TestFormatStage:
    def test_names_the_stage_memory_figures_and_the_limit_with_headroom_when_capped(
        self, status_file: str, tmp_path: UPath
    ) -> None:
        absent = _absent(tmp_path)
        capped = _write(tmp_path, "memory.max", str(12 * 1024**3))

        uncapped = _progress.format_stage("load", status_path=status_file, v2_path=absent, v1_path=absent)
        line = _progress.format_stage("load", status_path=status_file, v2_path=capped, v1_path=capped)

        assert uncapped == "load | rss=2.00 GB peak=3.00 GB"
        assert line == "load | rss=2.00 GB peak=3.00 GB limit=12.00 GB (25%)"

    def test_a_zero_limit_is_not_divided_by(self, status_file: str, tmp_path: UPath) -> None:
        v2 = _write(tmp_path, "memory.max", "0")

        line = _progress.format_stage("load", status_path=status_file, v2_path=v2, v1_path=v2)

        assert "limit=" not in line


class TestLogStage:
    def test_emits_one_lazily_formatted_record_at_the_requested_level(
        self, status_file: str, tmp_path: UPath, caplog: pytest.LogCaptureFixture
    ) -> None:
        absent = _absent(tmp_path)
        logger = logging.getLogger("drevalpy.tests.progress")
        paths = {"status_path": status_file, "v2_path": absent, "v1_path": absent}

        with caplog.at_level(logging.DEBUG, logger=logger.name):
            _progress.log_stage(logger, "plot heatmap", **paths)
            _progress.log_stage(logger, "load", level=logging.DEBUG, **paths)

        assert [r.getMessage() for r in caplog.records] == [
            "plot heatmap | rss=2.00 GB peak=3.00 GB",
            "load | rss=2.00 GB peak=3.00 GB",
        ]
        assert [r.levelno for r in caplog.records] == [logging.INFO, logging.DEBUG]
        assert caplog.records[0].msg == "%s"
