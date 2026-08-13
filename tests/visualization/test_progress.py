"""Tests for :mod:`drevalpy.visualization._progress`.

The readers take their ``/proc`` and cgroup paths as arguments precisely so these
tests can point them at fixture files; nothing here reads the host's real
``/proc/self/status`` except the two probes that assert the live path still works.
"""

from __future__ import annotations

import logging

import pytest
from upath import UPath

from drevalpy.visualization import _progress

_STATUS = """Name:\tpython3
VmPeak:\t 8388608 kB
VmSize:\t 8388608 kB
VmHWM:\t 3145728 kB
VmRSS:\t 2097152 kB
Threads:\t1
"""


@pytest.fixture()
def status_file(tmp_path: UPath) -> str:
    """A ``/proc/<pid>/status`` fixture reporting 2 GiB RSS and a 3 GiB high-water mark."""
    path = UPath(tmp_path) / "status"
    path.write_text(_STATUS)
    return str(path)


def _write(tmp_path: UPath, name: str, text: str) -> str:
    path = UPath(tmp_path) / name
    path.write_text(text)
    return str(path)


class TestReadStatus:
    def test_reads_the_requested_field_in_kib(self, status_file: str) -> None:
        assert _progress._read_status_kib("VmRSS", status_file) == 2097152

    def test_does_not_confuse_vmpeak_with_vmsize(self, status_file: str) -> None:
        assert _progress._read_status_kib("VmSize", status_file) == 8388608

    def test_missing_field_is_none(self, status_file: str) -> None:
        assert _progress._read_status_kib("VmSwap", status_file) is None

    def test_missing_file_is_none(self, tmp_path: UPath) -> None:
        assert _progress._read_status_kib("VmRSS", str(UPath(tmp_path) / "absent")) is None

    def test_unparseable_value_is_none(self, tmp_path: UPath) -> None:
        path = _write(tmp_path, "status", "VmRSS:\tnot-a-number kB\n")

        assert _progress._read_status_kib("VmRSS", path) is None

    def test_field_without_a_value_is_none(self, tmp_path: UPath) -> None:
        path = _write(tmp_path, "status", "VmRSS:\n")

        assert _progress._read_status_kib("VmRSS", path) is None


class TestRss:
    def test_current_rss_comes_from_vmrss(self, status_file: str) -> None:
        assert _progress.rss_gb(status_file) == pytest.approx(2.0)

    def test_peak_rss_comes_from_vmhwm(self, status_file: str) -> None:
        assert _progress.peak_rss_gb(status_file) == pytest.approx(3.0)

    def test_rss_falls_back_to_rusage_without_proc(self, tmp_path: UPath) -> None:
        assert _progress.rss_gb(str(UPath(tmp_path) / "absent")) > 0

    def test_peak_falls_back_to_rusage_without_proc(self, tmp_path: UPath) -> None:
        assert _progress.peak_rss_gb(str(UPath(tmp_path) / "absent")) > 0

    def test_rusage_is_normalised_to_kib_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_progress.sys, "platform", "linux")

        linux = _progress._rusage_max_rss_gb()

        monkeypatch.setattr(_progress.sys, "platform", "darwin")
        assert linux == pytest.approx(_progress._rusage_max_rss_gb() * 1024)

    def test_the_live_proc_path_still_reports_a_positive_rss(self) -> None:
        assert _progress.rss_gb() > 0


class TestMemoryLimit:
    def test_reads_cgroup_v2_bytes(self, tmp_path: UPath) -> None:
        v2 = _write(tmp_path, "memory.max", str(36 * 1024**3))

        assert _progress.memory_limit_gb(v2, str(UPath(tmp_path) / "absent")) == pytest.approx(36.0)

    def test_falls_back_to_cgroup_v1(self, tmp_path: UPath) -> None:
        v1 = _write(tmp_path, "limit_in_bytes", str(8 * 1024**3))

        assert _progress.memory_limit_gb(str(UPath(tmp_path) / "absent"), v1) == pytest.approx(8.0)

    def test_the_literal_max_is_not_a_limit(self, tmp_path: UPath) -> None:
        v2 = _write(tmp_path, "memory.max", "max\n")
        v1 = _write(tmp_path, "limit_in_bytes", str(4 * 1024**3))

        assert _progress.memory_limit_gb(v2, v1) == pytest.approx(4.0)

    def test_an_unbounded_sentinel_is_not_a_limit(self, tmp_path: UPath) -> None:
        v2 = _write(tmp_path, "memory.max", str(2**63 - 1))

        assert _progress.memory_limit_gb(v2, str(UPath(tmp_path) / "absent")) is None

    def test_an_empty_file_is_not_a_limit(self, tmp_path: UPath) -> None:
        v2 = _write(tmp_path, "memory.max", "\n")

        assert _progress.memory_limit_gb(v2, str(UPath(tmp_path) / "absent")) is None

    def test_an_unparseable_value_is_not_a_limit(self, tmp_path: UPath) -> None:
        v2 = _write(tmp_path, "memory.max", "unlimited\n")

        assert _progress.memory_limit_gb(v2, str(UPath(tmp_path) / "absent")) is None

    def test_both_paths_absent_is_none(self, tmp_path: UPath) -> None:
        absent = str(UPath(tmp_path) / "absent")

        assert _progress.memory_limit_gb(absent, absent) is None


class TestFormatStage:
    def test_names_the_stage_and_both_memory_figures(self, status_file: str, tmp_path: UPath) -> None:
        absent = str(UPath(tmp_path) / "absent")

        line = _progress.format_stage("load", status_path=status_file, v2_path=absent, v1_path=absent)

        assert line == "load | rss=2.00 GB peak=3.00 GB"

    def test_reports_the_limit_and_headroom_when_capped(self, status_file: str, tmp_path: UPath) -> None:
        v2 = _write(tmp_path, "memory.max", str(12 * 1024**3))

        line = _progress.format_stage("load", status_path=status_file, v2_path=v2, v1_path=v2)

        assert line == "load | rss=2.00 GB peak=3.00 GB limit=12.00 GB (25%)"

    def test_a_zero_limit_is_not_divided_by(self, status_file: str, tmp_path: UPath) -> None:
        v2 = _write(tmp_path, "memory.max", "0")

        line = _progress.format_stage("load", status_path=status_file, v2_path=v2, v1_path=v2)

        assert "limit=" not in line


class TestLogStage:
    def test_emits_one_info_record_carrying_the_summary(
        self, status_file: str, tmp_path: UPath, caplog: pytest.LogCaptureFixture
    ) -> None:
        absent = str(UPath(tmp_path) / "absent")
        logger = logging.getLogger("drevalpy.tests.progress")

        with caplog.at_level(logging.INFO, logger=logger.name):
            _progress.log_stage(logger, "plot heatmap", status_path=status_file, v2_path=absent, v1_path=absent)

        assert [r.getMessage() for r in caplog.records] == ["plot heatmap | rss=2.00 GB peak=3.00 GB"]

    def test_honours_the_requested_level(
        self, status_file: str, tmp_path: UPath, caplog: pytest.LogCaptureFixture
    ) -> None:
        absent = str(UPath(tmp_path) / "absent")
        logger = logging.getLogger("drevalpy.tests.progress")

        with caplog.at_level(logging.DEBUG, logger=logger.name):
            _progress.log_stage(
                logger,
                "load",
                level=logging.DEBUG,
                status_path=status_file,
                v2_path=absent,
                v1_path=absent,
            )

        assert caplog.records[0].levelno == logging.DEBUG

    def test_uses_lazy_formatting_so_the_message_is_the_argument(
        self, status_file: str, tmp_path: UPath, caplog: pytest.LogCaptureFixture
    ) -> None:
        absent = str(UPath(tmp_path) / "absent")
        logger = logging.getLogger("drevalpy.tests.progress")

        with caplog.at_level(logging.INFO, logger=logger.name):
            _progress.log_stage(logger, "load", status_path=status_file, v2_path=absent, v1_path=absent)

        assert caplog.records[0].msg == "%s"
