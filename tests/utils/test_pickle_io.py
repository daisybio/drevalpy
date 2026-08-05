"""Tests for drevalpy.utils.pickle_io."""

from __future__ import annotations

from pathlib import Path

from drevalpy.utils.pickle_io import dump_trusted_pickle, load_trusted_pickle


def test_dump_and_load_trusted_pickle_round_trip(tmp_path: Path) -> None:
    payload = {"fold": 0, "values": [1.0, 2.5]}
    target = tmp_path / "artifact.pkl"
    dump_trusted_pickle(payload, target)
    assert load_trusted_pickle(target) == payload


def test_load_trusted_pickle_from_open_stream(tmp_path: Path) -> None:
    payload = {"ok": True}
    target = tmp_path / "artifact.pkl"
    dump_trusted_pickle(payload, target)
    with target.open("rb") as handle:
        assert load_trusted_pickle(handle) == payload
