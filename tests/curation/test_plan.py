"""Tests for drevalpy.curation._plan: the plan, its files and the assembled result."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.curation import _plan_curation
from drevalpy.curation._plan import (
    Fit,
    assemble,
    read_chunk,
    read_fitted,
    read_plan,
    run_chunks,
    write_chunk,
    write_fitted,
    write_plan,
)
from tests.curation.conftest import build_dose_response_df, build_replicated_wells


def _plan(**kwargs):
    return _plan_curation(
        build_replicated_wells(), replicate_fits=True, metadata=("plate_date",), fit_speed="fast", **kwargs
    )


class TestPlan:
    def test_the_replicate_fit_has_its_own_chunks(self) -> None:
        work = _plan(max_chunk_size=1)

        assert {chunk.fit for chunk in work.chunks} == {Fit.POOLED, Fit.REPLICATE}
        assert all(len(chunk.frame) == 1 for chunk in work.chunks)
        assert len({chunk.name for chunk in work.chunks}) == len(work.chunks)

    def test_without_replicate_fits_there_is_only_the_pooled_fit(self) -> None:
        work = _plan_curation(build_dose_response_df(), fit_speed="fast")

        assert set(work.infos) == {Fit.POOLED}
        assert {chunk.fit for chunk in work.chunks} == {Fit.POOLED}

    def test_a_missing_fit_is_named(self) -> None:
        work = _plan()

        with pytest.raises(KeyError, match="pooled-00000"):
            assemble(work, {})


class TestFiles:
    def test_a_plan_round_trips(self, tmp_path: UPath) -> None:
        work = _plan(max_chunk_size=2)

        write_plan(work, tmp_path / "plan")
        read = read_plan(tmp_path / "plan")

        assert read.options == work.options
        assert read.replicate_fits
        assert read.infos == work.infos
        assert [chunk.name for chunk in read.chunks] == [chunk.name for chunk in work.chunks]
        for ours, theirs in zip(work.chunks, read.chunks, strict=True):
            assert (ours.fit, ours.group, ours.info, ours.options) == (
                theirs.fit,
                theirs.group,
                theirs.info,
                theirs.options,
            )
            pd.testing.assert_frame_equal(ours.frame, theirs.frame)
        pd.testing.assert_frame_equal(read.series, work.series.reset_index(drop=True), check_dtype=False)

    def test_equal_chunks_are_equal_files(self, tmp_path: UPath) -> None:
        chunk = _plan().chunks[0]

        write_chunk(chunk, tmp_path / "a.parquet")
        write_chunk(chunk, tmp_path / "b.parquet")

        assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()

    def test_a_parquet_file_that_is_no_chunk_is_rejected(self, tmp_path: UPath) -> None:
        pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "x.parquet")

        with pytest.raises(ValueError, match="not a curation chunk"):
            read_chunk(tmp_path / "x.parquet")

    def test_a_plan_of_another_format_is_rejected(self, tmp_path: UPath) -> None:
        write_plan(_plan(), tmp_path)
        description = json.loads((tmp_path / "plan.json").read_text())
        (tmp_path / "plan.json").write_text(json.dumps({**description, "format": 0}))

        with pytest.raises(ValueError, match="format 0"):
            read_plan(tmp_path)

    @pytest.mark.slow
    def test_fits_written_and_read_back_assemble_to_the_same_result(self, tmp_path: UPath) -> None:
        work = _plan(max_chunk_size=2)
        fitted = run_chunks(work.chunks, max_workers=1)
        for name, frame in fitted.items():
            write_fitted(frame, tmp_path / f"{name}.parquet")
        read = {name: read_fitted(tmp_path / f"{name}.parquet") for name in fitted}

        direct, via_files = assemble(work, fitted), assemble(work, read)

        np.testing.assert_array_equal(direct.X, via_files.X)
        pd.testing.assert_frame_equal(direct.uns["replicate_fits"], via_files.uns["replicate_fits"])
