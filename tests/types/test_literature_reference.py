"""Tests for LiteratureReference."""

from __future__ import annotations

from drevalpy.types.literature_reference import LiteratureReference


def test_literature_reference_strips_fields() -> None:
    ref = LiteratureReference(
        repo_url="  https://github.com/example/repo  ",
        citation_doi=" 10.1234/example ",
        citation_text=" Example paper. ",
        deviations=" none ",
    )
    assert ref.repo_url == "https://github.com/example/repo"
    assert ref.citation_doi == "10.1234/example"
    assert ref.citation_text == "Example paper."
    assert ref.deviations == "none"
