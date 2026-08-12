"""Tests for the frozen literature reference constants.

Mirrors the private module
``drevalpy.components.predictors.literature._metadata`` with the leading
underscore stripped. The module holds data only, so the assertions are
data-shape assertions rather than behaviour.
"""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.literature._metadata import (
    DIPK_REFERENCE,
    DRUGGNN_REFERENCE,
    LITERATURE_INTEGRATION_DEVIATIONS,
    MOLIR_REFERENCE,
    PHARMAFORMER_REFERENCE,
    PRECILY_REFERENCE,
    SPARSEGO_REFERENCE,
    SRMF_REFERENCE,
    SUPERFELTR_REFERENCE,
)
from drevalpy.types.enums.literature_reference import LiteratureReference

ALL_REFERENCES = (
    pytest.param(DRUGGNN_REFERENCE, id="druggnn"),
    pytest.param(PRECILY_REFERENCE, id="precily"),
    pytest.param(SRMF_REFERENCE, id="srmf"),
    pytest.param(MOLIR_REFERENCE, id="molir"),
    pytest.param(SUPERFELTR_REFERENCE, id="superfeltr"),
    pytest.param(PHARMAFORMER_REFERENCE, id="pharmaformer"),
    pytest.param(DIPK_REFERENCE, id="dipk"),
    pytest.param(SPARSEGO_REFERENCE, id="sparsego"),
)

WITH_DOI = (
    pytest.param(MOLIR_REFERENCE, "10.1186/s12859-023-05166-7", id="molir"),
    pytest.param(SUPERFELTR_REFERENCE, "10.1186/s12859-023-05166-7", id="superfeltr"),
    pytest.param(PHARMAFORMER_REFERENCE, "10.1038/s41698-025-01082-6", id="pharmaformer"),
    pytest.param(SPARSEGO_REFERENCE, "10.1016/j.ebiom.2023.104767", id="sparsego"),
)


@pytest.mark.parametrize("reference", ALL_REFERENCES)
def test_every_constant_is_a_literature_reference(reference: LiteratureReference) -> None:
    assert isinstance(reference, LiteratureReference)


@pytest.mark.parametrize("reference", ALL_REFERENCES)
def test_every_reference_points_at_a_github_repository(reference: LiteratureReference) -> None:
    assert reference.repo_url.startswith("https://github.com/")


@pytest.mark.parametrize("reference", ALL_REFERENCES)
def test_every_reference_has_citation_text(reference: LiteratureReference) -> None:
    assert reference.citation_text


@pytest.mark.parametrize("reference", ALL_REFERENCES)
def test_every_reference_shares_the_integration_deviation_note(reference: LiteratureReference) -> None:
    assert reference.deviations == LITERATURE_INTEGRATION_DEVIATIONS.strip()


@pytest.mark.parametrize(("reference", "doi"), WITH_DOI)
def test_references_with_a_published_doi_record_it(reference: LiteratureReference, doi: str) -> None:
    assert reference.citation_doi == doi


@pytest.mark.parametrize(
    "reference",
    [
        pytest.param(DRUGGNN_REFERENCE, id="druggnn"),
        pytest.param(PRECILY_REFERENCE, id="precily"),
        pytest.param(SRMF_REFERENCE, id="srmf"),
        pytest.param(DIPK_REFERENCE, id="dipk"),
    ],
)
def test_code_only_references_leave_the_doi_empty(reference: LiteratureReference) -> None:
    assert reference.citation_doi == ""


def test_repo_urls_are_unique_across_references() -> None:
    urls = [param.values[0].repo_url for param in ALL_REFERENCES]

    assert len(set(urls)) == len(urls)


def test_every_reference_is_immutable() -> None:
    with pytest.raises(AttributeError):
        DIPK_REFERENCE.repo_url = "https://example.invalid"
