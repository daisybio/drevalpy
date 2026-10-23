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

DOIS = (
    pytest.param(DRUGGNN_REFERENCE, "10.1109/TCBB.2021.3060430", id="druggnn"),
    pytest.param(PRECILY_REFERENCE, "10.1038/s41467-022-33291-z", id="precily"),
    pytest.param(SRMF_REFERENCE, "10.1186/s12885-017-3500-5", id="srmf"),
    pytest.param(MOLIR_REFERENCE, "10.1186/s12859-023-05166-7", id="molir"),
    pytest.param(SUPERFELTR_REFERENCE, "10.1186/s12859-023-05166-7", id="superfeltr"),
    pytest.param(PHARMAFORMER_REFERENCE, "10.1038/s41698-025-01082-6", id="pharmaformer"),
    pytest.param(DIPK_REFERENCE, "10.1093/bib/bbae153", id="dipk"),
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


def test_every_reference_records_a_concrete_model_specific_deviation() -> None:
    deviations = [param.values[0].deviations for param in ALL_REFERENCES]

    assert all(len(note.split()) >= 12 for note in deviations)
    assert len(set(deviations)) == len(deviations)
    assert all("may differ" not in note for note in deviations)


@pytest.mark.parametrize(("reference", "doi"), DOIS)
def test_every_reference_records_its_published_doi(reference: LiteratureReference, doi: str) -> None:
    assert reference.citation_doi == doi


@pytest.mark.parametrize(
    ("reference", "cited"),
    [
        pytest.param(MOLIR_REFERENCE, ("MOLI", "10.1093/bioinformatics/btz318"), id="molir"),
        pytest.param(SUPERFELTR_REFERENCE, ("Super.FELT", "10.1186/s12859-021-04146-z"), id="superfeltr"),
    ],
)
def test_benchmark_doi_references_name_the_original_model_in_the_citation(
    reference: LiteratureReference, cited: tuple[str, str]
) -> None:
    assert all(part in reference.citation_text for part in cited)


def test_repo_urls_are_unique_across_references() -> None:
    urls = [param.values[0].repo_url for param in ALL_REFERENCES]

    assert len(set(urls)) == len(urls)


def test_every_reference_is_immutable() -> None:
    with pytest.raises(AttributeError):
        DIPK_REFERENCE.repo_url = "https://example.invalid"
