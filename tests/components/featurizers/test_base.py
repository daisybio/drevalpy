"""Tests for the ``Featurizer`` base class contract and view declarations.

Mirrors :mod:`drevalpy.components.featurizers.base`, which is where ``contract``,
``input_views``, ``source_views``, ``requires_view``, ``entity_id_only`` and
``resolve_input_views`` are defined. The registry sweeps below assert those
class-body declarations across every registered featurizer, so they belong to
this module rather than to any single concrete featurizer.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat, featurizer_contract
from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.cell_line_featurizer import (
    get as get_cell_line_featurizer,
)
from drevalpy.registry.cell_line_featurizer import (
    list as list_cell_line_featurizers,
)
from drevalpy.registry.drug_featurizer import (
    get as get_drug_featurizer,
)
from drevalpy.registry.drug_featurizer import (
    list as list_drug_featurizers,
)
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.modalities import resolve_omics_accessor

_PROBE_VIEW = "gene_expression"


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def _featurizer_names(registry: str) -> list[str]:
    register_builtin_components()
    return list_cell_line_featurizers() if registry == "cell_line" else list_drug_featurizers()


def test_featurizer_rejects_class_body_contract() -> None:
    with pytest.raises(TypeError, match="do not set contract on the class body"):

        class BadFeaturizer(Featurizer):  # noqa: B903
            contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


@pytest.mark.parametrize("registry", ["cell_line", "drug"])
def test_every_featurizer_declares_input_views(registry: str) -> None:
    get = get_cell_line_featurizer if registry == "cell_line" else get_drug_featurizer
    names = _featurizer_names(registry)
    assert names
    for name in names:
        cls = get(name)
        if issubclass(cls, ConcatFeaturizersMixin):
            continue
        kwargs = {"view": _PROBE_VIEW} if cls.requires_view else {}
        views = cls.resolve_input_views(**kwargs)
        assert isinstance(views, tuple), name
        assert all(isinstance(view, str) and view.strip() for view in views), name


@pytest.mark.parametrize("registry", ["cell_line", "drug"])
def test_entity_id_only_featurizers_need_no_views(registry: str) -> None:
    get = get_cell_line_featurizer if registry == "cell_line" else get_drug_featurizer
    entity_id_only = [name for name in _featurizer_names(registry) if get(name).entity_id_only]
    assert entity_id_only
    for name in entity_id_only:
        assert get(name).resolve_input_views() == (), name


@pytest.mark.parametrize("registry", ["cell_line", "drug"])
def test_concat_featurizer_refuses_standalone_view_resolution(registry: str) -> None:
    get = get_cell_line_featurizer if registry == "cell_line" else get_drug_featurizer
    cls = get("concatFeaturizers")
    with pytest.raises(TypeError, match="has no input views of its own"):
        cls.resolve_input_views()


@pytest.mark.parametrize("name", ["raw", "pca"])
def test_view_parameterized_featurizers_require_an_explicit_view(name: str) -> None:
    cls = get_cell_line_featurizer(name)
    assert cls.requires_view
    assert cls.resolve_input_views(view="mutations") == ("mutations",)
    with pytest.raises(TypeError, match="requires an explicit view"):
        cls.resolve_input_views()


_CELL_LINE_CONTRACTS = [
    ("landmarkGenes", FeatureFormat.NUMERIC_MATRIX),
    ("landmarkGenesReduced", FeatureFormat.NUMERIC_MATRIX),
    ("pathways", FeatureFormat.NUMERIC_MATRIX),
    ("bionic", FeatureFormat.NUMERIC_MATRIX),
    ("dipkGeneExpression", FeatureFormat.NUMERIC_MATRIX),
    ("pharmaFormerGeneExpression", FeatureFormat.NUMERIC_MATRIX),
    ("sparsegoOntology", FeatureFormat.NUMERIC_MATRIX),
    ("molirOmics", FeatureFormat.NUMERIC_MATRIX),
    ("superfeltrOmics", FeatureFormat.NUMERIC_MATRIX),
    ("concatFeaturizers", FeatureFormat.NUMERIC_MATRIX),
    ("raw", FeatureFormat.NUMERIC_MATRIX),
    ("pca", FeatureFormat.NUMERIC_MATRIX),
]

_DRUG_CONTRACTS = [
    ("molgnet", FeatureFormat.RAGGED_SEQUENCE),
    ("bpePharmaformer", FeatureFormat.NUMERIC_MATRIX),
    ("smilesvec", FeatureFormat.NUMERIC_MATRIX),
    ("drugGraph", FeatureFormat.GRAPH),
]


@pytest.mark.parametrize(("name", "expected_format"), _CELL_LINE_CONTRACTS)
def test_cell_line_literature_featurizer_contracts(
    name: str,
    expected_format: FeatureFormat,
) -> None:
    cls = get_cell_line_featurizer(name)
    contract = featurizer_contract(cls)
    assert isinstance(contract, FeatureContract)
    assert contract.format == expected_format


@pytest.mark.parametrize(("name", "expected_format"), _DRUG_CONTRACTS)
def test_drug_literature_featurizer_contracts(
    name: str,
    expected_format: FeatureFormat,
) -> None:
    cls = get_drug_featurizer(name)
    contract = featurizer_contract(cls)
    assert isinstance(contract, FeatureContract)
    assert contract.format == expected_format


#: Featurizers whose declared source views the synthetic fixture deliberately
#: does not carry, mapped to why. ``bionic`` reaches for an S3 artifact CI cannot
#: download, and ``sparsegoOntology`` declares no source views at all.
_NO_FIXTURE_SOURCE = {"bionic", "sparsegoOntology"}


@pytest.mark.parametrize(
    ("name", "getter"),
    [(name, get_cell_line_featurizer) for name, _ in _CELL_LINE_CONTRACTS]
    + [(name, get_drug_featurizer) for name, _ in _DRUG_CONTRACTS],
)
def test_literature_featurizer_source_views_exist_in_the_fixture(
    name: str,
    getter: Callable[[str], type],
    synthetic_dataset: Dataset,
) -> None:
    """Every literature featurizer's raw inputs are present in the synthetic fixture.

    Views are resolved through :data:`OMICS_ACCESSORS` before being looked up,
    because the fixture stores omics under the accessor the published datasets
    use. That resolution is exactly what the library's own read sites still do
    not do, which is why ``molirOmics`` and ``superfeltrOmics`` work here yet
    their models are xfailed in the model tests: the data is present, the lookup
    is what asks for the wrong name.

    :param name: Featurizer registry name.
    :param getter: Registry lookup for the featurizer's side.
    :param synthetic_dataset: Session-scoped synthetic raw-omics dataset.
    """
    if name in _NO_FIXTURE_SOURCE:
        pytest.skip(f"{name} declares no source views the fixture can supply")

    cls = getter(name)
    declared = tuple(cls.source_views or ()) or tuple(cls.input_views or ())
    if not declared:
        pytest.skip(f"{name} declares neither source_views nor input_views")

    missing = [
        view
        for view in declared
        if view != "canonical_smiles" and not synthetic_dataset._has_required_views((resolve_omics_accessor(view),))
    ]
    assert not missing, f"{name} reads {missing}, which the synthetic fixture does not provide"
