"""Registry-wide checks of what every built-in featurizer declares on the ``Featurizer`` interface."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._featurizer_label import requires_explicit_view
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.registry.featurizer import featurizers, get_featurizer, names_for_side
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import FeatureSource

_PROBE_VIEW = "gene_expression"


def _config(name: str) -> dict[str, object]:
    cls = get_featurizer(name)
    config: dict[str, object] = dict(cls.default_hyperparameters())
    if requires_explicit_view(name):
        config["view"] = _PROBE_VIEW
    return config


_LEAVES = [name for name in featurizers.names() if name != "concatFeaturizers"]


def test_every_featurizer_is_on_the_interface_and_declares_its_sides() -> None:
    assert _LEAVES
    for name, cls in featurizers.items():
        assert issubclass(cls, Featurizer), name
        assert cls.name == name
        assert cls.sides and cls.sides <= {"sample", "drug"}, name


def test_shared_featurizers_support_both_sides_and_side_featurizers_one() -> None:
    sample, drug = set(names_for_side("sample")), set(names_for_side("drug"))
    assert sample & drug == {"constant", "identity", "concatFeaturizers"}
    assert {"raw", "pca", "tissue", "scaledGeneExpression"} <= sample - drug
    assert {"fingerprints", "drugGraph", "molgnet", "view"} <= drug - sample


@pytest.mark.parametrize("name", _LEAVES)
def test_every_featurizer_reports_its_views(name: str) -> None:
    views = get_featurizer(name).views(_config(name))
    assert isinstance(views, tuple)
    assert all(isinstance(view, str) and view.strip() for view in views)


@pytest.mark.parametrize("name", ["raw", "pca"])
def test_view_parameterized_featurizers_require_an_explicit_view(name: str) -> None:
    cls = get_featurizer(name)
    assert requires_explicit_view(name)
    assert cls.views({"view": "mutations"}) == ("mutations",)


def test_featurizers_with_input_views_need_no_explicit_view() -> None:
    assert not requires_explicit_view("scaledGeneExpression")
    assert not requires_explicit_view("identity")
    assert not requires_explicit_view("notRegistered")


_FORMATS = [
    ("landmarkGenes", FeatureFormat.NUMERIC_MATRIX),
    ("landmarkGenesReduced", FeatureFormat.NUMERIC_MATRIX),
    ("pathways", FeatureFormat.NUMERIC_MATRIX),
    ("dipkGeneExpression", FeatureFormat.NUMERIC_MATRIX),
    ("pharmaFormerGeneExpression", FeatureFormat.NUMERIC_MATRIX),
    ("sparsegoOntology", FeatureFormat.NUMERIC_MATRIX),
    ("molirOmics", FeatureFormat.NUMERIC_MATRIX),
    ("superfeltrOmics", FeatureFormat.NUMERIC_MATRIX),
    ("raw", FeatureFormat.NUMERIC_MATRIX),
    ("pca", FeatureFormat.NUMERIC_MATRIX),
    ("molgnet", FeatureFormat.RAGGED_SEQUENCE),
    ("bpePharmaformer", FeatureFormat.NUMERIC_MATRIX),
    ("smilesvec", FeatureFormat.NUMERIC_MATRIX),
    ("drugGraph", FeatureFormat.GRAPH),
]


@pytest.mark.parametrize(("name", "expected_format"), _FORMATS)
def test_literature_featurizer_output_formats(name: str, expected_format: FeatureFormat) -> None:
    blocks = get_featurizer(name).output_blocks(_config(name))
    assert blocks
    assert {block.format for block in blocks} == {expected_format}


#: Featurizers whose views the synthetic fixture deliberately does not carry: ``bionic``
#: reaches for an S3 artifact CI cannot download.
_NO_FIXTURE_SOURCE = {"bionic"}


@pytest.mark.parametrize("name", [name for name, _ in _FORMATS])
def test_literature_featurizer_views_exist_in_the_fixture(name: str, synthetic_dataset: Dataset) -> None:
    if name in _NO_FIXTURE_SOURCE:
        pytest.skip(f"{name} reads views the fixture cannot supply")
    cls = get_featurizer(name)
    side = "sample" if "sample" in cls.sides else "drug"
    source = FeatureSource(synthetic_dataset, side, np.asarray([]))
    views = cls.views(_config(name))
    if not views:
        pytest.skip(f"{name} reads no view")
    present = all(source.has_view(view) for view in views)
    from_source = bool(cls.source_views) and all(source.has_view(view) for view in cls.source_views)
    assert present or from_source, f"{name} reads {views} (source views {cls.source_views}), absent from the fixture"
