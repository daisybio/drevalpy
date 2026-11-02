"""Pins the accelerator level and the licenses every built-in component declares.

The license values were read from each upstream repository's license file and each artifact's
source. Changing one here should come with a source for the new value.
"""

from __future__ import annotations

import pytest

from drevalpy.registry._component import component_license
from drevalpy.registry.featurizer import featurizer_registries
from drevalpy.registry.predictor import predictor_registry
from drevalpy.types.enums.accelerator import Accelerator

_OPTIONAL = {
    ("predictor", "dipk"),
    ("predictor", "drugGNN"),
    ("predictor", "ensembleMF"),
    ("predictor", "molir"),
    ("predictor", "neuralNetwork"),
    ("predictor", "paccMann"),
    ("predictor", "pharmaFormer"),
    ("predictor", "precily"),
    ("predictor", "sparsego"),
    ("predictor", "superfeltr"),
    ("sample", "aeGeneExpression"),
    ("sample", "dipkGeneExpression"),
}

_CODE_LICENSES = {
    ("predictor", "dipk"): "NOASSERTION",
    ("predictor", "drugGNN"): "Apache-2.0",
    ("predictor", "ensembleMF"): "MIT",
    ("predictor", "molir"): "GPL-3.0-only",
    ("predictor", "paccMann"): "MIT",
    ("predictor", "pharmaFormer"): "Apache-2.0",
    ("predictor", "precily"): "NOASSERTION",
    ("predictor", "sparsego"): "NOASSERTION",
    ("predictor", "srmf"): "NOASSERTION",
    ("predictor", "superfeltr"): "NOASSERTION",
    ("drug", "bpePharmaformer"): "Apache-2.0",
}

_ARTIFACT_LICENSES = {
    ("drug", "chemberta"): "MIT",
    ("drug", "molgnet"): "NOASSERTION",
    ("drug", "molgnetPooled"): "NOASSERTION",
    ("drug", "smilesvec"): "MIT",
    ("sample", "bionic"): "NOASSERTION",
    ("sample", "pathways"): "CC-BY-4.0 AND LicenseRef-KEGG AND LicenseRef-BioCarta",
}


def _builtins() -> dict[tuple[str, str], type]:
    components = {("predictor", name): predictor_registry.get(name) for name in predictor_registry.names()}
    for side, registry in featurizer_registries.items():
        components.update({(str(side), name): registry.get(name) for name in registry.names()})
    return components


_BUILTINS = _builtins()


@pytest.mark.parametrize("key", sorted(_BUILTINS), ids="/".join)
def test_builtin_accelerator_level(key: tuple[str, str]) -> None:
    expected = Accelerator.OPTIONAL if key in _OPTIONAL else Accelerator.NONE

    assert _BUILTINS[key].accelerator is expected


@pytest.mark.parametrize("key", sorted(_BUILTINS), ids="/".join)
def test_builtin_licenses(key: tuple[str, str]) -> None:
    cls = _BUILTINS[key]

    assert component_license(cls) == _CODE_LICENSES.get(key, "MIT")
    assert cls.artifact_license == _ARTIFACT_LICENSES.get(key)


def test_every_pinned_component_is_a_builtin() -> None:
    assert set(_OPTIONAL) | set(_CODE_LICENSES) | set(_ARTIFACT_LICENSES) <= set(_BUILTINS)


def test_every_builtin_with_a_reference_declares_its_license() -> None:
    assert {key for key, cls in _BUILTINS.items() if cls.reference is not None} == set(_CODE_LICENSES)
