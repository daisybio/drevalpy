"""Tests for predictor input declarations."""

from __future__ import annotations

from typing import ClassVar

import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs, predictor_inputs
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.registry.predictor import predictor_registry
from drevalpy.types.data.batch.feature_block import BlockSpec

_NUMERIC = FeatureFormat.NUMERIC_MATRIX


def _blocks(*names: str) -> Inputs:
    return Inputs.blocks(*(BlockSpec(name, _NUMERIC) for name in names))


_NONE = Inputs.none()
_MATRIX = Inputs.matrix()
_OMICS = _blocks("gene_expression", "mutations", "copy_number_variation_gistic")

#: ``(sample_inputs, drug_inputs)`` of every built-in predictor.
EXPECTED: dict[str, tuple[Inputs, Inputs]] = {
    "naiveMean": (_NONE, _NONE),
    "naiveDrugMean": (Inputs.blocks(), Inputs.blocks()),
    "naiveSampleMean": (Inputs.blocks(), Inputs.blocks()),
    "naiveMeanEffects": (_blocks("oneHot"), _blocks("oneHot")),
    "naiveTissueMean": (_blocks("tissue"), Inputs.blocks()),
    "naiveTissueDrugMean": (_blocks("tissue"), _blocks("oneHot")),
    **{
        name: (_MATRIX, _MATRIX)
        for name in (
            "elasticNet",
            "lasso",
            "ridge",
            "randomForest",
            "svr",
            "gradientBoosting",
            "adaboost",
            "knn",
            "lightgbm",
            "xgboost",
            "neuralNetwork",
        )
    },
    # Single-drug predictors route pairs by ``batch.drug_ids`` and read no drug features.
    "singleDrugElasticNet": (_MATRIX, _NONE),
    "singleDrugRandomForest": (_MATRIX, _NONE),
    "singleDrugKnn": (_MATRIX, _NONE),
    "molir": (_OMICS, _NONE),
    "superfeltr": (_OMICS, _NONE),
    "dipk": (
        _blocks("gene_expression", "bionic_features"),
        Inputs.blocks(BlockSpec("molgnet_features", FeatureFormat.RAGGED_SEQUENCE)),
    ),
    "drugGNN": (_blocks("gene_expression"), Inputs.blocks(BlockSpec("drug_graph", FeatureFormat.GRAPH))),
    "ensembleMF": (_blocks("gene_expression"), _blocks("fingerprints")),
    "srmf": (_blocks("gene_expression"), _blocks("fingerprints")),
    "paccMann": (_blocks("gene_expression"), _blocks("smiles_tokens", "smiles_token_variants")),
    "pharmaFormer": (_blocks("gene_expression"), _blocks("bpe_smiles")),
    "precily": (_blocks("pathways"), _blocks("smilesvec")),
    "sparsego": (
        Inputs.blocks((BlockSpec("gene_expression", _NUMERIC), BlockSpec("mutations", _NUMERIC))),
        _blocks("fingerprints"),
    ),
}


def test_constructors() -> None:
    spec = BlockSpec("gene_expression", _NUMERIC)
    either = (BlockSpec("a", _NUMERIC), BlockSpec("b", _NUMERIC))

    assert Inputs.none() == Inputs("none")
    assert Inputs.matrix() == Inputs("matrix")
    assert Inputs.blocks(spec, either) == Inputs("blocks", (spec, either))
    assert Inputs.blocks().specs == ()


def test_the_expected_table_covers_every_builtin_predictor() -> None:
    builtins = {name for name in predictor_registry.names() if get_predictor(name).__module__.startswith("drevalpy.")}
    assert builtins == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_builtin_predictor_declares_its_inputs(name: str) -> None:
    cls = get_predictor(name)
    sample, drug = EXPECTED[name]

    assert isinstance(cls.sample_inputs, Inputs)
    assert isinstance(cls.drug_inputs, Inputs)
    assert (cls.sample_inputs, cls.drug_inputs) == (sample, drug)
    assert (predictor_inputs(cls, "sample"), predictor_inputs(cls, "drug")) == (sample, drug)


def test_an_undeclared_side_is_a_type_error() -> None:
    class Undeclared(Predictor):
        sample_inputs: ClassVar[Inputs] = Inputs.matrix()

        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return None

    assert predictor_inputs(Undeclared, "sample") == Inputs.matrix()
    with pytest.raises(TypeError, match="Undeclared.drug_inputs must be an Inputs"):
        predictor_inputs(Undeclared, "drug")
