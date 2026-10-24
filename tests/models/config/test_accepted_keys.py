"""Tests for the accepted-key helpers shared by key validation and the tuning modules."""

from __future__ import annotations

import pytest

from drevalpy.errors import UnknownNameError
from drevalpy.models.config import SampleFeaturizerConfig
from drevalpy.models.config._accepted_keys import featurizer_accepted_keys, leaf_selector, predictor_accepted_keys


def _predictor(defaults: dict, space: dict, **extra):
    return type(
        "Predictor",
        (),
        {
            "get_default_hyperparameters": staticmethod(lambda: defaults),
            "get_hyperparameter_space": staticmethod(lambda: space),
            **extra,
        },
    )


@pytest.mark.parametrize(
    ("predictor", "expected"),
    [
        (_predictor({"alpha": 1.0}, {"l1_ratio": {"default": 0.5}}), {"alpha", "l1_ratio"}),
        (_predictor({"alpha": 1.0}, {}, non_tunable_hyperparameters={"max_iter": 1000}), {"alpha", "max_iter"}),
        (_predictor({}, {}), set()),
    ],
    ids=["defaults-and-space", "non-tunable", "empty"],
)
def test_predictor_accepted_keys(predictor, expected) -> None:
    assert predictor_accepted_keys(predictor) == expected


def test_featurizer_accepted_keys_read_the_registered_class_space() -> None:
    config = SampleFeaturizerConfig(name="pca", view="gene_expression")

    assert "n_components" in featurizer_accepted_keys(config, "sample")
    with pytest.raises(UnknownNameError):
        featurizer_accepted_keys(config, "tissue")  # type: ignore[arg-type]


def test_leaf_selector_qualifies_only_view_parametric_featurizers() -> None:
    assert leaf_selector(SampleFeaturizerConfig(name="pca", view="methylation")) == "pca[methylation]"
    assert leaf_selector(SampleFeaturizerConfig(name="identity")) == "identity"
