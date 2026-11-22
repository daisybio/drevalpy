"""A featurizer config is checked against the constructor of the registered class when it is built.

The built-in featurizers are used as they are: what they accept is read from their signatures and
``input_views``, which is what the checks under test do.
"""

from __future__ import annotations

import pytest

from drevalpy.models.config import FeaturizerConfig, from_dict, from_recipe


def _parse(raw: object, side: str) -> FeaturizerConfig:
    return FeaturizerConfig.parse(raw, side)


@pytest.mark.parametrize(
    ("recipe", "match"),
    [
        (
            "scaledGeneExpression[mutations]:fingerprints:randomForest",
            "sample featurizer 'scaledGeneExpression' cannot read the view 'mutations'; valid views: gene_expression",
        ),
        (
            "scaledGeneExpression[mutations]+view[gene_expression]:fingerprints:randomForest",
            "sample featurizer 'scaledGeneExpression' cannot read the view 'mutations'; valid views: gene_expression",
        ),
        (
            "scaledGeneExpression:fingerprints[mutations]:randomForest",
            "drug featurizer 'fingerprints' cannot read the view 'mutations'; valid views: morgan_fingerprint",
        ),
        (
            "landmarkGenes[morgan_fingerprint]:fingerprints:randomForest",
            "sample featurizer 'landmarkGenes' cannot read the view 'morgan_fingerprint'; valid views: gene_expression",
        ),
        ("oneHot[foo]:fingerprints:randomForest", "sample featurizer 'oneHot' takes no view, but 'foo' was given"),
        ("scaledGeneExpression:oneHot[foo]:randomForest", "drug featurizer 'oneHot' takes no view"),
        (
            "molirOmics[mutations]:fingerprints:randomForest",
            "sample featurizer 'molirOmics' takes no view, but 'mutations' was given; remove it. "
            "Its options: gene_lists, n_gene_expression_features",
        ),
        ("constant[x]:fingerprints:randomForest", "sample featurizer 'constant' takes no view"),
    ],
    ids=[
        "sample-view-not-an-input",
        "inside-a-concat",
        "drug-featurizer-with-a-sample-view",
        "sample-featurizer-with-a-drug-view",
        "id-only-sample",
        "id-only-drug",
        "several-input-views",
        "no-constructor",
    ],
)
def test_a_view_the_featurizer_cannot_read_fails_when_the_recipe_is_read(recipe: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        from_recipe(recipe)


def test_the_error_of_a_view_a_featurizer_cannot_read_points_to_the_view_featurizer() -> None:
    with pytest.raises(ValueError, match="use the 'view' featurizer"):
        from_recipe("scaledGeneExpression[mutations]:fingerprints:randomForest")


@pytest.mark.parametrize(
    ("raw", "side"),
    [
        ("scaledGeneExpression[gene_expression]", "sample"),
        ("fingerprints[morgan_fingerprint]", "drug"),
        ("view[anything_the_dataset_has]", "sample"),
        ("view[anything_the_dataset_has]", "drug"),
        ("pca[methylation]", "sample"),
        ("view[mutations]+pca[methylation]", "sample"),
        ("landmarkGenesReduced[gene_expression]", "sample"),
    ],
)
def test_a_view_the_featurizer_reads_is_accepted(raw: str, side: str) -> None:
    assert _parse(raw, side).views()


@pytest.mark.parametrize("side", ["sample", "drug"])
def test_concatenation_takes_no_view(side: str) -> None:
    raw = {"name": "concatFeaturizers", "view": "mutations", "featurizers": ["view[a]"]}
    with pytest.raises(ValueError, match="concatFeaturizers' takes no view"):
        _parse(raw, side)


@pytest.mark.parametrize(
    ("raw", "side", "match"),
    [
        (
            {"name": "scaledGeneExpression", "options": {"gene_lst": "landmark_genes"}},
            "sample",
            "options of sample featurizer 'scaledGeneExpression' name unknown keys (gene_lst); "
            "valid keys: gene_list, transform",
        ),
        (
            {"name": "fingerprints", "options": {"n_bit": 512}},
            "drug",
            "options of drug featurizer 'fingerprints' name unknown keys (n_bit); valid keys: ",
        ),
        ({"name": "oneHot", "options": {"x": 1}}, "drug", "name unknown keys (x); valid keys: none"),
        (
            {"name": "scaledGeneExpression", "options": {"view": "mutations"}},
            "sample",
            "takes 'view' as a key next to 'name', not under 'options'; "
            "write {name: scaledGeneExpression, view: mutations}",
        ),
        (
            {
                "name": "scaledGeneExpression",
                "hyperparameter_space": {
                    "transfrm": {"type": "categorical", "choices": ["arcsinh"], "default": "arcsinh"}
                },
            },
            "sample",
            "hyperparameter_space of sample featurizer 'scaledGeneExpression' name unknown keys (transfrm); "
            "valid keys: gene_list, transform",
        ),
    ],
    ids=["typo", "one-of-several", "no-options-at-all", "view-under-options", "space-key"],
)
def test_an_option_the_constructor_lacks_is_rejected_with_the_valid_keys(raw: dict, side: str, match: str) -> None:
    with pytest.raises(ValueError, match=match.replace("(", r"\(").replace(")", r"\)")):
        _parse(raw, side)


def test_an_unknown_option_fails_through_the_model_mapping_too() -> None:
    payload = {
        "sample_featurizer": {"name": "scaledGeneExpression", "options": {"gene_lst": "landmark_genes"}},
        "drug_featurizer": "fingerprints",
        "predictor": "randomForest",
    }
    with pytest.raises(ValueError, match="sample_featurizer: options of sample featurizer 'scaledGeneExpression'"):
        from_dict(payload)


def test_a_constructor_with_keyword_arguments_accepts_any_option() -> None:
    config = _parse({"name": "landmarkGenesReduced", "options": {"anything": 1}}, "sample")
    assert config.options == {"anything": 1}


def test_options_that_are_constructor_parameters_are_accepted() -> None:
    config = _parse(
        {"name": "scaledGeneExpression", "options": {"gene_list": "landmark_genes", "transform": "rank"}}, "sample"
    )
    assert config.options == {"gene_list": "landmark_genes", "transform": "rank"}
