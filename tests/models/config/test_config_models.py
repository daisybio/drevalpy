"""Config objects: featurizer, predictor and model configs, their immutability, derived views and scope."""

from __future__ import annotations

from types import MappingProxyType

import pytest
from pydantic import ValidationError

from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig, from_recipe
from drevalpy.models.zoo import get_zoo_config
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode

SAMPLE = FeaturizerConfig(name="scaledGeneExpression", side="sample")


def test_view_validation() -> None:
    with pytest.raises(ValidationError, match="view must be a non-empty string when set"):
        FeaturizerConfig(name="denseSample", side="sample", view="   ")
    assert FeaturizerConfig(name="denseSample", side="sample", view="gene_expression").view == "gene_expression"
    with pytest.raises(ValidationError):
        FeaturizerConfig(name="landmarkGenes", side="sample", views=["gene_expression"])  # type: ignore[call-arg]


def test_defaults_move_the_space_default() -> None:
    config = FeaturizerConfig.parse({"name": "pca", "view": "methylation", "defaults": {"n_components": 42}}, "sample")
    assert (config.name, config.view) == ("pca", "methylation")
    assert config.hyperparameter_space is not None
    assert config.hyperparameter_space["n_components"]["default"] == 42


def test_featurizer_config_compact_forms() -> None:
    config = FeaturizerConfig.parse("fingerprints", "drug")
    assert config.name == "fingerprints"


def test_featurizer_config_has_no_registry_field() -> None:
    with pytest.raises(ValidationError):
        FeaturizerConfig.parse({"name": "scaledGeneExpression", "registry": "drug"}, "sample")


@pytest.mark.parametrize(
    ("drug", "predictor", "model_id"),
    [
        (FeaturizerConfig(name="oneHot", side="drug"), "randomForest", "scaledGeneExpression:oneHot:randomForest"),
        (None, "singleDrugElasticNet", "scaledGeneExpression:singleDrugElasticNet"),
    ],
    ids=["full-triple", "single-drug-without-drug-featurizer"],
)
def test_model_id(drug, predictor, model_id) -> None:
    config = ModelConfig(sample_featurizer=SAMPLE, drug_featurizer=drug, predictor=PredictorConfig(name=predictor))

    assert config.model_id == model_id
    assert config.drug_featurizer == drug


def test_model_id_for_predictor_only_baseline() -> None:
    config = ModelConfig(sample_featurizer=None, drug_featurizer=None, predictor=PredictorConfig(name="naiveMean"))

    assert config.model_id == "naiveMean"


@pytest.mark.parametrize(
    "recipe",
    [
        "view[gene_expression]+view[mutations]:fingerprints:randomForest",
        "pca[gene_expression]:fingerprints:elasticNet",
        "oneHot+tissue:oneHot:naiveMeanEffects",
    ],
)
def test_model_id_reads_back_into_the_same_components(recipe: str) -> None:
    config = from_recipe(recipe)

    assert config.model_id == recipe
    assert from_recipe(config.model_id) == config


def test_model_id_of_every_zoo_preset_reads_back_into_its_featurizer_trees() -> None:
    from drevalpy.models.zoo import list_zoo_names

    for name in list_zoo_names():
        config = get_zoo_config(name)
        recipe = from_recipe(config.model_id)
        for slot in ("sample_featurizer", "drug_featurizer"):
            original, read = getattr(config, slot), getattr(recipe, slot)
            assert (original and [leaf.selector for leaf in original.leaves()]) == (
                read and [leaf.selector for leaf in read.leaves()]
            ), name


def test_model_id_of_a_drug_only_stack_is_none() -> None:
    config = ModelConfig.model_construct(
        sample_featurizer=None,
        drug_featurizer=FeaturizerConfig(name="fingerprints", side="drug"),
        predictor=PredictorConfig(name="elasticNet"),
    )

    assert config.model_id is None


@pytest.mark.parametrize(
    ("drug", "predictor", "match"),
    [
        (None, "randomForest", "Predictor 'randomForest' requires a drug_featurizer"),
        (None, "elasticNet", "Predictor 'elasticNet' requires a drug_featurizer"),
    ],
    ids=["random-forest", "elastic-net"],
)
def test_inconsistent_featurizer_slots_are_rejected(drug, predictor, match) -> None:
    with pytest.raises(ValidationError, match=match):
        ModelConfig(sample_featurizer=SAMPLE, drug_featurizer=drug, predictor=PredictorConfig(name=predictor))


def test_scope_is_derived_from_the_predictor_and_never_stated() -> None:
    payload = {"sample_featurizer": "scaledGeneExpression", "predictor": "singleDrugElasticNet"}

    config = ModelConfig.model_validate(payload)

    assert config.scope is ModelScope.SINGLE_DRUG
    assert "scope" not in config.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelConfig.model_validate({**payload, "scope": "single_drug"})


def test_model_config_parses_compact_sections() -> None:
    config = ModelConfig.model_validate(
        {
            "sample_featurizer": {
                "name": "concatFeaturizers",
                "featurizers": [
                    "scaledGeneExpression",
                    {"name": "pca", "view": "methylation", "defaults": {"n_components": 100}},
                    "view[mutations]",
                ],
            },
            "drug_featurizer": "fingerprints",
            "predictor": {"name": "randomForest", "defaults": {"n_estimators": 60}},
        }
    )

    assert isinstance(config.sample_featurizer, FeaturizerConfig)
    assert config.sample_featurizer.name == "concatFeaturizers"
    assert isinstance(config.drug_featurizer, FeaturizerConfig)
    assert config.drug_featurizer.name == "fingerprints"
    assert config.predictor.name == "randomForest"
    assert config.predictor.hyperparameter_space is not None
    assert config.predictor.hyperparameter_space["n_estimators"]["default"] == 60


def test_model_config_rejects_a_featurizer_in_a_slot_of_a_side_it_does_not_support() -> None:
    with pytest.raises(
        ValidationError, match="'fingerprints' is a drug featurizer and cannot be used on the sample side"
    ):
        ModelConfig.model_validate(
            {
                "sample_featurizer": "fingerprints",
                "drug_featurizer": FeaturizerConfig(name="fingerprints", side="drug"),
                "predictor": PredictorConfig(name="elasticNet"),
            }
        )


def test_config_is_serializable() -> None:
    config = ModelConfig(
        sample_featurizer=SAMPLE,
        drug_featurizer=FeaturizerConfig(name="fingerprints", side="drug"),
        predictor=PredictorConfig(
            name="elasticNet",
            hyperparameter_space={"alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0}},
        ),
        prediction_mode=PredictionMode.REGRESSION,
    )
    payload = config.model_dump(mode="python")
    assert payload["sample_featurizer"]["name"] == "scaledGeneExpression"
    assert payload["predictor"]["hyperparameter_space"]["alpha"]["default"] == 1.0


def _template() -> ModelConfig:
    return ModelConfig(
        sample_featurizer=FeaturizerConfig(name="scaledGeneExpression", side="sample"),
        drug_featurizer=FeaturizerConfig(name="fingerprints", side="drug"),
        predictor=PredictorConfig(name="elasticNet"),
    )


def test_model_config_is_frozen_and_scope_is_read_only() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        _template().drug_featurizer = None
    assert isinstance(ModelConfig.scope, property)
    assert ModelConfig.scope.fset is None


def test_concat_children_are_tuple_not_list() -> None:
    config = FeaturizerConfig.parse("scaledGeneExpression+pca[methylation]", "sample")

    assert config.name == "concatFeaturizers"
    assert isinstance(config.featurizers, tuple)
    assert config.featurizers[1].name == "pca"
    assert isinstance(config.model_dump(mode="json")["featurizers"], list)
    with pytest.raises(TypeError):
        config.featurizers[0] = FeaturizerConfig(name="view", side="sample", view="expression")  # type: ignore[index]


def test_predictor_defaults_write_hyperparameter_space_defaults() -> None:
    config = ModelConfig.model_validate(
        {
            "sample_featurizer": "scaledGeneExpression",
            "drug_featurizer": "fingerprints",
            "predictor": {"name": "randomForest", "defaults": {"n_estimators": 60}},
        }
    )
    space = config.predictor.hyperparameter_space
    assert space is not None
    assert space["n_estimators"]["default"] == 60
    with pytest.raises(AttributeError):
        _ = config.predictor.hyperparameters  # type: ignore[attr-defined]


def test_a_shared_zoo_config_is_deeply_frozen() -> None:
    first = get_zoo_config("MultiViewLightGBM")
    assert first is get_zoo_config("MultiViewLightGBM")
    assert first.sample_featurizer is not None
    assert first.sample_featurizer.featurizers is not None
    child = first.sample_featurizer.featurizers[1]
    assert child.options is not None
    with pytest.raises(TypeError):
        child.options["gene_list"] = "landmark_genes"  # type: ignore[index]


def test_mapping_fields_are_deeply_frozen_views() -> None:
    featurizer = FeaturizerConfig.parse(
        {"name": "tissue", "options": {"allow_missing": True, "nested": {"inner": [1, {"deep": 2}]}}}, "sample"
    )
    predictor = PredictorConfig.model_validate(
        {"name": "randomForest", "hyperparameter_space": {"n_estimators": {"default": 5, "options": [1, 2]}}}
    )

    assert featurizer.options is not None
    assert featurizer.hyperparameter_space is None
    assert predictor.hyperparameter_space is not None
    for mapping in (
        featurizer.options,
        featurizer.options["nested"],
        predictor.hyperparameter_space,
        predictor.hyperparameter_space["n_estimators"],
    ):
        assert isinstance(mapping, MappingProxyType)
        with pytest.raises(TypeError):
            mapping["injected"] = 1  # type: ignore[index]
    assert featurizer.options["nested"]["inner"] == (1, MappingProxyType({"deep": 2}))
    assert predictor.hyperparameter_space["n_estimators"]["options"] == (1, 2)


def test_frozen_mappings_dump_plain_containers() -> None:
    featurizer = FeaturizerConfig.parse(
        {"name": "tissue", "options": {"nested": {"inner": [1, {"deep": 2}]}}}, "sample"
    )

    dumped = featurizer.model_dump(mode="python")

    assert dumped["options"] == {"nested": {"inner": [1, {"deep": 2}]}}
    assert isinstance(dumped["options"], dict)
    assert isinstance(dumped["options"]["nested"]["inner"], list)
    assert isinstance(dumped["options"]["nested"]["inner"][1], dict)
    assert featurizer.model_dump(mode="json")["options"] == dumped["options"]
    assert PredictorConfig(name="elasticNet").model_dump(mode="python")["hyperparameter_space"] is None


class TestFields:
    def test_defaults_and_coercion(self) -> None:
        assert PredictorConfig(name="elasticNet").hyperparameter_space is None
        assert PredictorConfig.model_validate({"name": 5, "hyperparameter_space": None}).name == "5"

    def test_name_is_required_and_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            PredictorConfig()  # type: ignore[call-arg]
        with pytest.raises(ValidationError, match="[Ee]xtra"):
            PredictorConfig(name="elasticNet", not_a_field=1)  # type: ignore[call-arg]


class TestRecipeNormalization:
    def test_accepted_notations(self) -> None:
        assert PredictorConfig.model_validate("elasticNet").name == "elasticNet"

        space = {"n_estimators": {"type": "int", "low": 2, "high": 20, "default": 4}}
        canonical = PredictorConfig.model_validate({"name": "randomForest", "hyperparameter_space": space})
        assert canonical.hyperparameter_space is not None
        assert canonical.hyperparameter_space["n_estimators"]["default"] == 4

    def test_rejects_a_non_tunable_default(self) -> None:
        with pytest.raises(ValidationError, match="'not_a_knob' is not a tunable hyperparameter"):
            PredictorConfig.model_validate({"name": "randomForest", "defaults": {"not_a_knob": 1}})


class TestHyperparameterSpaceValidation:
    @pytest.mark.parametrize(
        ("space", "match"),
        [
            ({"n_estimators": {"low": 1, "high": 5}}, "default"),
            ({"n_estimators": 5}, "n_estimators"),
            ({"n_estimators": {"low": 1}}, r"PredictorConfig\('randomForest'\).hyperparameter_space"),
        ],
        ids=["no-default", "non-mapping-entry", "error-names-the-config"],
    )
    def test_rejects_invalid_entries(self, space, match) -> None:
        with pytest.raises(ValidationError, match=match):
            PredictorConfig.model_validate({"name": "randomForest", "hyperparameter_space": space})

    def test_accepts_default_entries_and_an_empty_space(self) -> None:
        config = PredictorConfig.model_validate(
            {"name": "randomForest", "hyperparameter_space": {"n_estimators": {"default": 5}}}
        )
        assert config.hyperparameter_space is not None
        assert (
            PredictorConfig.model_validate({"name": "elasticNet", "hyperparameter_space": {}}).hyperparameter_space
            == {}
        )


class TestImmutability:
    def test_frozen_hashable_and_deeply_read_only(self) -> None:
        config = PredictorConfig.model_validate({"name": "randomForest", "hyperparameter_space": {"a": {"default": 1}}})

        with pytest.raises(ValidationError, match="frozen"):
            config.name = "other"
        assert hash(PredictorConfig(name="elasticNet")) == hash(PredictorConfig(name="elasticNet"))
        assert config.hyperparameter_space is not None
        with pytest.raises(TypeError):
            config.hyperparameter_space["b"] = {"default": 2}  # type: ignore[index]
        with pytest.raises(TypeError):
            config.hyperparameter_space["a"]["default"] = 2  # type: ignore[index]

        dumped = config.model_dump(mode="json")
        assert dumped == {"name": "randomForest", "hyperparameter_space": {"a": {"default": 1}}}
        assert isinstance(dumped["hyperparameter_space"], dict)


class TestCreateInstance:
    def test_builds_the_registered_class(self) -> None:
        instance = PredictorConfig(name="elasticNet").create_instance({"alpha": 0.1, "l1_ratio": 0.5})

        assert isinstance(instance, get_predictor("elasticNet"))
        assert PredictorConfig(name="naiveMeanEffects").create_instance() is not None

    def test_unknown_name_raises_at_instantiation_time(self) -> None:
        config = PredictorConfig.model_construct(name="notAPredictor", hyperparameter_space=None)

        with pytest.raises(ValueError, match="notAPredictor"):
            config.create_instance()


@pytest.mark.parametrize(
    ("name", "expected"),
    [("elasticNet", ModelScope.MULTI_DRUG), ("singleDrugElasticNet", ModelScope.SINGLE_DRUG)],
)
def test_scope_reads_the_class_declaration(name: str, expected: ModelScope) -> None:
    drug = None if expected is ModelScope.SINGLE_DRUG else FeaturizerConfig(name="fingerprints", side="drug")
    config = ModelConfig(sample_featurizer=SAMPLE, drug_featurizer=drug, predictor=PredictorConfig(name=name))

    assert config.scope is expected


def _model_config(**kwargs: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "predictor": PredictorConfig(name="randomForest"),
        "sample_featurizer": FeaturizerConfig(name="scaledGeneExpression", side="sample"),
        "drug_featurizer": FeaturizerConfig(name="fingerprints", side="drug"),
    }
    defaults.update(kwargs)
    return ModelConfig.model_validate(defaults)


@pytest.mark.parametrize("name", ["oneHot", "constant"])
def test_id_only_featurizers_resolve_to_empty_views(name: str) -> None:
    config = _model_config(
        sample_featurizer=FeaturizerConfig.parse(name, "sample"),
        drug_featurizer=FeaturizerConfig.parse(name, "drug"),
    )
    assert config.sample_views() == []
    assert config.drug_views() == []


@pytest.mark.parametrize(
    ("featurizer", "views"),
    [
        (FeaturizerConfig.parse("view[mutations]+pca[methylation]", "sample"), ["mutations", "methylation"]),
        (FeaturizerConfig.parse("landmarkGenes", "sample"), ["gene_expression"]),
        (FeaturizerConfig.parse("landmarkGenesReduced", "sample"), ["gene_expression"]),
        (
            FeaturizerConfig.parse("molirOmics", "sample"),
            ["gene_expression", "mutations", "copy_number_variation_gistic"],
        ),
        (
            FeaturizerConfig.parse("superfeltrOmics", "sample"),
            ["gene_expression", "mutations", "copy_number_variation_gistic"],
        ),
        (
            FeaturizerConfig.parse({"name": "sparsegoOntology", "defaults": {"input_type": "expression"}}, "sample"),
            ["gene_expression"],
        ),
        (
            FeaturizerConfig.parse({"name": "sparsegoOntology", "defaults": {"input_type": "mutations"}}, "sample"),
            ["mutations"],
        ),
        (FeaturizerConfig.parse("tissue", "sample"), []),
        (FeaturizerConfig(name="scaledGeneExpression", side="sample", options={"view": "proteomics"}), ["proteomics"]),
    ],
    ids=[
        "brackets",
        "landmark",
        "landmark-reduced",
        "molir",
        "superfeltr",
        "sparsego-expression",
        "sparsego-mutations",
        "tissue",
        "view-override",
    ],
)
def test_sample_views(featurizer, views) -> None:
    assert _model_config(sample_featurizer=featurizer).sample_views() == views


def test_drug_views() -> None:
    config = _model_config()
    assert config.drug_views() == ["morgan_fingerprint"]

    view = _model_config(
        drug_featurizer=FeaturizerConfig(name="view", side="drug", options={"view": "drug_chemberta_embeddings"})
    )
    assert view.drug_views() == ["drug_chemberta_embeddings"]


def test_feature_based_predictor_requires_a_drug_featurizer() -> None:
    with pytest.raises(ValidationError, match="requires a drug_featurizer"):
        _model_config(predictor=PredictorConfig(name="naiveSampleMean"), drug_featurizer=None)
