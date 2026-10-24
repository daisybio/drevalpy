"""Config objects: featurizer, predictor and model configs, their immutability, derived views and scope."""

from __future__ import annotations

from types import MappingProxyType

import pytest
from pydantic import ValidationError

from drevalpy.models.config import (
    DrugFeaturizerConfig,
    FeaturizerConfig,
    ModelConfig,
    ModelScope,
    PredictionMode,
    PredictorConfig,
    ResolvedModelConfig,
    SampleFeaturizerConfig,
)
from drevalpy.models.config._predictor_traits import needs_identity_drug_routing, scope
from drevalpy.models.zoo import get_zoo_config
from drevalpy.registry.predictor import get as get_predictor

SAMPLE = SampleFeaturizerConfig(name="scaledGeneExpression")


def test_view_validation() -> None:
    with pytest.raises(ValidationError, match="view must be a non-empty string when set"):
        SampleFeaturizerConfig(name="denseSample", view="   ")
    assert SampleFeaturizerConfig(name="denseSample", view="gene_expression").view == "gene_expression"
    with pytest.raises(ValidationError):
        FeaturizerConfig(name="landmarkGenes", views=["gene_expression"])  # type: ignore[call-arg]


@pytest.mark.parametrize("cls", [FeaturizerConfig, SampleFeaturizerConfig])
def test_one_key_shorthand_is_accepted_by_base_and_pinned(cls: type[FeaturizerConfig]) -> None:
    config = cls.model_validate({"pca[methylation]": {"n_components": 42}})
    assert (config.name, config.view, config.registry) == ("pca", "methylation", "sample")
    assert config.hyperparameter_space is not None
    assert config.hyperparameter_space["n_components"]["default"] == 42


def test_featurizer_config_compact_forms() -> None:
    config = FeaturizerConfig.model_validate("fingerprints")
    assert (config.name, config.registry) == ("fingerprints", "sample")


def test_slot_subclasses_fix_the_registry() -> None:
    sample = SampleFeaturizerConfig.model_validate({"name": "scaledGeneExpression", "registry": "drug"})
    drug = DrugFeaturizerConfig.model_validate({"name": "fingerprints", "registry": "sample"})

    assert (sample.registry, drug.registry) == ("sample", "drug")
    assert isinstance(sample, FeaturizerConfig)
    assert isinstance(drug, FeaturizerConfig)


@pytest.mark.parametrize(
    ("drug", "predictor", "model_id"),
    [
        (DrugFeaturizerConfig(name="identity"), "randomForest", "scaledGeneExpression:identity:randomForest"),
        (None, "singleDrugElasticNet", "scaledGeneExpression:singleDrugElasticNet"),
    ],
    ids=["full-triple", "implicit-identity-single-drug"],
)
def test_model_id(drug, predictor, model_id) -> None:
    config = ModelConfig(sample_featurizer=SAMPLE, drug_featurizer=drug, predictor=PredictorConfig(name=predictor))

    assert config.model_id == model_id
    if drug is None:
        assert config.drug_featurizer is not None
        assert config.drug_featurizer.name == "identity"


def test_model_id_for_predictor_only_baseline() -> None:
    config = ModelConfig(sample_featurizer=None, drug_featurizer=None, predictor=PredictorConfig(name="naiveMean"))

    assert config.model_id == "naiveMean"


@pytest.mark.parametrize(
    ("drug", "predictor", "match"),
    [
        (None, "randomForest", "requires.*drug_featurizer|requires featurizers"),
        (None, "elasticNet", "requires.*drug_featurizer|requires featurizers"),
        (DrugFeaturizerConfig(name="fingerprints"), "singleDrugElasticNet", "requires drug_featurizer='identity'"),
    ],
    ids=["partial-multi-drug", "multi-drug-gets-no-identity", "single-drug-keeps-explicit-drug"],
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
            "sample_featurizer": [
                "scaledGeneExpression",
                {"pca[methylation]": {"n_components": 100}},
                "raw[mutations]",
            ],
            "drug_featurizer": "fingerprints",
            "predictor": {"randomForest": {"n_estimators": 10}},
        }
    )

    assert isinstance(config.sample_featurizer, SampleFeaturizerConfig)
    assert config.sample_featurizer.name == "concatFeaturizers"
    assert isinstance(config.drug_featurizer, DrugFeaturizerConfig)
    assert config.drug_featurizer.name == "fingerprints"
    assert config.predictor.name == "randomForest"
    assert config.predictor.hyperparameter_space is not None
    assert config.predictor.hyperparameter_space["n_estimators"]["default"] == 10


def test_model_config_rejects_base_featurizer_config_in_slots() -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(
            {
                "sample_featurizer": FeaturizerConfig(name="scaledGeneExpression", registry="drug"),
                "drug_featurizer": FeaturizerConfig(name="fingerprints", registry="sample"),
                "predictor": PredictorConfig(name="elasticNet"),
            }
        )


def test_config_is_serializable() -> None:
    config = ModelConfig(
        sample_featurizer=SAMPLE,
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
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
        sample_featurizer=SampleFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )


def test_model_config_is_frozen_and_scope_is_read_only() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        _template().drug_featurizer = None
    assert isinstance(ModelConfig.scope, property)
    assert ModelConfig.scope.fset is None


def test_concat_children_are_tuple_not_list() -> None:
    config = SampleFeaturizerConfig.model_validate(["scaledGeneExpression", {"pca[methylation]": {"n_components": 32}}])

    assert config.name == "concatFeaturizers"
    assert isinstance(config.featurizers, tuple)
    assert config.featurizers[1].name == "pca"
    assert isinstance(config.model_dump(mode="json")["featurizers"], list)
    with pytest.raises(TypeError):
        config.featurizers[0] = SampleFeaturizerConfig(name="raw", view="expression")  # type: ignore[index]


def test_predictor_shorthand_writes_hyperparameter_space_defaults() -> None:
    config = ModelConfig.model_validate(
        {
            "sample_featurizer": "scaledGeneExpression",
            "drug_featurizer": "fingerprints",
            "predictor": {"randomForest": {"n_estimators": 10}},
        }
    )
    space = config.predictor.hyperparameter_space
    assert space is not None
    assert space["n_estimators"]["default"] == 10
    with pytest.raises(AttributeError):
        _ = config.predictor.hyperparameters  # type: ignore[attr-defined]


def test_zoo_config_copy_isolation() -> None:
    first = get_zoo_config("MultiViewLightGBM")
    second = get_zoo_config("MultiViewLightGBM")
    assert first == second
    assert first is not second
    assert first.sample_featurizer is not None
    assert first.sample_featurizer.featurizers is not None
    child = first.sample_featurizer.featurizers[1]
    assert child.options is not None
    with pytest.raises(TypeError):
        child.options["gene_list"] = "landmark_genes"  # type: ignore[index]


def test_mapping_fields_are_deeply_frozen_views() -> None:
    featurizer = FeaturizerConfig.model_validate(
        {"name": "tissue", "options": {"allow_missing": True, "nested": {"inner": [1, {"deep": 2}]}}}
    )
    predictor = PredictorConfig.model_validate(
        {"name": "randomForest", "hyperparameter_space": {"n_estimators": {"default": 5, "options": [1, 2]}}}
    )
    resolved = ResolvedModelConfig(template=_template(), values={"predictor.elasticNet.alpha": 0.5})

    assert featurizer.options is not None
    assert featurizer.hyperparameter_space is None
    assert predictor.hyperparameter_space is not None
    for mapping in (
        featurizer.options,
        featurizer.options["nested"],
        predictor.hyperparameter_space,
        predictor.hyperparameter_space["n_estimators"],
        resolved.values,
        ResolvedModelConfig(template=_template()).values,
    ):
        assert isinstance(mapping, MappingProxyType)
        with pytest.raises(TypeError):
            mapping["injected"] = 1  # type: ignore[index]
    assert featurizer.options["nested"]["inner"] == (1, MappingProxyType({"deep": 2}))
    assert predictor.hyperparameter_space["n_estimators"]["options"] == (1, 2)


def test_frozen_mappings_dump_plain_containers() -> None:
    featurizer = FeaturizerConfig.model_validate({"name": "tissue", "options": {"nested": {"inner": [1, {"deep": 2}]}}})

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

    def test_rejects_a_non_tunable_value(self) -> None:
        with pytest.raises(ValidationError, match="non-tunable options"):
            PredictorConfig.model_validate({"randomForest": {"not_a_knob": 1}})


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
    assert scope(name) is expected


def test_scope_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown Predictor"):
        scope("noSuchPredictor")


@pytest.mark.parametrize(
    ("slot", "expected"),
    [
        ("singleDrugElasticNet", True),
        ({"singleDrugElasticNet": None}, True),
        ({"name": "singleDrugElasticNet"}, True),
        ({"singleDrugElasticNet": {"alpha": 0.5}}, True),
        (PredictorConfig(name="singleDrugElasticNet"), True),
        ("elasticNet", False),
        (None, False),
        (42, False),
        ({"name": "noSuchPredictor"}, False),
        ({"first": None, "second": None}, False),
        ({"name": "singleDrugElasticNet", "unexpected": 1}, False),
    ],
)
def test_identity_drug_routing(slot: object, expected: bool) -> None:
    assert needs_identity_drug_routing(slot) is expected


def _model_config(**kwargs: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "predictor": PredictorConfig(name="randomForest"),
        "sample_featurizer": SampleFeaturizerConfig(name="scaledGeneExpression"),
        "drug_featurizer": DrugFeaturizerConfig(name="fingerprints"),
    }
    defaults.update(kwargs)
    return ModelConfig.model_validate(defaults)


@pytest.mark.parametrize("name", ["identity", "constant"])
def test_id_only_featurizers_resolve_to_empty_views(name: str) -> None:
    config = _model_config(
        sample_featurizer=SampleFeaturizerConfig.model_validate(name),
        drug_featurizer=DrugFeaturizerConfig.model_validate(name),
    )
    assert config.sample_views() == []
    assert config.drug_views() == []


@pytest.mark.parametrize(
    ("featurizer", "views"),
    [
        (SampleFeaturizerConfig.model_validate("raw[mutations]+pca[methylation]"), ["mutations", "methylation"]),
        (SampleFeaturizerConfig.model_validate("landmarkGenes"), ["gene_expression"]),
        (SampleFeaturizerConfig.model_validate("landmarkGenesReduced"), ["gene_expression"]),
        (
            SampleFeaturizerConfig.model_validate("molirOmics"),
            ["gene_expression", "mutations", "copy_number_variation_gistic"],
        ),
        (
            SampleFeaturizerConfig.model_validate("superfeltrOmics"),
            ["gene_expression", "mutations", "copy_number_variation_gistic"],
        ),
        (SampleFeaturizerConfig(name="sparsegoOntology", options={"input_type": "expression"}), ["gene_expression"]),
        (SampleFeaturizerConfig(name="sparsegoOntology", options={"input_type": "mutations"}), ["mutations"]),
        (SampleFeaturizerConfig.model_validate("tissue"), []),
        (SampleFeaturizerConfig(name="scaledGeneExpression", options={"view": "proteomics"}), ["proteomics"]),
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
        drug_featurizer=DrugFeaturizerConfig(name="view", options={"view": "drug_chemberta_embeddings"})
    )
    assert view.drug_views() == ["drug_chemberta_embeddings"]


def test_feature_based_predictor_requires_a_drug_featurizer() -> None:
    with pytest.raises(ValidationError, match="requires a drug_featurizer"):
        _model_config(predictor=PredictorConfig(name="naiveSampleMean"), drug_featurizer=None)
