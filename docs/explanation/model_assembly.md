---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# How a model name becomes a running model

Whether you pass a zoo preset like `"ElasticNet"`, a recipe like
`"scaledGeneExpression:fingerprints:elasticNet"`, or a YAML file, DrEvalPy turns it into a model
ready to train through the same pipeline. This page walks through the three types a model moves
through — `ModelConfig`, `ModelSpec`, and `DRPModel` — and explains why each layer exists.

For the bigger picture around this pipeline, see {doc}`models_from_components` (why models split
into featurizers and a predictor), {doc}`registries_and_plugins` (how registries find components),
{doc}`data_flow` (how a built model consumes data), and {doc}`training_lifecycle` (how folds and
tuning wrap training).

## The path at a glance

```text
"scaledGeneExpression:fingerprints:elasticNet" | Zoo name | YAML | Python dict
  │  ModelSpec.from_name: checks zoo_registry() first, else parses recipe
  │    parse_model_recipe: syntax → dict (no registry lookup)
  │    model_validate: normalizes featurizer and predictor entries
  │    validate: checks predictor inputs against featurizer outputs (before loading data)
  ▼
ModelConfig  immutable model configuration + search spaces, no concrete hyperparameter values
  │  space(config): collects search spaces under qualified keys
  │  qualify(space, {...}): resolves short keys → qualified keys
  │  hyperparameters: merges defaults and freezes them
  ▼
ModelSpec  one concrete value per qualified key; unfitted, safe to reuse across folds
  ▼
DRPModel(spec, overrides)  public entry point that owns fitted state
  │  _build_components (on each train): instantiates fresh featurizers and predictor
  │  train(dataset, scope): fits response transformation, featurizers, then predictor
  │  predict(dataset, scope): applies inverse transformation to predictor output
```

## Why three separate types?

Splitting a model's lifecycle across three types keeps configuration, hyperparameter choices, and
fitted weights cleanly separated:

- **`ModelConfig`** is the reusable template. It says which components a model uses and defines
  their hyperparameter search spaces, but holds no concrete values — a `pca` featurizer declares
  `{"n_components": {"type": "int", "low": 8, "high": 512, "default": 128}}`, yet the config never
  commits to `n_components=128`. Configs are frozen at construction so shared or cached configs
  cannot be mutated, and any search space written in the config overrides the component's own
  default space without needing a subclass.
- **`ModelSpec`** pairs a `ModelConfig` with a display name and one concrete value for every
  hyperparameter, keyed by qualified names such as `predictor.elasticNet.alpha`. It holds no
  fitted state, which makes it safe to reuse across folds, tuning trials, and prediction calls.
  Keeping values out of `ModelConfig` is what lets tuning draw dozens of trial specs from one
  template without rebuilding it.
- **`DRPModel`** wraps a `ModelSpec` and owns the fitted state: the response transformation, the
  featurizers, and the predictor. Every call to `train` builds fresh component instances from
  scratch, so old weights or scalers from one fold can never leak into the next.

```{code-cell} ipython3
:tags: [remove-stderr]

import tempfile
from upath import UPath

from drevalpy.models.config.io import from_yaml
from drevalpy.models.spec import ModelSpec

# Every accepted spelling resolves into the same ModelConfig.
spec_from_recipe = ModelSpec.from_name("naiveMean")
spec_from_zoo = ModelSpec.from_name("NaivePredictor")

with tempfile.TemporaryDirectory() as tmp:
    yaml_path = UPath(tmp) / "model.yaml"
    yaml_path.write_text("predictor: naiveMean\n", encoding="utf-8")
    config_from_yaml = from_yaml(yaml_path)
spec_from_yaml = ModelSpec("FromYAML", config_from_yaml)

print("Recipe name:", spec_from_recipe.name)
print("Zoo name:", spec_from_zoo.name)
print("YAML name:", spec_from_yaml.name)
print("config.predictor.name:", spec_from_recipe.config.predictor.name)
print("All three configs are equal:", spec_from_recipe.config == spec_from_zoo.config == spec_from_yaml.config)
print("ModelConfig is immutable:", spec_from_recipe.config.model_config["frozen"])
```

## Parsing recipes without looking anything up

The recipe parser ({func}`~drevalpy.models.config._parse.parse_model_recipe`) is purely syntactic.
It splits a string like `scaledGeneExpression:fingerprints:elasticNet` on its delimiters and
produces the exact same dictionary a YAML file would have written, without touching any registry.

It understands `+` for concatenating featurizers and `[]` for selecting a view, so
`pca[methylation]+tissue:fingerprints:elasticNet` becomes a concat node with two sample
featurizers. Because recipes and YAML files turn into the same dictionary before any validation
runs, they behave identically everywhere downstream.

```{code-cell} ipython3
from drevalpy.models.config._parse import parse_model_recipe

# parse_model_recipe produces a plain dict, not a validated config.
raw = parse_model_recipe("scaledGeneExpression:fingerprints:elasticNet")
print("Parsed recipe keys:", list(raw.keys()))
print("sample_featurizer:", raw["sample_featurizer"])
print("drug_featurizer:", raw["drug_featurizer"])
print("predictor:", raw["predictor"])
```

## Catching mismatches before loading data

When {class}`~drevalpy.models.config.model.ModelConfig` is built,
{func}`~drevalpy.models.config.validation.validate` checks that the chosen featurizers and
predictor actually fit together:

1. Each featurizer and predictor entry is normalized
   ({func}`~drevalpy.models.config._parse.normalize_featurizer` and
   {func}`~drevalpy.models.config._parse.normalize_predictor`).
2. Every leaf featurizer declares which feature blocks it produces at default hyperparameters via
   {meth}`~drevalpy.components.featurizers.featurizer.Featurizer.output_blocks`.
3. The predictor declares what it reads from each side via
   {func}`~drevalpy.components.predictors.inputs.predictor_inputs`.
4. Validation verifies that every side the predictor reads has a featurizer, every side it ignores
   has none, matrix predictors receive only numeric blocks, and block predictors receive every
   named block they ask for.

Because this runs at config construction — before any dataset is loaded — an incompatible recipe
fails immediately instead of hours into an experiment. The trade-off is that featurizers must
declare their output feature formats without inspecting the dataset (and a concat featurizer's
outputs are simply the union of its children's).

```{code-cell} ipython3
from drevalpy.models.config import ModelConfig

# A matrix predictor rejects a drug graph featurizer.
try:
    ModelConfig.model_validate(
        {
            "sample_featurizer": "scaledGeneExpression",
            "drug_featurizer": "drugGraph",
            "predictor": "elasticNet",
        }
    )
except ValueError as exc:
    print("Rejected (matrix vs graph):", str(exc)[:180])

# A tissue-mean predictor requires the tissue block.
try:
    ModelConfig.model_validate(
        {
            "sample_featurizer": "scaledGeneExpression",
            "predictor": "naiveTissueMean",
        }
    )
except ValueError as exc:
    print("Rejected (missing block):", str(exc)[:180])
```

## Qualified hyperparameter keys

A model can combine several components — even two instances of the same featurizer on different
views, like `pca[gene_expression]+pca[proteomics]`. To keep their hyperparameters unambiguous,
every hyperparameter has a **qualified key** of the form `<slot>.<selector>.<param>`, built by
{func}`~drevalpy.models.hyperparameters.key`:

- **`slot`**: `predictor`, `sample_featurizer`, or `drug_featurizer`.
- **`selector`**: the predictor name, or the featurizer name with its view (`pca[methylation]`, or
  just `oneHot` when no view is selected).
- **`param`**: the component's hyperparameter name.

The full search space of a {class}`~drevalpy.models.spec.ModelSpec` is computed once by
{func}`~drevalpy.models.tuning.search_space.space` under these qualified keys, and results,
Weights & Biases logs, and saved models all record them.

For convenience, you don't always have to type the full key.
{func}`~drevalpy.models.hyperparameters.qualify` is the single entry point that resolves user-given
keys: a short name like `"alpha"` works whenever exactly one component owns it, while an unknown or
ambiguous short name raises `ValueError` up front.
{meth}`ModelSpec.with_hyperparameters <drevalpy.models.spec.ModelSpec.with_hyperparameters>` routes
every override through `qualify`, so key resolution behaves the same everywhere.

```{code-cell} ipython3
spec = ModelSpec.from_name("scaledGeneExpression:fingerprints:elasticNet")

print("Qualified search space (first 5):", dict(list(spec.hyperparameter_space.items())[:5]))
print()
print(
    "Short key 'alpha' resolves to:",
    spec.with_hyperparameters({"alpha": 0.5}).hyperparameters["predictor.elasticNet.alpha"],
)

# An ambiguous short key raises ValueError when two components own it.
spec_two_pca = ModelSpec.from_name("pca[gene_expression]+pca[proteomics]:fingerprints:elasticNet")
try:
    spec_two_pca.with_hyperparameters({"n_components": 50})
except ValueError as exc:
    print("Ambiguous 'n_components':", str(exc)[:160])
```

## Fresh components on every `train` call

A newly created {class}`~drevalpy.models.drp_model.DRPModel` holds no featurizers or predictor yet.
Only when `train` is called does `_build_components` instantiate fresh ones by calling
{meth}`~drevalpy.models.config.featurizer.FeaturizerConfig.create_instance` and
{meth}`~drevalpy.models.config.predictor.PredictorConfig.create_instance` with each component's
hyperparameters from {func}`~drevalpy.models.hyperparameters.component_values`. For a concat
featurizer it builds the children first and passes them in, and a predictor also receives
`prediction_mode`.

Rebuilding components on every `train` call prevents subtle cross-fold leakage: a PCA fitted on
fold 0 is discarded before fold 1 begins, and a neural network always starts from fresh weights.

```{code-cell} ipython3
:tags: [remove-stderr]

import tempfile

import drevalpy
from drevalpy.models.drp_model import DRPModel

spec = ModelSpec.from_name("ElasticNet")
model = DRPModel(spec, hyperparameters={"alpha": 0.1})

print("Before train, state:", model.state.value)
print("Predictor is None:", model._predictor is None)

dataset = drevalpy.synthetic(n_samples=50, n_drugs=20)
folds = drevalpy.split(dataset, "LPO", n_splits=2)

with tempfile.TemporaryDirectory() as tmp:
    model.train(dataset, folds[0].train, model_checkpoint_dir=tmp)
    print("\nAfter first train, state:", model.state.value)
    print("Predictor exists:", model._predictor is not None)
    first_pred_id = id(model._predictor)

    model.train(dataset, folds[1].train, model_checkpoint_dir=tmp)
    print("After second train, state:", model.state.value)
    second_pred_id = id(model._predictor)
    print("Fresh predictor built (different identity):", first_pred_id != second_pred_id)
```

## What happens next

Once `DRPModel.train` has built fresh components, it fits the optional response transformation
({func}`~drevalpy.models.response_transformation._pipeline_ops.fit_response_transformation`),
featurizes the unique sample and drug IDs
({func}`~drevalpy.models._feature_preparation.prepare_features`), packs the response table and
entity tables into a {class}`~drevalpy.types.data.predictor_input.PredictorInput`, and passes it to
`Predictor.fit`. See {doc}`data_flow` for how `PredictorInput` works and {doc}`training_lifecycle`
for how tuning and early stopping wrap this core path.
