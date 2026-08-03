# Models

Built-in models are generated from zoo presets under `drevalpy/models/zoo/` and
exposed through `construct_model` on the root `drevalpy.models` package.

## Extension path

Do **not** subclass `DRPModel` directly for new models. Instead:

1. Register featurizers and/or predictors under `drevalpy.components`.
2. Compose them with a `ModelConfig` (YAML zoo entry, recipe triple, or dict).
3. Call `construct_model(name)`, `construct_model(name, spec)`, or
   `construct_model(name, config)`.

Example:

```python
from drevalpy.components import load_extensions
from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig

load_extensions(directories=["./my_components"], zoo_files=["./my_zoo.yaml"])

MyModel = construct_model("MyModel", "scaledGeneExpression:fingerprints:elasticNet")
# Or resolve a registered zoo name:
MyModelZoo = construct_model("MyModel")
config = ModelConfig.from_spec("MyModel")
MyModelFromConfig = construct_model("MyModel", config)
model = MyModelZoo()
```

Reload a fitted checkpoint without a class handle:

```python
from drevalpy.models import load_model

loaded = load_model("checkpoints/my_model")
```

See `docs/python/custom_models.rst` for a complete external extension walkthrough.

## Breaking changes

**Preferred:** `construct_model`, declarative `ModelConfig`, zoo discovery via
`list_zoo_names(scope=...)`, experiment/CLI model-name strings.

**Deprecated (still functional, emits `FutureWarning`):**

- `MODEL_FACTORY`, `MULTI_DRUG_MODEL_FACTORY`, `SINGLE_DRUG_MODEL_FACTORY`
  — lazy built-in-only views equal to `construct_model(name)` for zoo names
- Flat `cell_line_views` / `drug_views` in constructor / hpam YAML

**Removed:**

- Named root exports (`ElasticNetModel`, `DIPKModel`, …) — use
  `construct_model("ElasticNet")`
- `ModelConfig.create_model()` — use `construct_model(...)()` instead
- Deep imports such as `drevalpy.models.baselines.*` or `drevalpy.models.DIPK.*`
- Legacy checkpoint formats (including `composed_model.joblib`) — retrain and
  save via `model.save` / `ModelClass.load` (`model.joblib`, format `drevalpy-model`)
- Iterating `get_hyperparameter_set()` as a full grid — use
  `hyperparameter_tuning=True` or `get_structured_hyperparameter_space()`

**Core dependencies:** `pydantic`, `optuna`, `ray[tune]`, `xgboost`, `lightgbm`,
`gseapy`, `mygene`, and `obonet` ship with the default install.

## Unsupported

- Direct `DRPModel` subclass authoring as the documented extension path
- Fitted-state introspection on legacy attributes (`.model`, private scalers, naive means)
