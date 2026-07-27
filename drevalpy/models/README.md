# Models

Built-in models are generated from zoo presets under `drevalpy/models/zoo/` and
exposed through the root `drevalpy.models` package (`construct_model`, named
classes, `ModelConfig`).

## Extension path

Do **not** subclass `DRPModel` directly for new models. Instead:

1. Register featurizers and/or predictors under `drevalpy.components`.
2. Compose them with a `ModelConfig` (YAML zoo entry, recipe triple, or dict).
3. Use `construct_model(name)`, `construct_model(name, spec)`, or
   `ModelConfig.from_spec(...)`.

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
composed = config.create_model()
```

See `docs/python/custom_models.rst` for a complete external extension walkthrough.

## Breaking changes

**Preferred:** `construct_model`, named root exports (`ElasticNetModel`, …),
`ModelConfig`, zoo discovery via `list_zoo_names(scope=...)`, experiment/CLI
model-name strings.

**Deprecated (still functional, emits `FutureWarning`):**

- `MODEL_FACTORY`, `MULTI_DRUG_MODEL_FACTORY`, `SINGLE_DRUG_MODEL_FACTORY`
- Flat `cell_line_views` / `drug_views` in `build_model` / hpam YAML

**Removed:**

- Deep imports such as `drevalpy.models.baselines.*` or `drevalpy.models.DIPK.*`
  — use `from drevalpy.models import DIPKModel` or `construct_model("DIPK")`.
- Legacy checkpoint formats — retrain and save via `composed_model.joblib`.
- Iterating `get_hyperparameter_set()` as a full grid — use
  `hyperparameter_tuning=True` or `get_structured_hyperparameter_space()`.

**Core dependencies:** `pydantic`, `optuna`, `ray[tune]`, `xgboost`, `lightgbm`,
`gseapy`, `mygene`, and `obonet` ship with the default install.

## Unsupported

- Direct `DRPModel` subclass authoring as the documented extension path
- Fitted-state introspection on legacy attributes (`.model`, private scalers, naive means)
