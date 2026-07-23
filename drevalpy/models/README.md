# Models

Built-in models are generated from zoo presets under `drevalpy/models/zoo/` and
exposed through the root `drevalpy.models` package (`MODEL_FACTORY`, named
classes, `construct_model`).

## Extension path

Do **not** subclass `DRPModel` directly for new models. Instead:

1. Register featurizers and/or predictors under `drevalpy.components`.
2. Compose them with a `ModelConfig` (YAML zoo entry, recipe triple, or dict).
3. Use `construct_model(name, spec)`, `ModelConfig.from_spec(...)`, or add a zoo
   YAML so the root factory generates a facade automatically.

Example:

```python
from drevalpy.components import load_extensions
from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig

load_extensions(directories=["./my_components"], zoo_files=["./my_zoo.yaml"])

MyModel = construct_model("MyModel", "scaledGeneExpression:fingerprints:elasticNet")
config = ModelConfig.from_spec("MyModel")  # after zoo registration
model = config.create_model()
```

See `docs/runyourmodel.rst` for a complete external extension walkthrough.

## Breaking changes

**Still supported:** root imports (`MODEL_FACTORY`, `ElasticNetModel`, …),
`construct_model`, experiment/CLI model names, flat `build_model` hyperparameters.

**Removed:**

- Deep imports such as `drevalpy.models.baselines.*` or `drevalpy.models.DIPK.*`
  — use `from drevalpy.models import DIPKModel` instead.
- Legacy checkpoint formats — retrain and save via `composed_model.joblib`.
- Iterating `get_hyperparameter_set()` as a full grid — use
  `hyperparameter_tuning=True` or `get_structured_hyperparameter_space()`.

**Core dependencies:** `pydantic`, `optuna`, and `ray[tune]` ship with the default
install (not optional extras).

## Unsupported

- Direct `DRPModel` subclass authoring as the documented extension path
- Fitted-state introspection on legacy attributes (`.model`, private scalers, naive means)
