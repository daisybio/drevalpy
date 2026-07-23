# Models

Built-in models are generated from zoo presets under `drevalpy/models/zoo/` and
exposed through the root `drevalpy.models` package (`MODEL_FACTORY`, named
classes, `construct_model`).

## Extension path

Do **not** subclass `DRPModel` directly for new models. Instead:

1. Register featurizers and/or predictors under `drevalpy.components`.
2. Compose them with a `ModelConfig` (YAML zoo entry, recipe triple, or dict).
3. Use `construct_model(name, spec)` or add a zoo YAML so the root factory
   generates a facade automatically.

Example:

```python
from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig

MyModel = construct_model("MyModel", "scaledGeneExpression:fingerprints:elasticNet")
# or
config = ModelConfig.from_yaml("path/to/preset.yaml")
composed = config.create_model()
```

## Unsupported

- Deep imports such as `drevalpy.models.baselines.*` or `drevalpy.models.DIPK.*`
- Direct `DRPModel` subclass authoring as the documented extension path
- Legacy checkpoints and fitted-state introspection (`.model`, scalers, means)
