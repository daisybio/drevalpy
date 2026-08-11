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
from drevalpy.models import config, construct_model

load_extensions(directories=["./my_components"], zoo_files=["./my_zoo.yaml"])

MyModel = construct_model("MyModel", "scaledGeneExpression:fingerprints:elasticNet")
# Or resolve a registered zoo name:
MyModelZoo = construct_model("MyModel")
cfg = config.from_spec("MyModel")
MyModelFromConfig = construct_model("MyModel", cfg)
model = MyModelZoo()
```

Reload a fitted checkpoint without a class handle:

```python
from drevalpy.models import load_model

loaded = load_model("checkpoints/my_model")  # reads checkpoints/my_model.zip
```

See `docs/python/custom_models.rst` for a complete external extension walkthrough.

## Unsupported

- Direct `DRPModel` subclass authoring as the documented extension path
- Direct fitted-state introspection (use `model.save`/`ModelClass.load` instead)
