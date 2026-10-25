# Models

Built-in models are reusable `ModelSpec` recipes resolved from the zoo and
fitted through `DRPModel`.

```python
from drevalpy.models import DRPModel, ModelSpec

spec = ModelSpec.from_name("ElasticNet")
model = DRPModel(spec, {"alpha": 0.1, "l1_ratio": 0.5})
model.train(dataset, scope)
prediction = model.predict(dataset, scope)
```

For a custom component configuration, create a `ModelConfig` with
`from_spec`, then give it a public name:

```python
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import from_spec

spec = ModelSpec("MyModel", from_spec("scaledGeneExpression:fingerprints:elasticNet"))
model = DRPModel(spec)
```

Plugins register their featurizers and predictors through the normal registry
or `load_extensions` path. A fitted checkpoint is restored with
`DRPModel.load(path)` after the same custom components are registered.
