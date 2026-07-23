Run your own model
===================

DrEvalPy models are composed from registered **featurizers** and **predictors**.
Do not subclass ``DRPModel`` directly for new models. Register components, then
compose them with a ``ModelConfig`` / zoo preset; the root factory exposes a
generated facade with the usual train/predict/save/load lifecycle.

High-level path
---------------

1. Register a featurizer and/or predictor under ``drevalpy.components``.
2. Describe the stack with a recipe string, YAML zoo entry, or ``ModelConfig`` dict.
3. Use ``construct_model`` or add a zoo YAML so ``MODEL_FACTORY`` picks it up.

.. code-block:: Python

    from drevalpy.models import construct_model
    from drevalpy.models.config import ModelConfig

    # Recipe: cellLineFeaturizer:drugFeaturizer:predictor
    MyModel = construct_model(
        "MyModel",
        "scaledGeneExpression:fingerprints:elasticNet",
    )
    model = MyModel()
    model.build_model({"alpha": 0.1, "l1_ratio": 0.5})

    # Or load a YAML preset
    config = ModelConfig.from_yaml("path/to/preset.yaml")
    composed = config.create_model()

Zoo presets
-----------

Built-in models live under ``drevalpy/models/zoo/*.yaml``. Each file is one
factory name. Single-drug models set ``scope: single_drug``. Early stopping is
derived from predictor capability metadata.

Example zoo entry:

.. code-block:: yaml

    cell_line_featurizer: scaledGeneExpression
    drug_featurizer: fingerprints
    predictor: elasticNet

Flat hyperparameters
--------------------

Public ``build_model`` still accepts the historical flat hyperparameter dict.
Overrides are translated onto the resolved ``ModelConfig`` (predictor and
featurizer local keys, plus ``cell_line_views`` / ``drug_views`` when present).

Unsupported
-----------

The following are intentionally removed:

* Deep imports such as ``drevalpy.models.baselines.*`` or ``drevalpy.models.DIPK.*``
* Documented ``DRPModel`` subclass authoring as the extension path
* Legacy checkpoint formats and fitted-state introspection (``.model``, scalers, naive means)

Root usage is unchanged: ``MODEL_FACTORY``, named root exports, experiment
routing, and the CLI continue to work as before.
