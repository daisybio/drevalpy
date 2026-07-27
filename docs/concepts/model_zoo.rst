Model zoo
=========

The model zoo is a set of named presets under ``drevalpy/models/zoo/*.yaml``.
Each preset wires cell-line featurizers, drug featurizers, and a predictor into
a runnable architecture (see :doc:`from_components_to_models`). CLI and Python
both refer to these preset names (for example ``ElasticNet`` or ``DIPK``).

Zoo inputs define architecture, not HPO search dimensions. Tuning searches
predictor (and tunable featurizer) hyperparameters on top of a fixed recipe.
Descriptions below come from the predictor registry; the composition column is
the resolved recipe for that preset.

.. include:: _generated_model_zoo.rst

Further reading
---------------

- :doc:`from_components_to_models` — recipes, concatenation, and hyperparameter spaces
- :doc:`/python/models` — constructing and migrating models in Python
- :doc:`/python/component_catalog` — predictors and featurizers behind the zoo
- :doc:`/cli/experiment` — selecting zoo names in CLI experiments

Backward compatibility
----------------------

Deprecated but still supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, factory dictionaries such as ``MODEL_FACTORY`` were a common
lookup path. This remains available for backward compatibility, but is
deprecated and may be removed in a future release. Prefer zoo presets and
the constructors documented in :doc:`/python/models`.

No longer supported
~~~~~~~~~~~~~~~~~~~

Deep imports such as ``drevalpy.models.baselines.*`` or
``drevalpy.models.DIPK.*`` no longer resolve. Use zoo names and the public
``drevalpy.models`` exports instead.
