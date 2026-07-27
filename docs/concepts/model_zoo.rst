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
- :doc:`/python/models` — constructing models in Python (including factory migration notes)
- :doc:`/python/component_catalog` — predictors and featurizers behind the zoo
- :doc:`/cli/experiment` — selecting zoo names in CLI experiments
