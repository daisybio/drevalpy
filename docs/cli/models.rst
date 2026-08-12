Models
======

If you are reading this, we assume you are already familiar with these
concepts:

- :doc:`/concepts/component_catalog`
- :doc:`/concepts/from_components_to_models`
- :doc:`/concepts/model_zoo`

This page covers how to select and configure models from the CLI.

Passing models to ``drevalpy run``
----------------------------------

Model names are passed as positional arguments to ``drevalpy run``:

.. code-block:: bash

   drevalpy run ElasticNet RandomForest --dataset GDSC1 --split-mode LCO

Names correspond to **zoo presets** — the same names you would pass to
``construct_model("ElasticNet")`` in Python. The full list of built-in presets
is documented in :doc:`/concepts/model_zoo`.

Multiple models
~~~~~~~~~~~~~~~

Space-separate model names to evaluate several in one run:

.. code-block:: bash

   drevalpy run ElasticNet RandomForest GradientBoosting --dataset GDSC1 --split-mode LPO

All models are evaluated on the same folds. Results are combined into a single
``ExperimentResult`` directory.

Custom models via extensions
----------------------------

If your model is defined in a custom extension (registered via decorators in
a ``.py`` file), make it available with ``--extensions-dir`` or the
``DREVALPY_EXTENSIONS_DIR`` environment variable:

.. code-block:: bash

   drevalpy --extensions-dir my_components/ run ToyRidge --dataset GDSC1 --split-mode LCO

The ``--extensions-dir`` flag is a **global option** that must appear before
the subcommand. It loads all ``.py`` files in the directory (triggering
registration decorators) and all ``.yaml`` files as zoo presets.

See :doc:`extensions` for more details on extension loading and
:doc:`/python/extensions` for how to write custom components.

Model composition
-----------------

From the CLI, models are selected by zoo name only. For custom compositions
(recipe strings, YAML, or ``ModelConfig``), register them in a zoo YAML file
under your extensions directory:

.. code-block:: yaml

   MyCustomEN:
     cell_line_featurizer:
       name: pca
       view: expression
     drug_featurizer: fingerprints
     predictor: elasticNet

Then reference by name:

.. code-block:: bash

   drevalpy --extensions-dir my_zoo/ run MyCustomEN --dataset GDSC1 --split-mode LCO

Composition concepts (recipe grammar, YAML fields, contracts) are documented
in :doc:`/concepts/from_components_to_models`.
