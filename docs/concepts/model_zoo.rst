Model zoo
=========

The first two pages established the vocabulary and grammar:

- :doc:`component_catalog` lists the available building blocks.
- :doc:`from_components_to_models` shows how they form a recipe.

The model zoo is the final layer: a collection of named, ready-to-run recipes
under ``drevalpy/models/zoo/*.yaml``. Instead of spelling out
``scaledGeneExpression:fingerprints:elasticNet``, workflows can select the
``ElasticNet`` preset. CLI and Python use the same preset names.

Why presets?
------------

A zoo preset gives an architecture:

- a short, stable name,
- a validated combination of featurizers and predictor,
- documented intent, and
- one definition shared by every interface.

Use a preset when it already represents the architecture you need. Compose a
custom recipe when you deliberately want a different combination of registered
components; register an extension only when the catalog itself lacks a needed
component.

Reading the catalog
-------------------

Zoo inputs define architecture, not HPO search dimensions. Tuning searches
predictor (and tunable featurizer) hyperparameters on top of a fixed recipe.
In the tables below:

- **Name** is the preset selected by a workflow.
- **Description** states its intended model family.
- **Composition** is the resolved recipe, using the exact atoms introduced in
  the component catalog.

.. include:: _generated_model_zoo.rst

Continue from here
------------------

- :doc:`/cli/experiment` — selecting zoo names in CLI experiments
- :doc:`/python/models` — selecting and constructing models in Python
- :doc:`/python/hyperparameter_tuning` — tuning a fixed preset architecture
- :doc:`from_components_to_models` — revisit recipes and custom composition
