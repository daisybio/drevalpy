Model zoo
=========

The previous two pages established the vocabulary and grammar of building models:

- :doc:`component_catalog` lists the available building blocks.
- :doc:`from_components_to_models` shows how they form a recipe.

While these interfaces allow building models in a flexible way, we are aware that certain featurizer-predictor combinations are used frequently.
In order to make it easier to use these frequently used combinations, we provide a so-called 'model zoo' which is a collection of curated model configurations.

Each model configuration is a YAML configuration file. The name is derived from the file name.
Just like any configuration YAML file, zoo YAML files can contain overrides for the hyperparameter spaces and default hyperparameters of the components.

In the table below, you can find the currently available zoo models.

- **Name** is the alias of the zoo model. It can be used everywhere a recipe string is accepted.
- **Description** the description of the predictor.
- **Composition** the equivalent recipe string, using the exact atoms introduced in the component catalog.

.. include:: _generated_model_zoo.rst

Continue from here
------------------

- :doc:`/cli/experiment` — selecting zoo names in CLI experiments
- :doc:`/python/models` — selecting and constructing models in Python
- :doc:`/python/hyperparameter_tuning` — tuning a fixed preset architecture
- :doc:`from_components_to_models` — revisit recipes and custom composition
