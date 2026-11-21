Model zoo
=========

The zoo is a set of named model configurations, called zoo presets. A zoo name selects a
complete model in ``drevalpy run``, ``drevalpy single`` and ``ModelSpec.from_name``.
``drevalpy list zoo-presets`` prints the names.

Each built-in zoo preset is a YAML file in ``drevalpy/models/zoo``. It has the keys of a model
configuration (``sample_featurizer``, ``drug_featurizer``, ``predictor``), so it can also
set fixed options, defaults and hyperparameter spaces of its components; see :doc:`recipes` and
:doc:`hyperparameters`. For example, ``MOLIR.yaml`` holds:

.. code-block:: yaml

   sample_featurizer:
     name: molirOmics
     options:
       gene_lists:
         gene_expression: gene_expression_intersection
         mutations: mutations_intersection
         copy_number_variation_gistic: copy_number_variation_gistic_intersection
   predictor: molir

How zoo presets are named and added
-----------------------------------

- A built-in zoo preset is named after its file.
- An external zoo file holds either one zoo preset, named by its ``name`` key or by the
  file stem, or a mapping from zoo name to configuration. The formats are in
  :doc:`extension_files`.
- An external zoo preset may not reuse a built-in zoo name; that raises ``ValueError``.
- ``drevalpy.models.zoo.list_zoo_names()`` returns the names, including external zoo presets.

Reading the table
-----------------

Name
   The zoo name.

Description
   The description of the predictor.

Composition
   The recipe of the zoo preset, its ``model_id``. Passing it to
   ``drevalpy.models.config.from_recipe`` gives the same components, but without the preset's
   fixed options, defaults and hyperparameter spaces, which a recipe cannot express. See
   :doc:`recipes`.

.. include:: _generated_model_zoo.rst
