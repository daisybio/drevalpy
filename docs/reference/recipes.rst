Recipe syntax
=============

A recipe is a string that names a model by its parts. The same notation is used for
``ModelConfig.model_id`` and for featurizer values in YAML files.

Grammar
-------

.. code-block:: text

   recipe     := sample ":" drug ":" predictor
               | sample ":" predictor
               | predictor
   sample     := featurizer ("+" featurizer)*
   drug       := featurizer ("+" featurizer)*
   featurizer := NAME ("[" VIEW "]")?

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Symbol
     - Meaning
   * - ``NAME``
     - A registry name: one or more characters other than ``[``, ``]``, ``+``, ``:`` and
       whitespace
   * - ``VIEW``
     - One or more characters other than ``[``, ``]`` and ``:``
   * - ``a:b:c``
     - Sample featurizer ``a``, drug featurizer ``b``, predictor ``c``
   * - ``a:c``
     - Sample featurizer ``a`` and predictor ``c``, without a drug featurizer
   * - ``c``
     - Predictor ``c`` alone
   * - ``x+y``
     - A ``concatFeaturizers`` node with the children ``x`` and ``y``
   * - ``name[view]``
     - The featurizer ``name`` reading the view ``view``; sets the ``view`` field

The recipe layer only checks this shape. Whether a name is registered, whether a featurizer
needs a view, and whether the featurizers fit the predictor (formats, required blocks, a drug
featurizer) are checked when the model configuration is built, the same way for a recipe and for
the equivalent YAML. Whether a view exists in the dataset is checked only when the model trains.

Examples
--------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - Recipe
     - Reads as
   * - ``scaledGeneExpression:fingerprints:elasticNet``
     - Sample featurizer, drug featurizer, predictor
   * - ``pca[gene_expression]:fingerprints:elasticNet``
     - ``pca`` on the ``gene_expression`` view
   * - ``raw[gene_expression]+raw[methylation]:fingerprints:elasticNet``
     - Concatenation of two ``raw`` views
   * - ``scaledGeneExpression:singleDrugElasticNet``
     - Single-drug predictor; the drug featurizer is filled in
   * - ``naiveMean``
     - Feature-free predictor

Rules
-----

Views
   The views a dataset offers are the sample view names ``gene_expression``,
   ``methylation``, ``mutations``, ``proteomics`` and ``copy_number_variation_gistic``, plus
   the drug feature keys it stores; see :doc:`datasets`. There are no aliases such as
   ``expression`` or ``cnv``. A view the dataset lacks fails when the model trains, for
   example ``KeyError: "Modality 'expression' not found. Available: ['gene_expression',
   'response']"``. ``drevalpy.run`` records such a failure and continues with the other runs; see
   :doc:`/how-to/evaluate_component_grids`.

Featurizers that need a view
   Some featurizers, among them ``raw`` and ``pca``, require an explicit view. ``raw`` without
   a view raises ``requires an explicit view``. :doc:`components` lists the featurizers.

Duplicate leaves
   One concatenation may not hold the same qualified featurizer twice, for example
   ``raw[gene_expression]+raw[gene_expression]``.

Drug featurizer
   A multi-drug predictor that takes featurizers needs both a sample and a drug featurizer;
   ``sample:predictor`` is valid for a single-drug predictor. A single-drug predictor given a
   sample featurizer and no drug featurizer receives the routing featurizer ``identity``. The
   scope of the model follows from the predictor.

Feature-free predictors
   A bare predictor name is valid only for a feature-free predictor such as ``naiveMean``.
   Other predictors raise ``requires featurizers``.

Where recipes are accepted
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Place
     - Accepts
   * - ``drevalpy.models.config.from_spec(spec)``
     - A zoo preset name, else a recipe. A zoo name wins over a bare predictor of the same
       name.
   * - ``ModelSpec(name, config)``
     - A ``ModelConfig``, for example ``from_spec(...)``
   * - ``ModelSpec.from_name``, the ``MODELS`` of ``drevalpy run`` and ``drevalpy single``
     - A zoo name, else a recipe. A zoo name wins over a recipe of the same name. The model is
       named by the string as given. A name that is neither raises ``UnknownModelError``, a
       ``ValueError`` whose message gives the reason the recipe reading failed
   * - ``drevalpy.models.combinations``, ``drevalpy run -S/-D/-P``
     - Lists of sample featurizer recipes, drug featurizer recipes and predictor names; every
       valid ``sample:drug:predictor`` is built and the others are reported. See
       :doc:`/how-to/evaluate_component_grids`
   * - The ``sample_featurizer`` and ``drug_featurizer`` keys of YAML and ``ModelConfig``
     - A featurizer part of a recipe, such as ``pca[methylation]`` or ``raw[a]+raw[b]``

A recipe needs no registration. To give it a short name, register it as a zoo preset; see
:doc:`extension_files`.

Model identifier
----------------

``ModelConfig.model_id`` writes a recipe from the names only. It drops the views and the
routing ``identity`` featurizer and writes ``concatFeaturizers`` for ``+``. For example,
``pca[gene_expression]:fingerprints:elasticNet`` has the identifier
``pca:fingerprints:elasticNet``. It is ``None`` when only one of the two featurizers is set.

Featurizer forms in YAML
------------------------

The ``sample_featurizer`` and ``drug_featurizer`` keys of a YAML file or a ``ModelConfig``
take any of these forms.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Form
     - Example
   * - Name
     - ``scaledGeneExpression``
   * - Name with a view, or a ``+`` string
     - ``raw[gene_expression]+raw[methylation]``
   * - List, a concatenation
     - ``[raw[gene_expression], raw[methylation]]``
   * - Mapping with fields
     - ``{name: raw, view: gene_expression}``
   * - Mapping with children
     - ``{name: concatFeaturizers, featurizers: [...]}``
   * - One-key mapping
     - ``{"pca[methylation]": {n_components: 8}}``

The fields of the mapping form are:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Meaning
   * - ``name``
     - Registry name
   * - ``registry``
     - ``sample`` or ``drug``; fixed by the key
   * - ``view``
     - The view to read
   * - ``featurizers``
     - Children of a ``concatFeaturizers`` node
   * - ``options``
     - Fixed constructor values
   * - ``hyperparameter_space``
     - The search space for this featurizer; see :doc:`hyperparameters`

In the one-key form, each value under the key either replaces the ``default`` of a
hyperparameter the component declares, or becomes a fixed entry in ``options`` if the
component does not declare it. A predictor takes the one-key form too, as
``{elasticNet: {alpha: 0.5}}``, but an option it does not declare raises an error. The
predictor can also be ``{name: ..., hyperparameter_space: ...}``.

A model configuration has the keys ``sample_featurizer``, ``drug_featurizer``, ``predictor``
and ``prediction_mode`` (``regression`` by default). Other keys raise an error.
