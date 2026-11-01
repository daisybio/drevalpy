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
needs a view, whether a featurizer supports its side, and whether the featurizers fit the
predictor (inputs, required blocks, a drug featurizer) are checked when the model configuration is built, the same way for a recipe and for
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
   * - ``view[gene_expression]+view[methylation]:fingerprints:elasticNet``
     - Concatenation of two ``view`` views
   * - ``scaledGeneExpression:singleDrugElasticNet``
     - Single-drug predictor, without a drug featurizer
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
   Some featurizers, among them ``view`` and ``pca``, require an explicit view. ``view`` without
   a view raises ``requires an explicit view``. :doc:`components` lists the featurizers.

Duplicate leaves
   One concatenation may not hold the same qualified featurizer twice, for example
   ``view[gene_expression]+view[gene_expression]``.

Sides
   Each featurizer is registered on one side and is looked up in that side's registry. A
   featurizer on the wrong side raises, for example, ``Unknown sample featurizer
   'fingerprints'``, followed by the sample featurizers there are.

Drug featurizer
   A multi-drug predictor that takes featurizers needs both a sample and a drug featurizer;
   ``sample:predictor`` is the form for a single-drug predictor. A single-drug predictor reads
   no drug features, so a drug featurizer raises ``Predictor 'singleDrugElasticNet' reads no
   drug features and forbids a drug_featurizer``. The scope of the model follows from the
   predictor.

Feature-free predictors
   A bare predictor name is valid only for a feature-free predictor such as ``naiveMean``.
   Other predictors raise, for example, ``Predictor 'elasticNet' requires a sample_featurizer``.

Where recipes are accepted
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Place
     - Accepts
   * - ``drevalpy.models.config.from_recipe(recipe)``
     - A recipe only; zoo names are resolved by ``ModelSpec.from_name``
   * - ``ModelSpec(name, config)``
     - A ``ModelConfig``, for example ``from_recipe(...)``
   * - ``ModelSpec.from_name``, the ``MODELS`` of ``drevalpy run`` and ``drevalpy single``
     - A zoo name, else a recipe. A zoo name wins over a recipe of the same name. The model is
       named by the string as given. A name that is neither raises ``UnknownModelError``, a
       ``ValueError`` whose message gives the reason the recipe reading failed
   * - ``drevalpy.models.combinations``, ``drevalpy run -S/-D/-P``
     - Lists of sample featurizer recipes, drug featurizer recipes and predictor names; every
       valid ``sample:drug:predictor`` is built and the others are reported. See
       :doc:`/how-to/evaluate_component_grids`
   * - The ``sample_featurizer`` and ``drug_featurizer`` keys of YAML and ``ModelConfig``
     - A featurizer part of a recipe, such as ``pca[methylation]`` or ``view[a]+view[b]``

A recipe needs no registration. To give it a short name, register it as a zoo preset; see
:doc:`extension_files`.

Model identifier
----------------

``ModelConfig.model_id`` writes the recipe of a configuration, keeping the views and the
concatenation children, so ``from_recipe(config.model_id)`` builds the same components. For
example, ``pca[gene_expression]:fingerprints:elasticNet`` has that string as its identifier, and
the zoo preset ``MultiViewNeuralNetwork`` has the identifier
``scaledGeneExpression+view[mutations]:fingerprints:neuralNetwork``. Fixed options, defaults and
search spaces are not part of a recipe. The identifier is ``None`` only for a model with a drug
featurizer and no sample featurizer, which no recipe can express.

Featurizer forms in YAML
------------------------

The ``sample_featurizer`` and ``drug_featurizer`` keys of a YAML file or a ``ModelConfig``
take one of two forms: a recipe string, such as ``scaledGeneExpression``, ``pca[methylation]``
or ``view[gene_expression]+view[methylation]``, or a mapping with these fields:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Meaning
   * - ``name``
     - Required. Registry name, or ``concatFeaturizers`` for a concatenation
   * - ``view``
     - The view to read
   * - ``featurizers``
     - Children of a ``concatFeaturizers`` node, each a recipe string or a mapping
   * - ``options``
     - Fixed constructor values, such as a gene list. An option may not name a tunable
       hyperparameter
   * - ``defaults``
     - New ``default`` values for tunable hyperparameters, such as ``{n_components: 8}``. A key
       that is not in the search space raises
   * - ``hyperparameter_space``
     - The search space for this featurizer; see :doc:`hyperparameters`

For example, a concatenation whose children carry options and defaults:

.. code-block:: yaml

   sample_featurizer:
     name: concatFeaturizers
     featurizers:
       - name: scaledGeneExpression
         options:
           gene_list: landmark_genes
       - name: pca
         view: methylation
         defaults:
           n_components: 8

``options`` and ``defaults`` keep fixed values and tunable ones apart. A tunable
hyperparameter under ``options`` raises, for example ``options of featurizer 'pca' name
tunable hyperparameters (n_components); set their value under 'defaults' instead``, and a key
under ``defaults`` that the search space lacks raises with the list of tunable keys. The
``defaults`` edit the declared ``hyperparameter_space``, or the component's own space when none
is declared, so tuning starts from them and an untuned run uses them.

The ``predictor`` key takes a predictor name or a mapping with the fields ``name``,
``defaults`` and ``hyperparameter_space``, for example
``{name: elasticNet, defaults: {alpha: 0.5}}``.

Other spellings, such as a list of featurizers or a one-key mapping like
``{"pca[methylation]": {n_components: 8}}``, raise a ``ValueError`` that shows the form to
write instead.

A model configuration has the keys ``sample_featurizer``, ``drug_featurizer``, ``predictor``
and ``prediction_mode`` (``regression`` by default). Other keys raise an error.
