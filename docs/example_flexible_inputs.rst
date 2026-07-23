.. _flexible-inputs:

Custom input with drevalpy's baselines
======================================

Configure cell-line and drug inputs through **explicit featurizer recipes** or
zoo YAML. Do not introduce a separate model class per omics modality.

Recommended: recipes and ModelConfig
------------------------------------

Use a recipe string or zoo featurizer blocks:

.. code-block:: python

    from drevalpy.models import construct_model
    from drevalpy.models.config import ModelConfig

    # Unknown / custom dense omics CSV -> raw[view]
    MyRF = construct_model(
        "MyRF",
        "raw[mynewdatamodality]:fingerprints:randomForest",
    )
    model = MyRF()
    model.build_model(model.get_default_hyperparameters())

    # Same stack via ModelConfig
    composed = ModelConfig.from_spec(
        "raw[mynewdatamodality]:fingerprints:randomForest"
    ).create_model()

.. important::
    If you do not want to write a custom loading function, this requires that
    there exists a CSV file with that name in ``{path_to_data}/{dataset_name}/``.
    I.e., if you specify ``mynewdatamodality``, you need
    ``mynewdatamodality.csv``.

Equivalent zoo YAML (``drevalpy/models/zoo`` style — **not** hpam YAML):

.. code-block:: yaml

    cell_line_featurizer:
      name: raw
      view: mynewdatamodality
    drug_featurizer: fingerprints
    predictor: randomForest

Proteomics with built-in preprocessing uses the ``normalizedProteomics``
featurizer:

.. code-block:: text

    normalizedProteomics:fingerprints:randomForest

or in zoo YAML:

.. code-block:: yaml

    cell_line_featurizer:
      name: normalizedProteomics
      hyperparameters:
        feature_threshold: 0.7
        n_features: 1000
        normalization_width: 0.3
        normalization_downshift: 1.8
    drug_featurizer: fingerprints
    predictor: randomForest

.. warning::
    Proteomics normalization must not fit on the full dataset: medians computed
    on all samples leak test information into training. The featurizer fits on
    the training split only (``fit_transform``) and applies the learned transform
    to validation/test (``transform``).

Generic CSV loading uses ``load_generic_csv`` from ``drevalpy.data.features``.
Depending on whether the featurizer is registered under the cell-line or drug
registry, the index column must be ``CELL_LINE_IDENTIFIER``
(``"cell_line_name"``) or ``DRUG_IDENTIFIER`` (``"pubchem_id"``).

You can then run it the same way as before:

.. code-block:: shell

    drevalpy --models RandomForest --dataset_name CTRPv2 --data_path data

Deprecated: flat ``cell_line_views`` / ``drug_views``
-----------------------------------------------------

Historical flat hyperparameters still accept view lists and translate them onto
featurizer configs, but this path emits a ``FutureWarning`` and will be removed
in a future release without a fixed deadline.

**Do not confuse hpam YAML with zoo YAML.** The following is experiment /
``build_model`` hyperparameter YAML (legacy), not a zoo preset:

.. code-block:: yaml

    RandomForest:
      cell_line_views:
        - mynewdatamodality
      drug_views:
        - fingerprints

Legacy Python form:

.. code-block:: python

    from drevalpy.models import construct_model

    model = construct_model("RandomForest")()
    model.build_model(
        {
            "cell_line_views": ["mynewdatamodality"],
            "drug_views": ["fingerprints"],
        }
    )

Prefer the recipe / zoo forms above. See :doc:`hyperparameter_migration` and
:doc:`model_architecture` for dotted HPO keys and full composition details.

For registering entirely new featurizers or predictors, see
:doc:`runyourmodel`.
