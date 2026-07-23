.. _flexible-inputs:

Custom input with drevalpy's baselines
======================================

Sklearn and neural-network baselines support **flexible inputs** through zoo
presets, ``ModelConfig``, or public ``cell_line_views`` / ``drug_views``
hyperparameters passed to ``build_model()``. You do not need a separate model
class per omics modality.

Example: Flexible Inputs with DrEvalPy's Baselines
--------------------------------------------------

The sklearn baselines (``ElasticNet``, ``Lasso``, ``RandomForest``,
``GradientBoosting``, ``SVR``, ``AdaBoostDecisionTree``, ``KNNRegressor``,
``SingleDrugRandomForest``, ``SingleDrugElasticNet``,
``MultiViewRandomForest``, ``MultiViewXGBoost``) and the neural-network
baselines (``SimpleNeuralNetwork``, ``MultiViewNeuralNetwork``) accept view
overrides.

For example, to run a Random Forest on **mynewdatamodality** instead of gene
expression, pass ``cell_line_views`` in the public hyperparameter dict or zoo
YAML:

.. code-block:: yaml

    RandomForest:
      cell_line_views:
        - mynewdatamodality
      drug_views:
        - fingerprints

.. important::
    If you do not want to write a custom loading function, this requires that
    there exists a CSV file with that name in ``{path_to_data}/{dataset_name}/``.
    I.e., if you specify ``mynewdatamodality``, you need
    ``mynewdatamodality.csv``.

The same override can be expressed as a featurizer recipe or ``ModelConfig``:

.. code-block:: python

    from drevalpy.models import MODEL_FACTORY, construct_model
    from drevalpy.models.config import ModelConfig

    # Flat hyperparameter override on a zoo factory entry
    model = MODEL_FACTORY["RandomForest"]()
    model.build_model(
        {
            "cell_line_views": ["mynewdatamodality"],
            "drug_views": ["fingerprints"],
        }
    )

    # Or compose explicitly (unknown views map to raw[view])
    MyRF = construct_model(
        "MyRF",
        "raw[mynewdatamodality]:fingerprints:randomForest",
    )
    composed = ModelConfig.from_spec(
        "raw[mynewdatamodality]:fingerprints:randomForest"
    ).create_model()

Internally, view names are translated onto featurizer configs (see
``drevalpy.models.featurizer_mapping``). Unknown cell-line views become
``raw[view]`` and are loaded via ``load_generic_csv`` from
``drevalpy.data.features``:

.. code-block:: python

    def load_generic_csv(path: str, dataset_name: str, feature_name: str, index_col=CELL_LINE_IDENTIFIER) -> FeatureDataset:
        """
        Loads a generic CSV file with cell line IDs as index and features as columns.

        :param path: path to the data, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC2
        :param feature_name: name of the feature, e.g., gene_expression
        :param index_col: name of the index column, e.g., cell_line_id
        :returns: FeatureDataset with the features
        """
        feature_csv = pd.read_csv(f"{path}/{dataset_name}/{feature_name}.csv", index_col=index_col)
        feature_csv.index = feature_csv.index.astype(str)
        if "cellosaurus_id" in feature_csv.columns:
            feature_csv = feature_csv.drop(columns=["cellosaurus_id"])
        return FeatureDataset(features=iterate_features(df=feature_csv, feature_type=feature_name))

Depending on whether you define it in ``cell_line_views`` or ``drug_views``,
the index column must be ``CELL_LINE_IDENTIFIER`` (``"cell_line_name"``) or
``DRUG_IDENTIFIER`` (``"pubchem_id"``).

You can then run it the same way as before:

.. code-block:: shell

    drevalpy --models RandomForest --dataset_name CTRPv2 --data_path data

Example: Proteomics With Built-in Preprocessing
-----------------------------------------------

For ``proteomics``, the ``normalizedProteomics`` featurizer applies
median-centering and imputation. Expose the related options as public
hyperparameters (or under the featurizer block in zoo YAML):

.. code-block:: yaml

    RandomForest:
      cell_line_views:
        - proteomics
      drug_views:
        - fingerprints
      proteomics_feature_threshold: 0.7
      proteomics_n_features: 1000
      proteomics_normalization_width: 0.3
      proteomics_normalization_downshift: 1.8

Equivalent recipe form:

.. code-block:: text

    normalizedProteomics:fingerprints:randomForest

.. warning::
    Proteomics normalization must not fit on the full dataset: medians computed
    on all samples leak test information into training. The featurizer fits on
    the training split only (``fit_transform``) and applies the learned transform
    to validation/test (``transform``).

For registering entirely new featurizers or predictors, see
:doc:`runyourmodel` and :doc:`model_architecture`.
