Custom input with drevalpy's baselines
======================================

These example use cases are about how to use your own custom input for baseline models implemented in drevalpy.

Example: Flexible Inputs with DrEvalPy's Baselines
-------------------------------------------------------------

The sklearn baseline models (``ElasticNet``, ``Lasso``, ``RandomForest``, ``GradientBoosting``, ``SVR``, ``AdaBoostDecisionTree``, ``KNNRegressor``,
``SingleDrugRandomForest``, ``SingleDrugElasticNet``, ``MultiViewRandomForest``, ``MultiViewXGBoost``) and the neural network baselines (``SimpleNeuralNetwork``, ``MultiViewNeuralNetwork``)
support **flexible inputs**. Instead of writing a new Python class for each omic data type, you can simply change which omic the model uses by editing ``hyperparameters.yaml``.

For example, to run a Random Forest on **mynewdatamodality** data instead of gene expression, change the
``cell_line_views`` in ``models/baselines/hyperparameters.yaml``:

.. code-block:: yaml

    RandomForest:
      cell_line_views:
        - mynewdatamodality
      drug_views:
        - fingerprints
      ..

.. important::
    If you do not want to write a custom loading function, this requires that there exists a csv file with that name in
    ``{path_to_data}/{dataset_name}/``. I.e., if you specify ``mynewdatamodality``, you need to have a ``mynewdatamodality.csv`` file.


The data is then loaded by the ``load_generic_csv`` function:

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

Depending on whether you define it in ``cell_line_views`` or ``drug_views``, the index column will have to be the
``CELL_LINE_IDENTIFIER`` ("cell_line_name") or the ``DRUG_IDENTIFIER`` ("pubchem_id").


You can then run it the same way as before:

.. code-block:: shell

    drevalpy --models RandomForest --dataset_name CTRPv2 --data_path data

For more details on the flexible input system, see the sklearn :ref:`flexible-inputs` documentation
and the SimpleNeuralNetwork :ref:`flexible-inputs-simplenn` documentation.

Example: Using Flexible Inputs with Sklearn Baselines With Custom Preprocessing
---------------------------------------------------------------------------------

For the ``proteomics`` input, we implemented custom preprocessing in the ``SklearnModel`` parent class and for that,
we define custom hyperparameters in ``hyperparameters.yaml``:

.. code-block:: yaml

    RandomForest:
      cell_line_views:
        - proteomics
      drug_views:
        - fingerprints
      ...
      proteomics_feature_threshold:
        - 0.7
      proteomics_n_features:
        - 1000
      proteomics_normalization_width:
        - 0.3
      proteomics_normalization_downshift:
        - 1.8

We add these parameters to the ``SklearnModel`` init method:

.. code-block:: python

        def __init__(self):
            # ... existing init method
            # proteomics-specific defaults
            self.proteomics_transformer = None
            self.proteomics_feature_threshold = 0.7
            self.proteomics_n_features = 1000
            self.proteomics_normalization_width = 0.3
            self.proteomics_normalization_downshift = 1.8

These parameters are filled with the parameters from the hyperparameter file in the ``build_model`` method:

.. code-block:: python

        def build_model(self, hyperparameters: dict):
            # ... existing build_model method
            # proteomics features are not supported for all models
            if "proteomics" in self.cell_line_views:
                self._init_proteomics_features(hyperparameters)

            def _init_proteomics_features(self, hyperparameters: dict):
                self.proteomics_feature_threshold = hyperparameters.get("proteomics_feature_threshold", 0.7)
                self.proteomics_n_features = hyperparameters.get("proteomics_n_features", 1000)
                self.proteomics_normalization_width = hyperparameters.get("proteomics_normalization_width", 0.3)
                self.proteomics_normalization_downshift = hyperparameters.get("proteomics_normalization_downshift", 1.8)
                self.proteomics_transformer = ProteomicsMedianCenterAndImputeTransformer(
                    feature_threshold=self.proteomics_feature_threshold,
                    n_features=self.proteomics_n_features,
                    normalization_downshift=self.proteomics_normalization_downshift,
                    normalization_width=self.proteomics_normalization_width,
                )

We want to normalize the proteomics data with a custom method which we implement in ``ProteomicsMedianCenterAndImputeTransformer`` (code see below).

.. warning::
    This can't be done before training because it will compute medians. If the medians are computed on the whole dataset,
    the test set's medians are **leaked** into the training set. The correct way to handle this is to compute the median
    on the training set only (fit_transform function) and then only apply the median to the validation and test set (transform function).

In the training function, we call our custom preprocessing function:

.. code-block:: python

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        # [...]
        if len(output) > 0:
            if "gene_expression" in self.cell_line_views:
                cell_line_input = scale_gene_expression(
                    cell_line_input=cell_line_input,
                    cell_line_ids=np.unique(output.cell_line_ids),
                    training=True,
                    gene_expression_scaler=self.gene_expression_scaler,
                )
            elif "proteomics" in self.cell_line_views:
                cell_line_input = prepare_proteomics(
                    cell_line_input=cell_line_input,
                    cell_line_ids=np.unique(output.cell_line_ids),
                    training=True,
                    transformer=self.proteomics_transformer,
                )
            # [...]

In the predict function, too:

.. code-block:: python

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        # [...]
        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                gene_expression_scaler=self.gene_expression_scaler,
            )
        elif "proteomics" in self.cell_line_views:
            cell_line_input = prepare_proteomics(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                transformer=self.proteomics_transformer,
            )
        # [...]

Utility functions:

.. code-block:: python

    class ProteomicsMedianCenterAndImputeTransformer(BaseEstimator, TransformerMixin):
    """Performs median centering and imputation of proteomics data."""

        def __init__(self, feature_threshold=0.7, n_features=1000, normalization_downshift=1.8, normalization_width=0.3):
            """
            Hyperparameters for the normalization.

            :param feature_threshold: Require that, e.g., 70% of the proteins are measured without NAs
                over all cell lines -> n_complete_features = number of proteins with at least 70% of the cell lines
            :param n_features: fallback for feature selection. Take top n complete features.
                Select max(n_complete_features, n_features) features.
            :param normalization_downshift: downshift factor for the mean
            :param normalization_width: width factor for the standard deviation
            """
            self.feature_threshold = feature_threshold
            self.n_features = n_features
            self.normalization_downshift = normalization_downshift
            self.normalization_width = normalization_width
            self.protein_indices = np.array([])
            self.mean_median = 0

        def fit(self, X, y=None):
            """
            Learns the top n_feature complete proteins and calculates the mean median of the train cell lines.

            :param X: input proteomics data
            :param y: not used
            :returns: self
            """
            required_proteins = int(X.shape[0] * self.feature_threshold)
            # identify the complete columns
            completeness = np.sum(~np.isnan(X), axis=0)
            n_complete_features = np.count_nonzero(completeness >= required_proteins)
            if n_complete_features < self.n_features:
                # select top 1000 complete features
                # sort by completeness
                sorted_indices = np.argsort(completeness)[::-1]
                self.protein_indices = sorted_indices[: self.n_features]
            else:
                # select the features meeting the required threshold
                self.protein_indices = np.where(completeness >= required_proteins)[0]
            X = X[:, self.protein_indices]
            # calculate mean of sample medians
            medians = np.nanmedian(X, axis=1)
            self.mean_median = np.nanmean(medians)
            return self

        def transform(self, X):
            """
            Median center the data and impute missing values with downshifted normal distribution.

            :param X: input proteomics data
            :returns: transformed proteomics data
            """
            X = X[0]

            X = X[self.protein_indices]

            correction_factor = self.mean_median / np.nanmedian(X)
            X = X * correction_factor
            # downshifted mean
            np.random.seed(seed=100)
            cell_line_mean = np.nanmean(X)
            cell_line_sd = np.nanstd(X)
            downshifted_mean = cell_line_mean - (self.normalization_downshift * cell_line_sd)
            shrinked_sd = self.normalization_width * cell_line_sd
            n_missing = np.count_nonzero(np.isnan(X))
            X[np.isnan(X)] = np.random.normal(loc=downshifted_mean, scale=shrinked_sd, size=n_missing)
            return [X]

    def prepare_proteomics(
        cell_line_input: FeatureDataset,
        cell_line_ids: np.ndarray,
        training: bool,
        transformer: ProteomicsMedianCenterAndImputeTransformer,
    ) -> FeatureDataset:
        """
        Applies log10 transform and proteomics normalization (centering + imputation) to proteomics view.

        :param cell_line_input: FeatureDataset with proteomics features
        :param cell_line_ids: cell line IDs for training or transformation
        :param training: whether to fit or only transform
        :param transformer: Proteomics transformer
        :returns: transformed FeatureDataset
        """
        cell_line_input = cell_line_input.copy()
        cell_line_input.apply(log10_and_set_na, view="proteomics")
        if training:
            cell_line_input.fit_transform_features(
                train_ids=cell_line_ids,
                transformer=transformer,
                view="proteomics",
            )
        else:
            cell_line_input.transform_features(
                ids=cell_line_ids,
                transformer=transformer,
                view="proteomics",
            )
        return cell_line_input
