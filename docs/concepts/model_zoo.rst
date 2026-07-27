Model zoo
=========

The model zoo is a set of named presets under ``drevalpy/models/zoo/*.yaml``.
Each preset wires cell-line featurizers, drug featurizers, and a predictor into
a runnable architecture. CLI and Python both refer to models by these preset
names (for example ``ElasticNet`` or ``DIPK``).

Zoo inputs define architecture, not HPO search dimensions. Tuning searches
predictor (and tunable featurizer) hyperparameters on top of a fixed recipe.

Multi-drug models
-----------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Name
     - Description
   * - AdaBoostDecisionTree
     - Sklearn AdaBoost regressor with decision-tree base learners
   * - DIPK
     - Prior-knowledge DNN integrating gene networks, expression, and MolGNet drug topology
   * - DrugGNN
     - Graph neural net over drug structure plus MLP over gene expression
   * - ElasticNet
     - Sklearn Elastic Net on concatenated cell-line and drug features
   * - GradientBoosting
     - Sklearn histogram gradient-boosting regressor
   * - KNNRegressor
     - Sklearn k-nearest neighbors regressor
   * - Lasso
     - Sklearn Lasso regressor
   * - MultiViewLightGBM
     - LightGBM over multiple cell-line views plus drug fingerprints
   * - MultiViewNeuralNetwork
     - Feed-forward network on concatenated multi-omics and drug features
   * - MultiViewRandomForest
     - Random forest on multiple cell-line views plus fingerprints
   * - MultiViewXGBoost
     - XGBoost over one or more cell-line views plus fingerprints
   * - NaiveCellLineMeanPredictor
     - Predicts the training mean response of each cell line
   * - NaiveDrugMeanPredictor
     - Predicts the training mean response of each drug
   * - NaiveMeanEffectsPredictor
     - ANOVA-style mean-effects baseline (dataset, tissue when available, cell line, drug)
   * - NaivePredictor
     - Predicts the global training-set mean response
   * - NaiveTissueDrugMeanPredictor
     - Predicts the training mean per tissue–drug pair
   * - NaiveTissueMeanPredictor
     - Predicts the training mean response of each tissue
   * - PharmaFormer
     - Transformer over byte-pair encoded SMILES with gene expression
   * - Precily
     - GSVA pathway scores with SMILESVec drug embeddings
   * - RandomForest
     - Sklearn random forest regressor
   * - Ridge
     - Sklearn ridge regressor
   * - SRMF
     - Similarity-regularized matrix factorization
   * - SVR
     - Sklearn support vector regressor
   * - SimpleNeuralNetwork
     - Small feed-forward network on concatenated cell-line and drug features
   * - SparseGO
     - GO-structured visible neural network

Single-drug models
------------------

Single-drug presets fit one model per drug (no shared drug encoder at
prediction time for held-out compounds in the usual LDO sense).

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Name
     - Description
   * - MOLIR
     - Regression adaption of MOLI (multi-omics late integration with triplet-style training)
   * - SingleDrugElasticNet
     - Per-drug Elastic Net on cell-line features
   * - SingleDrugRandomForest
     - Per-drug random forest on cell-line features
   * - SuperFELTR
     - Regression adaption of Super.FELT (supervised feature extraction with triplet loss)

Further reading
---------------

- :doc:`/python/models` — constructing and migrating models in Python
- :doc:`/python/component_catalog` — predictors and featurizers behind the zoo
- :doc:`/cli/experiment` — selecting zoo names in CLI experiments

Backward compatibility
----------------------

Deprecated but still supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, factory dictionaries such as ``MODEL_FACTORY`` were a common
lookup path. This remains available for backward compatibility, but is
deprecated and may be removed in a future release. Prefer zoo presets and
the constructors documented in :doc:`/python/models`.

No longer supported
~~~~~~~~~~~~~~~~~~~

Deep imports such as ``drevalpy.models.baselines.*`` or
``drevalpy.models.DIPK.*`` no longer resolve. Use zoo names and the public
``drevalpy.models`` exports instead.
