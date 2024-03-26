# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import pickle
import sys
import warnings
from os.path import dirname, join, abspath
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from base_model import BaseModel
from utils.utils import oversampling, cross_validation_fit

logger = logging.getLogger(__name__)


class GradientBoostClassifier(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probability = None
        self.kernel = "linear"

    @property
    def cell_line_views(self):
        """
        Returns the sources the model needs as input for describing the cell line.
        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression", "mutation"]
        """
        return ["gene_expression"]

    @property
    def drug_views(self):
        """
        Returns the sources the model needs as input for describing the drug.
        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """
        return ["fingerprints"]

    def train(self):
        """
        Trains the model for each target in the data dictionary as explained in base_model.py. The difference here is
        that the model is a gradient boosting model and the hyperparameters are tuned using grid search. Additionally
        oversampling can be performed to account for class imbalance by specifing the method in the config file.
        """
        logger.info("Started training models")
        self.models = {}
        # sum_all_lengths = 0
        for target in self.data_dict:

            X_train = self.data_dict.get(target).get("X_train")  # get the training data for the target from dict
            y_train = self.data_dict.get(target).get("y_train")
            y_train = y_train.reshape(-1)

            # account for class imbalance using an over sampling technique
            if self.oversampling_method != "None":
                X_train, y_train = oversampling(X_train, y_train, self.oversampling_method, self.n_cpus)
                self.data_dict.get(target)["X_train"] = X_train
                self.data_dict.get(target)["y_train"] = y_train

            clf = GradientBoostingClassifier(validation_fraction=0.1, n_iter_no_change=10) # initialize classifier

            # hyperparameter tuning using grid search cross validation
            model = cross_validation_fit(X_train, y_train, clf, self.nCV_folds, self.hyperparameters)

            self.models[target] = model

        logger.info("finished training models")

    def predict(self):
        """
        Predicts the drug response for the test set as explained in base_model.py. The difference here is that the model
        is a gradient boosting model and the probability of the positive class is also computed (needed for the
        evaluation and confusion matrix calculation).
        """
        logger.info("predicting drug response for test set")
        self.prediction = {}
        self.probability = {}
        for target in self.data_dict:
            model = self.models.get(target)
            X_test = self.data_dict.get(target).get("X_test")
            yhat = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,
                     1]  # since we are interested in the probability of the positive class
            self.prediction[target] = yhat  # predicted responses for each target
            self.probability[target] = y_prob
        logger.info("finished predicting")

    # logger.info(f"average length of training set: {sum_all_lengths / len(self.data_dict)}")

    def evaluate(self):
        """
        Evaluates the model by computing metrics such as confusion matrix, sensitivity, specificity, precision, accuracy,
        balanced accuracy, F1-score, ROC-auc and MCC as explained in base_model.py. The difference here is that the model
        is a gradient boosted classifier and the probability of the positive class is also computed (needed for the
        evaluation and confusion matrix calculation).
        """
        logger.info("evaluating models")

        # initialize pandas dataframe with y_true, y_pred, target
        pred_df = pd.DataFrame({"y_true": np.concatenate([self.data_dict.get(target).get("y_test").reshape(-1)
                                                          for target in self.data_dict]),
                                "y_pred": np.concatenate([self.prediction.get(target) for target in self.data_dict]),
                                "y_prob": np.concatenate([self.probability.get(target) for target in self.data_dict]),
                                "sample_id": np.concatenate(
                                    [self.data_dict.get(target).get("test_sample_ids") for target in self.data_dict]),
                                "target": np.concatenate(
                                    [np.repeat(target, len(self.data_dict.get(target).get("y_test").reshape(-1))) for
                                     target in self.data_dict])})

        # kick out all rows that have only one sample (target-wise) as pcc/scc needs more than one sample
        pred_df = pred_df.groupby("target").filter(lambda x: len(x) > 1)

        # filter out models with only one class -> no auc can be calculated
        pred_df = pred_df.groupby("target").filter(lambda x: x["y_true"].nunique() > 1)
        grouped = pred_df.groupby("target")

        # compute confusion matrix for each target
        confusion = pred_df.groupby("target").apply(
            lambda x: np.round(confusion_matrix(x["y_true"], x["y_pred"], normalize='all'), 3).ravel())
        confusion_df = pd.DataFrame.from_records(confusion.values, index=confusion.index,
                                                 columns=["TN", "FP", "FN", "TP"])

        # calculating classification metrices of interest
        confusion_df["sensitivity"] = grouped.apply(lambda x: recall_score(x["y_true"], x["y_pred"]))
        confusion_df["specificity"] = np.round(confusion_df["TN"] / (confusion_df["TN"] + confusion_df["FP"]), 3)
        confusion_df["precision"] = grouped.apply(lambda x: precision_score(x["y_true"], x["y_pred"]))
        confusion_df["accuracy"] = grouped.apply(lambda x: accuracy_score(x["y_true"], x["y_pred"]))
        confusion_df["balanced accuracy"] = grouped.apply(
            lambda x: balanced_accuracy_score(x["y_true"], x["y_pred"]))
        confusion_df["F1-score"] = grouped.apply(lambda x: f1_score(x["y_true"], x["y_pred"]))
        confusion_df["ROC-auc"] = grouped.apply(lambda x: roc_auc_score(x["y_true"], x["y_prob"]))
        confusion_df["MCC"] = grouped.apply(lambda x: matthews_corrcoef(x["y_true"], x["y_pred"]))

        # map performance metrics to targets in pred df - important for plotting later
        mapped_values = pred_df.apply(lambda row: confusion_df.loc[row["target"]], axis=1)
        pred_df = pred_df.join(mapped_values)

        self.metric_df = confusion_df
        self.pred_df = pred_df

        logger.info("finished evaluation")

    def save(self, result_path, best_model_dict):
        """
        Saves the model parameters and the accuracy metrics to the result path as explained in the base_model.py.
        """
        self.models_params = {}
        for target in best_model_dict["models"]:

            target_model = best_model_dict["models"].get(target)
            if isinstance(target_model, GradientBoostingClassifier):
                self.models_params[target] = {"gini_impurity": target_model.feature_importances_}
            else:
                # if model is gridsearchcv, save the best estimator params
                self.models_params[target] = {"gini_impurity": target_model.best_estimator_.feature_importances_}
        # save model params dict as pickle
        with open(result_path + self.task + '_model_params.pkl', 'wb') as f:
            pickle.dump(self.models_params, f)

        # save accuracy metrics as csv
        best_model_dict["metric_df"].to_csv(result_path + self.task + '_metrics.csv', index=True)
        logger.info("saved model parameters as pickle and accuracy metrics as csv")
