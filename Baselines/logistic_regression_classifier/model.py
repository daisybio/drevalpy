# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import pickle
import sys
import warnings
from os.path import dirname, join, abspath
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from base_model import BaseModel
from utils.utils import oversampling

logger = logging.getLogger(__name__)


class LogisticClassifier(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probability = None

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
        that the model is a logistic regression model and the hyperparameters are tuned using grid search. Additionally
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
            # if at least one unique class has less than or equal to  5 samples (i.e. <= to number of k neighbours set
            # in oversampling method), no oversampling is performed
            ovrs = sum(np.unique(y_train, return_counts=True)[1] <= 5) == 0
            if self.oversampling_method is not None and ovrs:
                X_train, y_train = oversampling(X_train, y_train, self.oversampling_method, self.n_cpus)
                self.data_dict.get(target)["X_train"] = X_train
                self.data_dict.get(target)["y_train"] = y_train

            clf = LogisticRegression()  # initialize classifier

            # check if CV fold is legal
            if len(X_train) == 1:
                warnings.warn("Only one sample for target {}."
                              " No Cross validation or Grid search performed.".format(target))

                model = clf  # fit the model
                model.fit(X_train, y_train)  # fit model to single sample (no CV or grid search, output is constant)

            elif len(X_train) < self.nCV_folds:
                nCV_folds = len(X_train)
                warnings.warn("Number of CV folds is larger than the number of samples. CV folds set to {}".format(
                    nCV_folds))

                # perform grid search to find the best hyperparameters
                model = GridSearchCV(clf, self.hyperparameters, cv=nCV_folds)
                model.fit(X_train, y_train)  # fit the model
            else:
                nCV_folds = self.nCV_folds

                # perform grid search to find the best hyperparameters
                model = GridSearchCV(clf, self.hyperparameters, cv=nCV_folds)
                model.fit(X_train, y_train)  # fit the model

            self.models[target] = model
        logger.info("finished training models")

    def predict(self):
        """
        Predicts the drug response for the test set as explained in base_model.py. The difference here is that the model
        is a logistic regression model and the probability of the positive class is also computed (needed for the
        evaluation and confusion matrix calculation).
        """
        logger.info("predicting drug response for test set")
        self.prediction = {}
        self.probability = {}
        for target in self.data_dict:
            model = self.models.get(target)
            X_test = self.data_dict.get(target).get("X_test")
            yhat = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] # since we are interested in the probability of the positive class
            self.prediction[target] = yhat  # predicted responses for each target
            self.probability[target] = y_prob
        logger.info("finished predicting")
    # logger.info(f"average length of training set: {sum_all_lengths / len(self.data_dict)}")

    def evaluate(self):
        """
        Evaluates the model by computing the pearson correlation coefficient, spearman correlation coefficient, mean
        squared error and root mean squared error for each target. The difference here is that probability of the
        positive class is also computed and saved in the prediction dataframe.
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

        #  also skip target where all predictions are the same, leading to a constant -> pcc/scc not calculated
        pred_df = pred_df.groupby("target").filter(lambda x: x["y_pred"].nunique() > 1)

        # compute the target-wise pcc, scc, mse, rmse and put it in self.metric_df
        pcc_target = pred_df.groupby("target").apply(lambda x: stats.pearsonr(x["y_true"], x["y_pred"])[0])
        scc_target = pred_df.groupby("target").apply(lambda x: stats.spearmanr(x["y_true"], x["y_pred"])[0])
        mse_target = pred_df.groupby("target").apply(lambda x: mean_squared_error(x["y_true"], x["y_pred"]))
        rmse_target = pred_df.groupby("target").apply(
            lambda x: mean_squared_error(x["y_true"], x["y_pred"], squared=False))

        # map performance metrics to targets in pred df - important for plotting later
        pred_df["pcc"] = pred_df["target"].map(pcc_target)
        pred_df["scc"] = pred_df["target"].map(scc_target)
        pred_df["mse"] = pred_df["target"].map(mse_target)
        pred_df["rmse"] = pred_df["target"].map(rmse_target)

        self.metric_df = pd.DataFrame({"pcc": pcc_target, "scc": scc_target, "mse": mse_target, "rmse": rmse_target})
        self.pred_df = pred_df

        logger.info("finished evaluation")

    def save(self, result_path, best_model_dict):
        """
        Saves the model parameters and the accuracy metrics to the result path as explained in the base_model.py. The
        difference here is that the model is a logisitc regression model.
        """
        self.models_params = {}
        for target in best_model_dict["models"]:

            target_model = best_model_dict["models"].get(target)
            if isinstance(target_model, LogisticRegression):
                self.models_params[target] = {"coef": target_model.coef_,
                                              "intercept": target_model.intercept_}
            else:
                # if model is gridsearchcv, save the best estimator params
                self.models_params[target] = {"coef": target_model.best_estimator_.coef_,
                                              "intercept": target_model.best_estimator_.intercept_}
        # save model params dict as pickle
        with open(result_path + self.task + '_model_params.pkl', 'wb') as f:
            pickle.dump(self.models_params, f)

        # save accuracy metrics as csv
        best_model_dict["metric_df"].to_csv(result_path + self.task + '_metrics.csv', index=True)
        logger.info("saved model parameters as pickle and accuracy metrics as csv")
