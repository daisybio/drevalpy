# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os
from os.path import dirname, join, abspath
from pathlib import Path
import pickle
import warnings
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, ParameterGrid

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils.utils import mkdir, preprocessing
from utils.load_data import get_train_test_set, get_cell_viab_data, get_gene_expression_data, get_morgan_fingerprints

logger = logging.getLogger(__name__)


class LinearRegression:

    def __init__(self, dataroot_drp, dataroot_feature, metric, task, feature_type, feature_selection=False,
                 selection_method=None, hyperparameters=None, nCV_folds=None, n_cpus=1):
        self.path_drp = dataroot_drp  # path to the drug response data
        self.path_feature = dataroot_feature  # path to the feature data
        self.metric = metric  # Amax, IC50, EC50, ...
        self.task = task  # LCO, LDO, LPO
        self.feature_type = feature_type  # view under cell_line_views or drug_views
        self.feature_selection = feature_selection  # whether to perform feature selection
        self.selection_method = selection_method  # method for feature selection
        self.hyperparameters = hyperparameters  # hyperparameters for the model
        self.nCV_folds = nCV_folds  # number of cross validation folds
        self.n_cpus = n_cpus  # nr of cpus to use for parallelization, relevant only for feature sel. using VST method

        self.train_drp = None  # train set
        self.test_drp = None  # test set
        self.data_dict = None  # dict containing all data needed for training and testing models
        self.metric_df = None  # dataframe with the performance metrics
        self.prediction = None  # predicted values
        self.pred_df = None  # dataframe with y_true, y_pred, target
        self.models = None  # model fit
        self.models_params = None  # model parameters

        logger.info("Reading in drug response data")
        self.drp_df = pd.read_csv(self.path_drp, header=0, index_col=0)  # load drp data
        self.drp_df.reset_index(inplace=True)

        # df containing features (not split ino training and test sets)
        logger.info(f"Reading in {self.feature_type} data")
        if self.feature_type == "gene_expression":
            self.feature_df = pd.read_csv(self.path_feature, index_col=0).T
        elif self.feature_type == "fingerprints":
            self.feature_df = pd.read_csv(self.path_feature, index_col=0)

    @property
    def cell_line_views(self):
        """
        Returns the sources the model needs as input for describing the cell line.
        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression", "mutation"]
        """
        return "gene_expression"

    @property
    def drug_views(self):
        """
        Returns the sources the model needs as input for describing the drug.
        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """
        return ["fingerprints"]

    def train(self):
        """
        Trains the model for each target in the data dictionary. in the case of LCO, single drug models are trained,
        meaning each target in the data_dict corresponds to a single drug for which a model is being generated. In
        the case of LDO, single cell line models are trained, meaning each target in the data_dict corresponds to a
        single cell line. Grid search is performed to find the best hyperparameters for each model with the number of
        cross validation folds specified by the user. Parameters to be optimized are specified by the user.
        """
        logger.info("Started training models")
        self.models = {}
        # sum_all_lengths = 0
        for target in self.data_dict:

            X_train = self.data_dict.get(target).get("X_train")  # get the training data for the target from dict
            y_train = self.data_dict.get(target).get("y_train")
            y_train = y_train.reshape(-1)
            # sum_all_lengths += len(X_train)

            reg = Lasso()  # initialize the linear regression model

            # check if CV fold is legal
            if len(X_train) == 1:
                warnings.warn("Only one sample for target {}."
                              " No Cross validation or Grid search performed.".format(target))

                model = reg  # fit the model
                model.fit(X_train, y_train)  # fit model to single sample (no CV or grid search, output is constant)

            elif len(X_train) < self.nCV_folds:
                nCV_folds = len(X_train)
                warnings.warn("Number of CV folds is larger than the number of samples. CV folds set to {}".format(
                    nCV_folds))

                # perform grid search to find the best hyperparameters
                model = GridSearchCV(reg, self.hyperparameters, cv=nCV_folds)
                model.fit(X_train, y_train)  # fit the model
            else:
                nCV_folds = self.nCV_folds

                # perform grid search to find the best hyperparameters
                model = GridSearchCV(reg, self.hyperparameters, cv=nCV_folds)
                model.fit(X_train, y_train)  # fit the model

            self.models[target] = model
        logger.info("finished training models")
        # logger.info(f"average length of training set: {sum_all_lengths / len(self.data_dict)}")

    def predict(self):

        logger.info("predicting drug response for test set")
        self.prediction = {}
        for target in self.data_dict:
            model = self.models.get(target)
            X_test = self.data_dict.get(target).get("X_test")
            yhat = model.predict(X_test)
            self.prediction[target] = yhat  # predicted responses for each target
        logger.info("finished predicting")

    def evaluate(self):

        logger.info("evaluating models")

        # initialize pandas dataframe with y_true, y_pred, target
        pred_df = pd.DataFrame({"y_true": np.concatenate([self.data_dict.get(target).get("y_test").reshape(-1)
                                                          for target in self.data_dict]),
                                "y_pred": np.concatenate([self.prediction.get(target) for target in self.data_dict]),
                                "target": np.concatenate(
                                    [np.repeat(target, len(self.data_dict.get(target).get("y_test").reshape(-1))) for
                                     target in
                                     self.data_dict])})

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
        self.metric_df = pd.DataFrame({"pcc": pcc_target, "scc": scc_target, "mse": mse_target, "rmse": rmse_target})
        self.pred_df = pred_df

        logger.info("finished evaluation")

    def get_drug_response_dataset(self):
        logger.info("preparing drug response data")
        self.train_drp, self.test_drp = get_train_test_set(self.drp_df, self.task, 0.8, self.metric)
        logger.info("finished preparing drug response data")

    def data_processing(self):
        logger.info("preprocessing drug response data")
        self.train_drp, self.test_drp = preprocessing(self.train_drp, self.test_drp, self.task, self.metric,
                                                      remove_out=True, log_transform=True)
        logger.info("finished preprocessing drug response data")

    def get_feature_dataset(self, ntop):

        if self.feature_type == "gene_expression":
            logger.info(f"preparing gene expression data - feature selection: {self.feature_selection}")
            drug_dict = get_gene_expression_data(self.feature_df, self.train_drp, self.test_drp, self.task,
                                                 feature_selection=self.feature_selection, ntop=ntop,
                                                 selection_method=self.selection_method, n_cpus=self.n_cpus)
            self.data_dict = drug_dict

        elif self.feature_type == "fingerprints":
            logger.info(f"preparing morgan fingerprints - feature selection: {self.feature_selection}")
            cl_dict = get_morgan_fingerprints(self.feature_df, self.train_drp, self.test_drp, self.task,
                                              feature_selection=self.feature_selection, ntop=ntop)
            self.data_dict = cl_dict

        logger.info("finished preparing feature data, output stored in data_dict")

    def save(self, result_path, best_model_dict):
        self.models_params = {}
        for target in best_model_dict["models"]:

            target_model = best_model_dict["models"].get(target)
            if isinstance(target_model, Lasso):
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
