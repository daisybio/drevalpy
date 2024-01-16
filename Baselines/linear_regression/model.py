# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os
from os.path import dirname, join, abspath
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import pickle

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils.utils import mkdir, preprocessing
from utils.load_data import get_train_test_set, get_cell_viab_data, get_gene_expression_data, get_morgan_fingerprints


class LinearRegression:

    def __init__(self, dataroot_drp, dataroot_feature, metric, task, feature_type):
        self.path_drp = dataroot_drp  # path to the drug response data
        self.path_feature = dataroot_feature  # path to the feature data
        self.metric = metric  # Amax, IC50, EC50, ...
        self.task = task  # LCO, LDO, LPO
        self.feature_type = feature_type  # view under cell_line_views or drug_views

        self.train_drp = None  # train set
        self.test_drp = None  # test set
        self.data_dict = None  # dict containing all data needed for training and testing models
        self.metric_df = None  # dataframe with the performance metrics
        self.prediction = None  # predicted values
        self.models = None  # model fit
        self.models_params = None  # model parameters

        self.drp_df = pd.read_csv(self.path_drp, header=0, index_col=0)  # load drp data
        self.drp_df.reset_index(inplace=True)

        # df containing features (not split ino training and test sets)
        if self.feature_type == "gene_expression":
            self.feature_df = pd.read_csv(self.path_feature, sep="\t", index_col=0).T
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

    def train(self, regularization_strength=0.1):
        """
        Trains the model for each target in the data dictionary. in the case of LCO, single drug models are trained,
        meaning each target in the data_dict corresponds to a single drug for which a model is being generated. In
        the case of LDO, single cell line models are trained, meaning each target in the data_dict corresponds to a
        single cell line.
        """
        self.models = {}
        for target in self.data_dict:
            X_train = self.data_dict.get(target).get("X_train")  # get the training data for the target from dict
            y_train = self.data_dict.get(target).get("y_train")

            target_fit = Lasso(alpha=regularization_strength)
            target_fit.fit(X_train, y_train)  # fit the model
            self.models[target] = target_fit

    def predict(self):

        self.prediction = {}
        for target in self.data_dict:
            model = self.models.get(target)
            X_test = self.data_dict.get(target).get("X_test")
            yhat = model.predict(X_test)
            self.prediction[target] = yhat  # predicted responses for each target

    def evaluate(self):

        pcc_ls = []
        scc_ls = []
        mse_ls = []
        rmse_ls = []
        cls = []

        for target in self.data_dict:
            #  skip targets with only one sample as scc can only be calculated with at least two samples
            #  e.g. for LCO, skip drugs with only one cell line
            if len(self.data_dict.get(target).get("y_test").reshape(-1)) <= 1:
                continue

            pcc = stats.pearsonr(self.data_dict.get(target).get("y_test").reshape(-1), self.prediction.get(target))[0]
            scc = stats.spearmanr(self.data_dict.get(target).get("y_test").reshape(-1), self.prediction.get(target))[0]
            mse = mean_squared_error(self.data_dict.get(target).get("y_test").reshape(-1), self.prediction.get(target))
            rmse = mean_squared_error(self.data_dict.get(target).get("y_test").reshape(-1), self.prediction.get(target),
                                      squared=False)

            pcc_ls.append(pcc)
            scc_ls.append(scc)
            mse_ls.append(mse)
            rmse_ls.append(rmse)
            cls.append(target)

        self.metric_df = pd.DataFrame({"pcc": pcc_ls, "scc": scc_ls, "mse": mse_ls, "rmse": rmse_ls},
                                      index=cls)

    def get_drug_response_dataset(self):
        self.train_drp, self.test_drp = get_train_test_set(self.drp_df, self.task, 0.8, self.metric)

    def data_processing(self):
        self.train_drp, self.test_drp = preprocessing(self.train_drp, self.test_drp, self.task, self.metric,
                                                      remove_out=True, log_transform=True)

    def get_feature_dataset(self, feature_selection=False, selection_method=None):

        if self.task == "LCO":
            if self.feature_type == "gene_expression":
                drug_dict = get_gene_expression_data(self.feature_df, self.train_drp, self.test_drp,
                                                     feature_selection=feature_selection,
                                                     selection_method=selection_method)
                self.data_dict = drug_dict

        if self.task == "LDO":
            if self.feature_type == "fingerprints":
                cl_dict = get_morgan_fingerprints(self.feature_df, self.train_drp, self.test_drp,
                                                  feature_selection=feature_selection)
                self.data_dict = cl_dict

    def save(self, result_path):
        self.models_params = {}
        for target in self.models:
            self.models_params[target] = {"coef": self.models.get(target).coef_,
                                          "intercept": self.models.get(target).intercept_,
                                          "params": self.models.get(target).get_params()}

        # save model params dict as pickle
        with open(result_path + self.task + '_model_params.pkl', 'wb') as f:
            pickle.dump(self.models_params, f)

        # save accuracy metrics as csv
        self.metric_df.to_csv(result_path + self.task + '_metrics.csv', index=True)


if __name__ == "__main__":
    # meta_data = {"dataroot_drp": "/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/EC50 ("
    #                              "µM)_matrix_cellosaurusID_intersection.csv",
    #              "dataroot_feature": "/nfs/home/students/m.lorenz/datasets/transcriptomics/CCLE/salmon.merged"
    #                                  ".gene_counts.cellosaurusID.intersection.tsv",
    #              "metric": "EC50 (µM)",
    #              "task": "LCO",
    #              "feature_type": "gene_expression"}

    meta_data = {"dataroot_drp": "/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/EC50 ("
                                 "µM)_matrix_cellosaurusID_intersection.csv",
                 "dataroot_feature": "/nfs/home/students/m.lorenz/datasets/compounds/c_morganfp.csv",
                 "metric": "EC50 (µM)",
                 "task": "LDO",
                 "feature_type": "fingerprints"}

    linear_regression = LinearRegression(meta_data.get("dataroot_drp"), meta_data.get("dataroot_feature"),
                                         meta_data.get("metric"), meta_data.get("task"), meta_data.get("feature_type"))

    linear_regression.cell_line_views
    linear_regression.drug_views

    linear_regression.get_drug_response_dataset()
    breakpoint()
    linear_regression.data_processing()

    # load cell viab/transcriptomic data doesn't matter, as long as cl names are the same as in the drug response data
    linear_regression.get_feature_dataset(feature_selection=True, selection_method="VST")

    # fit the model
    linear_regression.train(0.1)

    # predict the ec50 values for the test set
    linear_regression.predict()

    # evaluate the model
    linear_regression.evaluate()
    linear_regression.metric_df["scc"].mean()

    """
    # save model parameters and results
    dir_path = "results_transcriptomics/"
    mkdir(dir_path)
    linear_regression.save(dir_path)"""
