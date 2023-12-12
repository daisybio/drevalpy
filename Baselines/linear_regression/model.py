# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os
from os.path import dirname, join, abspath
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils.utils import mkdir, preprocessing, get_train_test_set, get_cell_viab_data


class LinearRegression:

    def __init__(self, dataroot, metric, task, avg_by):
        self.dataroot = dataroot # path to the matrix file
        self.metric = metric # Amax, IC50, EC50, ...
        self.task = task # LCO, LDO, LPO
        self.avg_by = avg_by # avg by drug or cl

        self.train_set = None # train set with the same shape as the test set
        self.test_set = None # test set with the same shape as the train set
        self.metric_df = None # dataframe with the performance metrics
        self.predicted_means = None    # predicted means

        self.labelmatrix = pd.read_csv(self.dataroot, header=0, index_col=0)
        self.labelmatrix.reset_index(inplace=True)

    @property
    def cell_line_views(self):
        """
        Returns the sources the model needs as input for describing the cell line.
        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression", "mutation"]
        """
        return "Cell viability"

    @property
    def drug_views(self):
        """
        Returns the sources the model needs as input for describing the drug.
        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """
        return None

    def train(self, train):
        if self.task == "LCO":
            train_avg = train.mean(axis=1)

        elif self.task == "LDO":
            train_avg = train.mean()

        elif self.task == "LPO":
            if self.avg_by == "drug":
                train_avg = train.groupby('Compound')[self.metric].mean()

            elif self.avg_by == "cl":
                train_avg = train.groupby('Primary Cell Line Name')[self.metric].mean()

        self.predicted_means = pd.DataFrame(train_avg.rename("avg_" + self.metric))

        return self.predicted_means


    def evaluate(self, test):

        return self.metric_df

    def get_drug_response_dataset(self):

        self.train_set, self.test_set = get_train_test_set(self.labelmatrix,
                                                           self.task,
                                                           0.8,
                                                           self.metric)

    def get_feature_dataset(self, path, train, test):
        X_train_2, y_train_2, X_test_2, y_test_2 = get_cell_viab_data(
            path, train, test)

        return X_train_2, y_train_2, X_test_2, y_test_2


    def save_results(self, result_path):
        self.predicted_means.to_csv(result_path + self.task + self.avg_by + '_results.csv', index=True)
        self.metric_df.to_csv(result_path + self.task + '_avg_by_' + self.avg_by + '_metrics.csv', index=True)


if __name__ == "__main__":

    linear_regression = LinearRegression("/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/EC50 (µM)_matrix.csv",
                                     "EC50 (µM)", "LCO", "cl")

    linear_regression.cell_line_views
    linear_regression.drug_views

    linear_regression.get_drug_response_dataset()

    train_2, test_2 = preprocessing(linear_regression.train_set,
                                    linear_regression.test_set,
                                    linear_regression.task,
                                    linear_regression.metric, remove_out=True, log_transform=True)

    # load cell viab data
    X_train, y_train, X_test, y_test = linear_regression.get_feature_dataset(
        "~/datasets/cell_viability/CCLE/cell_viab_data.csv", train_2, test_2)

    # fit the model
    AGG_17_fit = Lasso(alpha=0.1)
    AGG_17_fit.fit(X_train, y_train)
    AGG_17_fit.coef_

    # test: predict the ec50 values for the test set
    y_pred = AGG_17_fit.predict(X_test)

    # evaluate the model
    pcc = stats.pearsonr(y_test, y_pred)[0]
    scc = stats.spearmanr(y_test, y_pred)[0]
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # save model parameters
    AGG_17_fit_dict = {"coef": AGG_17_fit.coef_, "intercept": AGG_17_fit.intercept_, "params": AGG_17_fit.get_params()}


    # pred = linear_regression.train(train_2) # train the model
    # accuracy_meas = linear_regression.evaluate(test_2) # evaluate the model

    # mkdir("results_2/")
    #
    # linear_regression.save_results("results_2/")
