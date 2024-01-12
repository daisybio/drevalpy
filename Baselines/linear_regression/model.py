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
from utils.load_data import get_train_test_set, get_cell_viab_data, get_gene_expression_data


class LinearRegression:

    def __init__(self, dataroot, metric, task):
        self.dataroot = dataroot  # path to the matrix file
        self.metric = metric  # Amax, IC50, EC50, ...
        self.task = task  # LCO, LDO, LPO

        self.train_drp = None  # train set
        self.test_drp = None  # test set
        self.metric_df = None  # dataframe with the performance metrics
        self.prediction = None  # predicted values
        self.models = None  # model fit
        self.models_params = None  # model parameters

        self.labelmatrix = pd.read_csv(self.dataroot, header=0, index_col=0)
        self.labelmatrix.reset_index(inplace=True)

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

    def train(self, data_dict, regularization_strength=0.1):

        self.models = {}
        for drug in data_dict:
            X_train = data_dict.get(drug).get("X_train")  # get the training data for the drug from dict
            y_train = data_dict.get(drug).get("y_train")

            drug_fit = Lasso(alpha=regularization_strength)
            drug_fit.fit(X_train, y_train)  # fit the model
            self.models[drug] = drug_fit

    def predict(self, data_dict):

        self.prediction = {}
        for drug in data_dict:
            model = self.models.get(drug)
            X_test = data_dict.get(drug).get("X_test")
            yhat = model.predict(X_test)
            self.prediction[drug] = yhat  # predicted responses for each drug

    def evaluate(self, data_dict):

        pcc_ls = []
        scc_ls = []
        mse_ls = []
        rmse_ls = []

        for drug in data_dict:
            pcc = stats.pearsonr(data_dict.get(drug).get("y_test").reshape(-1), self.prediction.get(drug))[0]
            scc = stats.spearmanr(data_dict.get(drug).get("y_test").reshape(-1), self.prediction.get(drug))[0]
            mse = mean_squared_error(data_dict.get(drug).get("y_test").reshape(-1), self.prediction.get(drug))
            rmse = mean_squared_error(data_dict.get(drug).get("y_test").reshape(-1), self.prediction.get(drug),
                                      squared=False)

            pcc_ls.append(pcc)
            scc_ls.append(scc)
            mse_ls.append(mse)
            rmse_ls.append(rmse)

        self.metric_df = pd.DataFrame({"pcc": pcc_ls, "scc": scc_ls, "mse": mse_ls, "rmse": rmse_ls},
                                      index=list(data_dict.keys()))

    def get_drug_response_dataset(self):
        self.train_drp, self.test_drp = get_train_test_set(self.labelmatrix, self.task, 0.8, self.metric)

    def get_feature_dataset(self, path, train_drp, test_drp, feature):

        if self.task == "LCO":

            if feature == "cell_viab":
                (X_train_drp, y_train_feature,
                 X_test_drp, y_test_feature) = get_cell_viab_data(path, train_drp, test_drp)

            elif feature == "gene_expression":
                (X_train_drp, y_train_feature,
                 X_test_drp, y_test_feature) = get_gene_expression_data(path, train_drp, test_drp)

        if self.task == "LDO":
            (X_train_drp, y_train_feature,
             X_test_drp, y_test_feature) = get_drug_descriptors(path, train_drp, test_drp)

        return X_train_drp, y_train_feature, X_test_drp, y_test_feature

    def save(self, result_path):
        self.models_params = {}
        for drug in self.models:
            self.models_params[drug] = {"coef": self.models.get(drug).coef_,
                                        "intercept": self.models.get(drug).intercept_,
                                        "params": self.models.get(drug).get_params()}

        # save model params dict as pickle
        with open(result_path + self.task + '_model_params.pkl', 'wb') as f:
            pickle.dump(self.models_params, f)

        # save accuracy metrics as csv
        self.metric_df.to_csv(result_path + self.task + '_metrics.csv', index=True)


if __name__ == "__main__":
    linear_regression = LinearRegression("/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/EC50 "
                                         "(µM)_matrix_cellosaurusID_intersection.csv", "EC50 (µM)", "LDO")

    linear_regression.cell_line_views
    linear_regression.drug_views

    linear_regression.get_drug_response_dataset()

    train_drp_processed, test_drp_processed = preprocessing(linear_regression.train_drp,
                                                            linear_regression.test_drp,
                                                            linear_regression.task,
                                                            linear_regression.metric, remove_out=True,
                                                            log_transform=True)

    # load cell viab data / transcriptomic data doesnt matter, as long as cell line names are the same as in the
    # drug response data ( X = drp data, y = cell viab / transcriptomic data)

    gene_counts, drug_dict = get_gene_expression_data("/nfs/home/students/m.lorenz/datasets/transcriptomics/CCLE"
                                                      "/salmon.merged.gene_counts.cellosaurusID.intersection.tsv",
                                                      train_drp_processed, test_drp_processed, feature_selection=True,
                                                      selection_method="VST")

    """
    # fit the model
    linear_regression.train(drug_dict, 0.1)

    # predict the ec50 values for the test set
    linear_regression.predict(drug_dict)

    # evaluate the model
    linear_regression.evaluate(drug_dict)
    linear_regression.metric_df

    # save model parameters and results
    dir_path = "results_transcriptomics/"
    mkdir(dir_path)
    linear_regression.save(dir_path)"""
