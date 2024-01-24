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
        self.models = None  # model fit
        self.models_params = None  # model parameters

        logger.info("Reading in drug response data")
        self.drp_df = pd.read_csv(self.path_drp, header=0, index_col=0)  # load drp data
        self.drp_df.reset_index(inplace=True)

        # df containing features (not split ino training and test sets)
        logger.info(f"Reading in {self.feature_type} data")
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
        for target in self.data_dict:

            logger.info(f"training single single model for {target}")
            X_train = self.data_dict.get(target).get("X_train")  # get the training data for the target from dict
            y_train = self.data_dict.get(target).get("y_train")

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
        pcc_ls = []
        scc_ls = []
        mse_ls = []
        rmse_ls = []
        cls = []
        for target in self.data_dict:
            #  skip targets with only one sample as scc can only be calculated with at least two samples
            #  e.g. for LCO, skip drugs with only one cell line
            #  also skip target where all predictions are the same, leading to a constant -> pcc/scc not calculated
            if (len(self.data_dict.get(target).get("y_test").reshape(-1)) <= 1 or
                    len(np.unique(self.prediction.get(target))) == 1):
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

    def get_feature_dataset(self):

        if self.feature_type == "gene_expression":
            logger.info("preparing gene expression data")
            drug_dict = get_gene_expression_data(self.feature_df, self.train_drp, self.test_drp, self.task,
                                                 feature_selection=self.feature_selection,
                                                 selection_method=self.selection_method, n_cpus=self.n_cpus)
            self.data_dict = drug_dict

        elif self.feature_type == "fingerprints":
            logger.info("preparing morgan fingerprints")
            cl_dict = get_morgan_fingerprints(self.feature_df, self.train_drp, self.test_drp, self.task,
                                              feature_selection=self.feature_selection)
            self.data_dict = cl_dict

        logger.info("finished preparing feature data, output stored in data_dict")

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
        logger.info("saved model parameters as pickles and accuracy metrics as csv")


if __name__ == "__main__":
    # setting up logging
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    # setting up file logging as well
    file_logger = logging.FileHandler(Path(os.getcwd() / Path('Baseline-models.log')), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_logger.setFormatter(formatter)
    logging.getLogger(__name__).addHandler(file_logger)

    # start logging
    logger.info("Running linear regression model")

    # meta_data = {"dataroot_drp": "/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/EC50 ("
    #                              "µM)_matrix_cellosaurusID_intersection.csv",
    #              "dataroot_feature": "/nfs/home/students/m.lorenz/datasets/transcriptomics/CCLE/salmon.merged"
    #                                  ".gene_counts.cellosaurusID.intersection.tsv",
    #              "metric": "EC50 (µM)",
    #              "task": "LCO",
    #              "feature_type": "gene_expression",
    #              "feature_selection": True,
    #              "selection_method": "VST",
    #              "HP_tuning": {'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001], 'max_iter': [1000, 10000]},
    #              "CV_folds": 5,
    #              "n_cpus": 10}

    # meta_data = {"dataroot_drp": "/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/EC50 ("
    #                              "µM)_matrix_cellosaurusID_intersection.csv",
    #              "dataroot_feature": "/nfs/home/students/m.lorenz/datasets/compounds/c_morganfp.csv",
    #              "metric": "EC50 (µM)",
    #              "task": "LDO",
    #              "feature_type": "fingerprints",
    #              "feature_selection": True,
    #              "selection_method": "None",
    #              "HP_tuning": {'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001], 'max_iter': [1000, 10000]},
    #              "CV_folds": 5,
    #              "n_cpus": 1}

    logger.info("Reading in meta data")

    meta_data = {"dataroot_drp": "/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/EC50 ("
                                 "µM)_matrix_cellosaurusID_intersection.csv",
                 "dataroot_feature": "/nfs/home/students/m.lorenz/datasets/transcriptomics/CCLE/salmon.merged"
                                     ".gene_counts.cellosaurusID.intersection.tsv",
                 "metric": "EC50 (µM)",
                 "task": "LPO",
                 "feature_type": "gene_expression",
                 "feature_selection": True,
                 "selection_method": "VST",
                 "HP_tuning": {'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001], 'max_iter': [1000, 10000]},
                 "CV_folds": 5,
                 "n_cpus": 40}

    logger.info("Creating linear regression object")
    linear_regression = LinearRegression(meta_data.get("dataroot_drp"), meta_data.get("dataroot_feature"),
                                         meta_data.get("metric"), meta_data.get("task"), meta_data.get("feature_type"),
                                         meta_data.get("feature_selection"), meta_data.get("selection_method"),
                                         meta_data.get("HP_tuning"), meta_data.get("CV_folds"), meta_data.get("n_cpus"))

    linear_regression.cell_line_views
    linear_regression.drug_views

    # prepare drug response data (splitting it)
    linear_regression.get_drug_response_dataset()

    # pre process the drp (y) data
    linear_regression.data_processing()

    # load cell viab/transcriptomic data doesn't matter, as long as cl names are the same as in the drug response data
    linear_regression.get_feature_dataset()

    # fit the model
    linear_regression.train()

    # predict the ec50 values for the test set
    linear_regression.predict()

    # evaluate the model
    linear_regression.evaluate()

    """
    # save model parameters and results
    dir_path = "results_transcriptomics/"
    mkdir(dir_path)
    linear_regression.save(dir_path)"""
