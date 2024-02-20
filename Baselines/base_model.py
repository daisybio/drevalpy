import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy import stats

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self, dataroot_drp, dataroot_feature, metric, task, remove_out=True,
                 log_transform=True, feature_type="gene_expression", feature_selection=False,
                 norm_feat=False, norm_method=None, nCV_folds=None, n_cpus=1, hyperparameters=None):
        self.path_drp = dataroot_drp  # path to the drug response data
        self.path_feature = dataroot_feature  # path to the feature data
        self.metric = metric  # Amax, IC50, EC50, ...
        self.task = task  # LCO, LDO, LPO
        self.remove_out = remove_out  # whether to remove outliers from the drug response data
        self.log_transform = log_transform  # whether to log transform the drug response data
        self.feature_type = feature_type  # view under cell_line_views or drug_views
        self.feature_selection = feature_selection  # whether to perform feature selection
        self.norm_feat = norm_feat  # whether to normalize the features
        self.norm_method = norm_method  # method for feature normalization
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
        self.drp_df = pd.read_csv(self.path_drp, header=0, index_col=0)
        self.drp_df.reset_index(inplace=True)

        logger.info(f"Reading in {self.feature_type} data")
        if self.feature_type == "gene_expression":
            self.feature_df = pd.read_csv(self.path_feature, index_col=0).T
        elif self.feature_type == "fingerprints":
            self.feature_df = pd.read_csv(self.path_feature, index_col=0)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save(self, result_path, best_model_dict):
        pass

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
                                                      remove_out=self.remove_out, log_transform=self.log_transform)
        logger.info("finished preprocessing drug response data")

    def get_feature_dataset(self, ntop):

        if self.feature_type == "gene_expression":
            logger.info(f"preparing gene expression data - feature selection: {self.feature_selection}")
            drug_dict = get_gene_expression_data(self.feature_df, self.train_drp, self.test_drp, self.task,
                                                 feature_selection=self.feature_selection, ntop=ntop,
                                                 norm_feat=self.norm_feat, norm_method=self.norm_method,
                                                 n_cpus=self.n_cpus)
            self.data_dict = drug_dict

        elif self.feature_type == "fingerprints":
            logger.info(f"preparing morgan fingerprints - feature selection: {self.feature_selection}")
            cl_dict = get_morgan_fingerprints(self.feature_df, self.train_drp, self.test_drp, self.task,
                                              feature_selection=self.feature_selection, ntop=ntop)
            self.data_dict = cl_dict

        logger.info("finished preparing feature data, output stored in data_dict")