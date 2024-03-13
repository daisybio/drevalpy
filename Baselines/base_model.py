import logging
import numpy as np
import pandas as pd
import sys
from abc import ABC, abstractmethod, abstractproperty
from os.path import dirname, join, abspath
from scipy import stats
from sklearn.metrics import mean_squared_error

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils.load_data import get_train_test_set, get_gene_expression_data, get_morgan_fingerprints
from utils.utils import preprocessing

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self, dataroot_drp, dataroot_feature, metric, task, remove_out=True,
                 log_transform=True, feature_type="gene_expression", feature_selection=False,
                 norm_feat=False, norm_method=None, nCV_folds=None, oversampling_method=None, n_cpus=1,
                 hyperparameters=None):
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
        self.nCV_folds = nCV_folds  # number of cross validation folds
        self.oversampling_method = oversampling_method  # whether to perform oversampling
        self.n_cpus = n_cpus  # nr of cpus to use for parallelization, relevant only for feature sel. using VST method
        self.hyperparameters = hyperparameters  # hyperparameters for the model

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

    @abstractproperty
    def cell_line_views(self):
        """
        Returns the sources the model needs as input for describing the cell line.
        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression", "mutation"]
        """
        pass

    @abstractproperty
    def drug_views(self):
        """
        Returns the sources the model needs as input for describing the drug.
        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """
        pass

    @abstractmethod
    def train(self):
        """
        Trains the model for each target in the data dictionary. in the case of LCO, single drug models are trained,
        meaning each target in the data_dict corresponds to a single drug for which a model is being generated. In
        the case of LDO, single cell line models are trained, meaning each target in the data_dict corresponds to a
        single cell line. Grid search is performed to find the best hyperparameters for each model with the number of
        cross validation folds specified by the user. Parameters to be optimized are specified by the user.
        """
        pass

    @abstractmethod
    def save(self, result_path, best_model_dict):
        """
        Saves the model parameters and the accuracy metrics to the result path. The model parameters are saved as a pickle
        file and the accuracy metrics are saved as a csv file. The model parameters are saved as a dictionary with the
        target as the key and the model parameters as the value. The accuracy metrics are saved as a pandas dataframe.
        :param result_path: path to the directory where the results are saved
        :param best_model_dict: dictionary containing the best models, the best number of features, the best accuracy
        metrics and the best model parameters
        """
        pass

    def predict(self):
        """
        Predicts the drug response for the test set. The predicted values are stored in self.prediction. The probability
        of the positive class is stored in self.probability. The predicted values and the probability are stored in a
        pandas dataframe with the true values and the target for each sample.
        """
        logger.info("predicting drug response for test set")
        self.prediction = {}
        for target in self.data_dict:
            model = self.models.get(target)
            X_test = self.data_dict.get(target).get("X_test")
            yhat = model.predict(X_test)
            self.prediction[target] = yhat  # predicted responses for each target
        logger.info("finished predicting")

    def evaluate(self):
        """
        Evaluates the model by computing the pearson correlation coefficient, spearman correlation coefficient, mean
        squared error and root mean squared error for each target. The results are stored in a pandas dataframe. The
        predicted values and the true values are stored in a pandas dataframe with the target and the sample id.
        """
        logger.info("evaluating models")

        # initialize pandas dataframe with y_true, y_pred, target
        pred_df = pd.DataFrame({"y_true": np.concatenate([self.data_dict.get(target).get("y_test").reshape(-1)
                                                          for target in self.data_dict]),
                                "y_pred": np.concatenate([self.prediction.get(target) for target in self.data_dict]),
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

    def get_drug_response_dataset(self):
        """
        Prepares the drug response data by splitting it into a train and a test set. The train set contains 80% of the
        data and the test set contains 20% of the data. The train and test sets are stored in self.train_drp and self.test_drp
        respectively. the metric in this case indicates what unit the drug response is measured in (e.g. IC50, EC50, Amax).
        """
        logger.info("preparing drug response data")
        self.train_drp, self.test_drp = get_train_test_set(self.drp_df, self.task, 0.8, self.metric)
        logger.info("finished preparing drug response data")

    def data_processing(self):
        """
        Pre processes the drug response data by removing outliers, normalizing the data and log transforming the data.
        """
        logger.info("preprocessing drug response data")
        self.train_drp, self.test_drp = preprocessing(self.train_drp, self.test_drp, self.task, self.metric,
                                                      remove_out=self.remove_out, log_transform=self.log_transform)
        logger.info("finished preprocessing drug response data")

    def get_feature_dataset(self, ntop):
        """
        Prepares the feature data by selecting the most important features. The number of features to be selected is
        specified by the user. The feature data is stored in a dictionary with the target as the key and the feature
        data as the value. The feature data contains the train and test set for each target. The train set contains the
        feature data for the train set and the test set contains the feature data for the test set. The feature data is
        stored in a pandas dataframe with the samples as the rows and the features as the columns.
        :param ntop: number of features to be selected
        """

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
