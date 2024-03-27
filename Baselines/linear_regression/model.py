# -*- coding: utf-8 -*-
import logging
import pickle
import sys
from os.path import dirname, join, abspath
from sklearn.linear_model import Lasso

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from base_model import BaseModel
from utils.utils import cross_validation_fit

logger = logging.getLogger(__name__)


class LinearRegression(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

            # hyperparameter tuning using grid search cross validation
            model = cross_validation_fit(X_train, y_train, reg, self.nCV_folds, self.hyperparameters)

            self.models[target] = model
        logger.info("finished training models")
        # logger.info(f"average length of training set: {sum_all_lengths / len(self.data_dict)}")

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
