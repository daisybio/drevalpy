import logging
import pandas as pd
import sys
from os.path import dirname, join, abspath

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

logger = logging.getLogger(__name__)


def parse_data(meta_data, predictor_class):
    predictor_object = predictor_class(meta_data["metadata"]["dataroot_drp"], meta_data["metadata"]["dataroot_feature"],
                                       meta_data["metadata"]["metric"], meta_data["metadata"]["task"],
                                       meta_data["metadata"]["remove_outliers"], meta_data["metadata"]["log_transform"],
                                       meta_data["metadata"]["feature_type"],
                                       meta_data["metadata"]["feature_selection"],
                                       meta_data["metadata"]["norm_feat"], meta_data["metadata"]["norm_method"],
                                       meta_data["metadata"]["CV_folds"], meta_data["metadata"]["oversampling_method"],
                                       meta_data["metadata"]["n_cpus"], meta_data["metadata"]["HP_tuning"])
    return predictor_object


def train_test_eval(predictor, predictor_class, predictor_type, meta_data, dir_path):
    # prepare drug response data (splitting it)
    predictor.get_drug_response_dataset()

    # pre process the drp (y) data
    predictor.data_processing()

    # load cell viab/transcriptomic data doesn't matter, as long as cl names are the same as in the drug response data
    key_metric_median = 0
    best_key_metric = 0
    best_nfeatures = None
    for ntop in meta_data["metadata"]["HP_tuning_features"].get("nfeatures"):
        logger.info(f"Starting dataextraction / training / prediction loop for {ntop} features")
        predictor.get_feature_dataset(ntop)

        # fit the model
        predictor.train()

        # predict the ec50 values for the test set
        predictor.predict()

        # evaluate the model
        predictor.evaluate()

        if predictor_type == "classification":
            metric = "ROC-auc"
        elif predictor_type == "regression":
            metric = "scc"
        key_metric_median = predictor.metric_df[metric].median()

        # save the model if its key performance metric is better than the previous one in best_model_attr
        if key_metric_median > best_key_metric:
            logger.info(f"New best model found with {ntop} features")
            best_model_attr = dict(predictor.__dict__)  # vars(predictor)
            best_key_metric = key_metric_median
            best_nfeatures = ntop

    # get the best Hyperparameters for each model
    HPs = []
    for HP in meta_data["metadata"]["HP_tuning"]:
        HPs.append(HP)

    HPs_models = {}
    for HP in HPs:
        HPs_models[HP] = []
        for target in best_model_attr["models"]:
            target_model = best_model_attr["models"].get(target)
            if isinstance(target_model, predictor_class):
                HPs_models[HP].append(target_model.get_params()[HP])
            else:
                HPs_models[HP].append(target_model.best_params_.get(HP))

    # there are more cl with models in best_model_attr["models"] than in best_model_attr["metric_df"] since there we calc.
    # the scc for cls with more than one drug. Filter out the alpha and max_iter for cl models with more than one drug
    best_models_params = pd.DataFrame(HPs_models, index=best_model_attr["models"].keys())
    best_models_params = best_models_params.loc[best_model_attr["metric_df"].index]

    best_model_attr["metric_df"]["nfeatures"] = best_nfeatures
    best_model_attr["metric_df"] = best_model_attr["metric_df"].join(best_models_params, how="inner")

    predictor.save(dir_path, best_model_attr)

    return best_model_attr, best_nfeatures, best_key_metric, best_models_params
