import logging
import os
import sys
import toml
from os.path import dirname, join, abspath
from pathlib import Path
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from model import LogisticClassifier
from utils.utils import mkdir
from utils import testing, analysis

# setting up logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

# setting up directory for saving results
# save model parameters and results
dir_path = "lincf_LCO_2feat/"
mkdir(dir_path)

# setting up file logging as well
file_logger = logging.FileHandler(Path(os.getcwd() / Path(dir_path + 'Baseline-models.log')), mode='w')
file_logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_logger.setFormatter(formatter)
logging.getLogger().addHandler(file_logger)

# setting up logger for current module
logger = logging.getLogger(__name__)

# start logging
logger.info("Running linear regression model")

# read in meta data from TOML file
logger.info("Reading in meta data from TOML file")
with open('metadata_LCO.toml', 'r') as file:
    meta_data = toml.load(file)

# create linear regression object
logger.info("Creating linear regression object")

logistic_classifier = testing.parse_data(meta_data, LogisticClassifier)

# perform training, testing and evaluation
best_models, best_nfeatures, best_scc, best_models_params = testing.train_test_eval(logistic_classifier,
                                                                                    LogisticRegression, meta_data,
                                                                                    dir_path)

# perform data analysis
logger.info("Performing data analysis")
analysis.base_analysis(best_models, best_nfeatures, logistic_classifier, LogisticRegression, meta_data, dir_path)
#analysis.cluster(best_models, dir_path)
fstat_df = analysis.f_statistic(best_models, best_nfeatures)
analysis.f_distribution(best_nfeatures, fstat_df.iloc[0, 1], fstat_df.iloc[0, 3], fstat_df.iloc[0, 4])
