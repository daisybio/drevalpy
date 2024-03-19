import logging
import os
import sys
import toml
from os.path import dirname, join, abspath
from pathlib import Path
from sklearn.svm import SVC

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from model import SupportVectorClassifier
from utils.utils import mkdir
from utils import testing, analysis

# setting up logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

# setting up directory for saving results
# save model parameters and results
dir_path = "/nfs/home/students/m.lorenz/output/svc/GDSC/SVC_LDO_2feat_SMOTE_poly_kernel/"
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
logger.info("Running support vector classifier")

# read in meta data from TOML file
logger.info("Reading in meta data from TOML file")
with open('metadata_LDO.toml', 'r') as file:
    meta_data = toml.load(file)

# create linear regression object
logger.info("Creating support vector classifier object")

SVC_classifier = testing.parse_data(meta_data, SupportVectorClassifier, "SVC")

# perform training, testing and evaluation
best_models, best_nfeatures, best_scc, best_models_params = (
    testing.train_test_eval(SVC_classifier, SVC, "classification", meta_data, dir_path))

# perform data analysis
logger.info("Performing data analysis")
analysis.base_analysis(best_models, best_nfeatures, SVC_classifier, SVC, "classification", meta_data, dir_path)
# analysis.coef0_distribution(best_models, best_nfeatures, SVC, meta_data, dir_path) # only for linear kernel
analysis.scores_clustering(best_models, dir_path)
analysis.roc_plot(best_models, 0)
# analysis.decision_boundary(best_models, "CVCL_0002", SVC) # only for 2D data
