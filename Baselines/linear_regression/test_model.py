import os
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from model import LinearRegression

# setting up logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

# setting up file logging as well
file_logger = logging.FileHandler(Path(os.getcwd() / Path('Baseline-models.log')), mode='w')
file_logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_logger.setFormatter(formatter)
logging.getLogger().addHandler(file_logger)

# setting up logger for current module
logger = logging.getLogger(__name__)

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
             "selection_method": "NormTransform",
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

# check model performance
logger.info(f"\n\nSummary statistics on {meta_data.get('task')} - {meta_data.get('feature_type')}:\n"
            f"{linear_regression.metric_df.describe()}\n")

sns.histplot(linear_regression.metric_df["scc"])
median_value = linear_regression.metric_df["scc"].median()
plt.axvline(x=median_value, color='red', linestyle='dashed', linewidth=2, label='Median')
plt.xlabel("scc")
plt.ylabel("count")
plt.title(f"{meta_data.get('task')} - scc distribution with {meta_data.get('feature_type')}")
plt.legend()
plt.show()
plt.close()

if meta_data.get('feature_type') == "fingerprints":
    scc = linear_regression.metric_df["scc"]
    drp = linear_regression.test_drp

    if linear_regression.task == "LPO":
        drp = linear_regression.test_drp
        drp = drp.pivot(index="Primary Cell Line Name", columns="Compound", values=linear_regression.metric)
        var = drp.loc[scc.index].var(axis=1)
    else:
        var = drp.loc[scc.index].var()

elif meta_data.get('feature_type') == "gene_expression":
    scc = linear_regression.metric_df["scc"]
    drp = linear_regression.test_drp

    if linear_regression.task == "LPO":
        drp = linear_regression.test_drp.reset_index()
        drp = drp.pivot(index="Compound", columns="Primary Cell Line Name", values=linear_regression.metric)
        var = drp.loc[scc.index].var(axis=1)
    else:
        var = drp.loc[scc.index].var()

plt.scatter(var, scc)
plt.xlabel("variance")
plt.ylabel("scc")
plt.title(f"{meta_data.get('task')} - scc vs variance with {meta_data.get('feature_type')}")
plt.show()
plt.close()
