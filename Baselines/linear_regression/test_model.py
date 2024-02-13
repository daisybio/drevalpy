import os
import toml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso

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

logger.info("Reading in meta data from TOML file")
with open('metadata_LPO.toml', 'r') as file:
    meta_data = toml.load(file)

logger.info("Creating linear regression object")
linear_regression = LinearRegression(meta_data["metadata"]["dataroot_drp"], meta_data["metadata"]["dataroot_feature"],
                                     meta_data["metadata"]["metric"], meta_data["metadata"]["task"],
                                     meta_data["metadata"]["feature_type"], meta_data["metadata"]["feature_selection"],
                                     meta_data["metadata"]["selection_method"], meta_data["metadata"]["HP_tuning"],
                                     meta_data["metadata"]["CV_folds"], meta_data["metadata"]["n_cpus"])

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
dir_path = "results_transcriptomics/"
mkdir(dir_path)
linear_regression.save(dir_path)
"""

#################################################### DATA ANALYSIS #####################################################
logger.info("Performing data analysis")
logger.info(f"\n\nSummary statistics on {meta_data['metadata'].get('task')} - {meta_data['metadata'].get('feature_type')}:\n"
            f"{linear_regression.metric_df.describe()}\n")

### scc distribution ###
sns.histplot(linear_regression.metric_df["scc"])
median_value = linear_regression.metric_df["scc"].median()
plt.axvline(x=median_value, color='red', linestyle='dashed', linewidth=2, label='Median')
plt.xlabel("scc")
plt.ylabel("count")
plt.title(f"{meta_data['metadata'].get('task')} - scc distribution with {meta_data['metadata'].get('feature_type')}")
plt.legend()
plt.show()
plt.close()

### scc vs variance ###
if meta_data['metadata'].get('feature_type') == "fingerprints":
    scc = linear_regression.metric_df["scc"]
    drp = linear_regression.test_drp

    if linear_regression.task == "LPO":
        drp = linear_regression.test_drp
        drp = drp.pivot(index="Primary Cell Line Name", columns="Compound", values=linear_regression.metric)
        var = drp.loc[scc.index].var(axis=1)
    else:
        var = drp[scc.index].var()

elif meta_data["metadata"].get('feature_type') == "gene_expression":
    scc = linear_regression.metric_df["scc"]
    drp = linear_regression.test_drp

    if linear_regression.task == "LPO":
        drp = linear_regression.test_drp.reset_index()
        drp = drp.pivot(index="Compound", columns="Primary Cell Line Name", values=linear_regression.metric)
        var = drp.loc[scc.index].var(axis=1)
    else:
        var = drp.loc[scc.index].var(axis=1)

plt.scatter(var, scc)
plt.xlabel("variance")
plt.ylabel("scc")
plt.title(f"{meta_data['metadata'].get('task')} - scc vs variance with {meta_data['metadata'].get('feature_type')}")
plt.show()
plt.close()

### analysing how many coef. set to 0 ###
beta0_arr = []
targets = []

for target in linear_regression.models:
    if isinstance(linear_regression.models.get(target), Lasso):
        beta0_arr.append(linear_regression.models.get(target).coef_ == 0)
        targets.append(target)
    else:
        target_GCV = linear_regression.models.get(target)
        beta0_arr.append(target_GCV.best_estimator_.coef_ == 0)
        targets.append(target)


beta0_df = pd.DataFrame(index=targets, data=beta0_arr)
beta0_df.sum()
sns.barplot(x=beta0_df.sum().index, y=beta0_df.sum().values)
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
plt.xlabel('coefficient number')
plt.ylabel('count')
plt.title(f'Bar Plot of Coefficients set to 0 ('
          f'{meta_data["metadata"]["task"]} - {meta_data["metadata"]["feature_type"]})')
plt.show()
plt.close()
