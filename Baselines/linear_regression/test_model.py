import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import seaborn as sns
import sys
import toml
from os.path import dirname, join, abspath
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Lasso

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from model import LinearRegression
from utils.utils import mkdir

# setting up logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

# setting up directory for saving results
# save model parameters and results
dir_path = "~/output/linreg/GDSC/linreg_LPO_2feat_fp/"
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
with open('metadata_LPO.toml', 'r') as file:
    meta_data = toml.load(file)

# create linear regression object
logger.info("Creating linear regression object")
linear_regression = LinearRegression(meta_data["metadata"]["dataroot_drp"], meta_data["metadata"]["dataroot_feature"],
                                     meta_data["metadata"]["metric"], meta_data["metadata"]["task"],
                                     meta_data["metadata"]["remove_outliers"], meta_data["metadata"]["log_transform"],
                                     meta_data["metadata"]["feature_type"], meta_data["metadata"]["feature_selection"],
                                     meta_data["metadata"]["norm_feat"], meta_data["metadata"]["norm_method"],
                                     meta_data["metadata"]["CV_folds"], meta_data["metadata"]["n_cpus"],
                                     meta_data["metadata"]["HP_tuning"])

linear_regression.cell_line_views
linear_regression.drug_views

# prepare drug response data (splitting it)
linear_regression.get_drug_response_dataset()

# pre process the drp (y) data
linear_regression.data_processing()

# load cell viab/transcriptomic data doesn't matter, as long as cl names are the same as in the drug response data
scc_median = 0
best_scc = 0
best_nfeatures = None
for ntop in meta_data["metadata"]["HP_tuning_features"].get("nfeatures"):
    logger.info(f"Starting dataextraction / training / prediction loop for {ntop} features")
    linear_regression.get_feature_dataset(ntop)

    # fit the model
    linear_regression.train()

    # predict the ec50 values for the test set
    linear_regression.predict()

    # evaluate the model
    linear_regression.evaluate()
    scc_median = linear_regression.metric_df["scc"].median()

    # save the model if its scc is better than the previous one in best_model_attr
    if scc_median > best_scc:
        logger.info(f"New best model found with {ntop} features")
        best_model_attr = dict(linear_regression.__dict__)  # vars(linear_regression)
        best_scc = scc_median
        best_nfeatures = ntop

# get the best alpha and maximum number of iterations
alpha = []
max_iter = []
for target in best_model_attr["models"]:
    target_model = best_model_attr["models"].get(target)
    if isinstance(target_model, Lasso):
        alpha.append(target_model.get_params()["alpha"])
        max_iter.append(target_model.get_params()["max_iter"])
    else:
        alpha.append(target_model.best_params_.get("alpha"))
        max_iter.append(target_model.best_params_.get("max_iter"))

# there are more cl with models in best_model_attr["models"] than in best_model_attr["metric_df"] since there we calc.
# the scc for cls with more than one drug. Filter out the alpha and max_iter for cl models with more than one drug
best_models_params = pd.DataFrame({"alpha": alpha, "max_iter": max_iter}, index=best_model_attr["models"].keys())
best_models_params = best_models_params.loc[best_model_attr["metric_df"].index]

best_model_attr["metric_df"]["nfeatures"] = best_nfeatures
best_model_attr["metric_df"]["alpha"] = best_models_params["alpha"]
best_model_attr["metric_df"]["max_iter"] = best_models_params["max_iter"]

linear_regression.save(dir_path, best_model_attr)
#################################################### DATA ANALYSIS #####################################################
logger.info("Performing data analysis")
logger.info(
    f"\n\nSummary statistics on {meta_data['metadata'].get('task')} - {meta_data['metadata'].get('feature_type')}:\n"
    f"{best_model_attr['metric_df'].describe()}\n")

sns.set(style="ticks")

### correlation coefficient distribution ###
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
sns.histplot(best_model_attr["metric_df"]["pcc"], ax=axs[0], binrange=(-1, 1))
sns.histplot(best_model_attr["metric_df"]["scc"], ax=axs[1], binrange=(-1, 1))
median_value_pcc = best_model_attr["metric_df"]["pcc"].median()
median_value_scc = best_model_attr["metric_df"]["scc"].median()
axs[0].axvline(x=median_value_pcc, color='red', linestyle='dashed', linewidth=2, label='median')
axs[1].axvline(x=median_value_scc, color='red', linestyle='dashed', linewidth=2, label='median')
axs[0].set_xlabel("Pearsons's correlation coefficient (PCC)")
axs[1].set_xlabel("Spearman's correlation coefficient (SCC)")
plt.ylabel("count")
plt.suptitle(f"distribution of correlation coefficients "
             f"({meta_data['metadata'].get('task')} - {meta_data['metadata'].get('feature_type')})",
             fontsize=12, fontweight='bold')
# axs[0].legend()
# axs[1].legend()
sns.despine(right=True)
fig = plt.gcf()
plt.show()
fig.savefig(Path(dir_path + "correlation_coefficients_distribution.png"))
plt.close()

### scc vs variance ###
if meta_data['metadata'].get('feature_type') == "fingerprints":
    scc = best_model_attr["metric_df"]["scc"]
    pcc = best_model_attr["metric_df"]["pcc"]
    drp = best_model_attr["test_drp"]

    if linear_regression.task == "LPO":
        drp = best_model_attr["test_drp"]
        drp = drp.pivot(index="Primary Cell Line Name", columns="Compound", values=best_model_attr["metric"])
        var = drp.loc[scc.index].var(axis=1)
    else:
        var = drp[scc.index].var()

elif meta_data["metadata"].get('feature_type') == "gene_expression":
    scc = best_model_attr["metric_df"]["scc"]
    pcc = best_model_attr["metric_df"]["pcc"]
    drp = best_model_attr["test_drp"]

    if linear_regression.task == "LPO":
        drp = best_model_attr["test_drp"].reset_index()
        drp = drp.pivot(index="Compound", columns="Primary Cell Line Name", values=best_model_attr["metric"])
        var = drp.loc[scc.index].var(axis=1)
    else:
        var = drp.loc[scc.index].var(axis=1)

fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)']], figsize=(15, 10))
axs['a)'].scatter(var, pcc)
axs['b)'].scatter(var, scc)
plt.xlabel("variance")
axs['a)'].set_ylabel("Pearsons's correlation coefficient")
axs['b)'].set_ylabel("Spearman's correlation coefficient")
axs['a)'].set_title(f"correlation coefficient vs variance "
                    f"{meta_data['metadata'].get('task')} - {meta_data['metadata'].get('feature_type')}",
                    fontsize=12, fontweight='bold')
# sns.despine(right = True)
# plt.tight_layout()
# plt.show()
# plt.close()

### analysing how many coef. set to 0 ###
beta0_arr = []
targets = []

for target in best_model_attr["models"]:
    if isinstance(best_model_attr["models"].get(target), Lasso):
        beta0_arr.append(best_model_attr["models"].get(target).coef_ == 0)
        targets.append(target)
    else:
        target_GCV = best_model_attr["models"].get(target)
        beta0_arr.append(target_GCV.best_estimator_.coef_ == 0)
        targets.append(target)

beta0_df = pd.DataFrame(index=targets, data=beta0_arr)
beta0_df.sum()
sns.barplot(x=beta0_df.sum().index, y=beta0_df.sum().values, ax=axs['c)'])
axs['c)'].set_xlabel('coefficient number')
axs['c)'].set_ylabel('count')
axs['c)'].set_title(f'frequency of coefficients set to 0 ('
                    f'{meta_data["metadata"]["task"]} - {meta_data["metadata"]["feature_type"]})', fontsize=12,
                    fontweight='bold')
sns.despine(right=True)
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig(Path(dir_path + "coefficient_variance_frequency.png"))
plt.close()

logger.info(f"\nAverage number of coefficients set to 0 over all models: {beta0_df.T.sum().mean()}\n"
            f"Average number of coefficients not set to 0 over all models: {(best_nfeatures - beta0_df.T.sum()).mean()}\n"
            f"percentage of avg number of coefficients set to 0 over total coef: {beta0_df.T.sum().mean() / best_nfeatures * 100}\n"
            f"percentage of avg number of coefficients not set to 0 over total coef: {(best_nfeatures - beta0_df.T.sum()).mean() / best_nfeatures * 100}\n")

# generate scatter plot of predictions
# plot y_true vs y_pred, in title: overall correlation

# compute the overall pcc and scc
pcc = stats.pearsonr(best_model_attr["pred_df"]["y_true"], best_model_attr["pred_df"]["y_pred"])[0]
scc = stats.spearmanr(best_model_attr["pred_df"]["y_true"], best_model_attr["pred_df"]["y_pred"])[0]

fig = px.scatter(
    best_model_attr["pred_df"], x="y_true", y="y_pred", color="target", trendline="ols", hover_name="sample_id",
    hover_data=["scc", "pcc"], title="Overall PCC: {:.2f}, SCC: {:.2f}".format(pcc, scc)
)

fig.write_html(Path(dir_path + "scatter_plot_predictions.html"))

# average number of datapoints per model:
ls = []
for target in linear_regression.data_dict:
    ls.append(np.shape(linear_regression.data_dict.get(target).get("X_train"))[0])

logger.info(
    f"\n\nAverage number of datapoints per model for training: {np.mean(ls)}\n"
    f"Average number of datapoints per model for testing:"
    f" {best_model_attr['pred_df'].groupby('target').size().mean()}\n")

sns.histplot(x=best_model_attr['pred_df'].groupby('target').size())
plt.title(f"Average number of datapoints per model for testing:"
          f" {round(best_model_attr['pred_df'].groupby('target').size().mean())}",
          fontsize=12, fontweight='bold')
plt.xlabel('number of samples in a model')
plt.ylabel('number of models')
sns.despine(right=True)
fig = plt.gcf()
plt.show()
fig.savefig(Path(dir_path + "average_number_of_datapoints_per_model.png"))
plt.close()

# compute F statistic to see if fit is significant
groups = best_model_attr["pred_df"].groupby(by="target")
F = []
p_values = []
for name, group in groups:
    ssreg = ((group["y_pred"] - group["y_true"].mean()) ** 2).sum()
    ssres = ((group["y_true"] - group["y_pred"]) ** 2).sum()
    k = best_model_attr["models"][name].best_estimator_.coef_.shape[0]  # intercept B0 not included in k
    n = len(group)

    F_group = (ssreg / k) / (ssres / (n - k - 1))
    p_value = 1 - stats.f.cdf(F_group, k, n - k - 1)  # this returns nan if number of samples n < number of features k
    F.append(F_group)
    p_values.append(p_value)

logger.info(f"Number of models with p_val < 0.05: {(np.array(p_values) < 0.05).sum()}"
            f" ({round((np.array(p_values) < 0.05).sum() / len(p_values), 3)}%)")

# plot F-distribution
X = np.linspace(0, 5, 200)
dfn = k
dfd = n - k - 1
Y = stats.f.pdf(X, dfn, dfd)
plt.plot(X, Y, label=f"F-distribution dfn: {dfn}, dfd: {dfd}")
plt.fill_between(X, Y, where=(X > F_group), alpha=0.3)
plt.title(f"Computed F-statistic: {F_group:.2f}, p-value: {p_value:.2f}")
plt.xlabel("F-statistic")
plt.ylabel("probability")
plt.legend()
plt.show()
plt.close()
