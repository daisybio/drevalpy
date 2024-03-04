import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sys
from os.path import dirname, join, abspath
from pathlib import Path
from scipy import stats

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

logger = logging.getLogger(__name__)


def data_analysis(best_models, best_nfeatures, predictor, predictor_type, meta_data, dir_path):
    logger.info(
        f"\n\nSummary statistics on {meta_data['metadata'].get('task')} - {meta_data['metadata'].get('feature_type')}:\n"
        f"{best_models['metric_df'].describe()}\n")

    sns.set(style="ticks")

    ### correlation coefficient distribution ###
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    sns.histplot(best_models["metric_df"]["pcc"], ax=axs[0], binrange=(-1, 1))
    sns.histplot(best_models["metric_df"]["scc"], ax=axs[1], binrange=(-1, 1))
    median_value_pcc = best_models["metric_df"]["pcc"].median()
    median_value_scc = best_models["metric_df"]["scc"].median()
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
        scc = best_models["metric_df"]["scc"]
        pcc = best_models["metric_df"]["pcc"]
        drp = best_models["test_drp"]

        if predictor.task == "LPO":
            drp = best_models["test_drp"]
            drp = drp.pivot(index="Primary Cell Line Name", columns="Compound", values=best_models["metric"])
            var = drp.loc[scc.index].var(axis=1)
        else:
            var = drp[scc.index].var()

    elif meta_data["metadata"].get('feature_type') == "gene_expression":
        scc = best_models["metric_df"]["scc"]
        pcc = best_models["metric_df"]["pcc"]
        drp = best_models["test_drp"]

        if predictor.task == "LPO":
            drp = best_models["test_drp"].reset_index()
            drp = drp.pivot(index="Compound", columns="Primary Cell Line Name", values=best_models["metric"])
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

    for target in best_models["models"]:
        if isinstance(best_models["models"].get(target), predictor_type):
            beta0_arr.append(best_models["models"].get(target).coef_ == 0)
            targets.append(target)
        else:
            target_GCV = best_models["models"].get(target)
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
    pcc = stats.pearsonr(best_models["pred_df"]["y_true"], best_models["pred_df"]["y_pred"])[0]
    scc = stats.spearmanr(best_models["pred_df"]["y_true"], best_models["pred_df"]["y_pred"])[0]

    fig = px.scatter(
        best_models["pred_df"], x="y_true", y="y_pred", color="target", trendline="ols", hover_name="sample_id",
        hover_data=["scc", "pcc"], title="Overall PCC: {:.2f}, SCC: {:.2f}".format(pcc, scc)
    )

    fig.write_html(Path(dir_path + "scatter_plot_predictions.html"))

    # average number of datapoints per model:
    ls = []
    for target in predictor.data_dict:
        ls.append(np.shape(predictor.data_dict.get(target).get("X_train"))[0])

    logger.info(
        f"\n\nAverage number of datapoints per model for training: {np.mean(ls)}\n"
        f"Average number of datapoints per model for testing:"
        f" {best_models['pred_df'].groupby('target').size().mean()}\n")

    sns.histplot(x=best_models['pred_df'].groupby('target').size())
    plt.title(f"Average number of datapoints per model for testing:"
              f" {round(best_models['pred_df'].groupby('target').size().mean())}",
              fontsize=12, fontweight='bold')
    plt.xlabel('number of samples in a model')
    plt.ylabel('number of models')
    sns.despine(right=True)
    fig = plt.gcf()
    plt.show()
    fig.savefig(Path(dir_path + "average_number_of_datapoints_per_model.png"))
    plt.close()

    # compute F statistic to see if fit is significant
    groups = best_models["pred_df"].groupby(by="target")
    F = []
    p_values = []
    for name, group in groups:
        ssreg = ((group["y_pred"] - group["y_true"].mean()) ** 2).sum()
        ssres = ((group["y_true"] - group["y_pred"]) ** 2).sum()
        k = best_models["models"][name].best_estimator_.coef_.shape[0]  # intercept B0 not included in k
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
