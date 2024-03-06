import dash_bio
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)


def base_analysis(best_models, best_nfeatures, predictor, predictor_type, meta_data, dir_path):
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
            beta0_arr.append(np.reshape(best_models["models"].get(target).coef_ == 0),-1)
            targets.append(target)
        else:
            target_GCV = best_models["models"].get(target)
            beta0_arr.append(np.reshape(target_GCV.best_estimator_.coef_ == 0,-1))
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


def scatter_predictions(best_models, dir_path):
    # generate scatter plot of predictions
    # plot y_true vs y_pred, in title: overall correlation

    # compute the overall pcc and scc
    pcc = stats.pearsonr(best_models["pred_df"]["y_true"], best_models["pred_df"]["y_pred"])[0]
    scc = stats.spearmanr(best_models["pred_df"]["y_true"], best_models["pred_df"]["y_pred"])[0]

    # Create figure
    n_ticks = 21
    fig = px.scatter(best_models["pred_df"], x="y_true", y="y_pred", color="target", trendline="ols",
                     hover_name="sample_id", hover_data=["scc", "pcc"], title=f"Overall PCC: {pcc:.2f}, SCC: {scc:.2f}")

    # Create and add slider
    steps = []
    # take the range from scc and divide it into 10 equal parts
    scc_parts = np.linspace(-1, 1, n_ticks)
    for i in range(n_ticks):
        # from the fig data, get the hover data and check if it is greater than the scc_parts[i]
        # only iterate over even numbers
        sccs = [0 for _ in range(0, len(fig.data))]
        for j in range(0, len(fig.data)):
            if j % 2 == 0:
                sccs[j] = fig.data[j].customdata[0, 0]
            else:
                sccs[j] = fig.data[j - 1].customdata[0, 0]

        if i == n_ticks - 1:
            visible_traces = sccs >= scc_parts[i]
            if visible_traces.sum() > 0:
                scc_median = np.median(np.array(sccs)[visible_traces][0:visible_traces.sum():2])
            else:
                scc_median = np.nan
            title = f"Models with SCC >= {str(round(scc_parts[i], 1))} (Median SCC: {str(round(scc_median, 2))})"
        else:
            visible_traces_gt = sccs >= scc_parts[i]
            visible_traces_lt = sccs < scc_parts[i + 1]
            visible_traces = visible_traces_gt & visible_traces_lt
            if visible_traces.sum() > 0:
                scc_median = np.median(np.array(sccs)[visible_traces][0:visible_traces.sum():2])
            else:
                scc_median = np.nan
            title = (f"Models with SCC between {str(round(scc_parts[i], 1))} and {str(round(scc_parts[i + 1], 1))}"
                     f" (Median SCC: {str(round(scc_median, 2))})")
        step = dict(
            method="update",
            args=[{"visible": visible_traces},
                  {"title": title}],
            label=str(round(scc_parts[i], 1))
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "SCC="},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title_font_size=20,
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            entrywidth=1.01
        )
    )
    fig.update_xaxes(range=[np.min(best_models["pred_df"]["y_true"]), np.max(best_models["pred_df"]["y_true"])])
    fig.update_yaxes(range=[np.min(best_models["pred_df"]["y_pred"]), np.max(best_models["pred_df"]["y_pred"])])

    fig.write_html(Path(dir_path + "scatter_plot_predictions.html"))


def cluster(best_models, dir_path):

    df = best_models["metric_df"]
    df = df[["pcc", "scc", "mse", "rmse"]]

    fig = dash_bio.Clustergram(
        data=df,
        cluster="row",
        # standardize="column",
        column_labels=list(df.columns.values),
        row_labels=list(df.index),
        height=1000,
        width=900,
        hidden_labels='row',
        center_values=False,
        color_threshold={
            'row': 0,
            'col': 0
        }
    )
    #fig.layout.autosize=True

    fig.write_html(Path(dir_path + "cluster.html"))


def f_statistic(best_models, nfeatures):
    # compute F statistic to see if fit is significant
    groups = best_models["pred_df"].groupby(by="target")
    F_stats = []
    p_values = []
    n_samples = []
    model_name = []
    for name, group in groups:
        ssreg = ((group["y_pred"] - group["y_true"].mean()) ** 2).sum()
        ssres = ((group["y_true"] - group["y_pred"]) ** 2).sum()
        k = best_models["models"][name].best_estimator_.coef_.shape[0]  # intercept B0 not included in k
        n = len(group)

        F_group = (ssreg / k) / (ssres / (n - k - 1))
        p_value = 1 - stats.f.cdf(F_group, k,
                                  n - k - 1)  # this returns nan if number of samples n < number of features k
        F_stats.append(F_group)
        p_values.append(p_value)
        n_samples.append(n)
        model_name.append(name)

    logger.info(f"Number of models with p_val < 0.05: {(np.array(p_values) < 0.05).sum()}"
                f" ({round((np.array(p_values) < 0.05).sum() / len(p_values), 3)}%)")
    fstat_df = pd.DataFrame(
        data={"model": model_name, "n_samples": n_samples, "k": nfeatures, "F_statistic": F_stats, "p_value": p_values})
    return fstat_df


def f_distribution(k, n, F_group, p_value):
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
