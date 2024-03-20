import dash_bio
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

logger = logging.getLogger(__name__)


def base_analysis(best_models, predictor, predictor_type, meta_data, dir_path):
    """
    Perform a basic analysis of the models. The analysis includes the following:
    - Summary statistics on the correlation coefficients (PCC, SCC)
    - Distribution of the correlation coefficients
    - performance metric vs variance
    - Avg number of datapoints per model
    """
    logger.info(
        f"\n\nSummary statistics on {meta_data['metadata'].get('task')} - {meta_data['metadata'].get('feature_type')}:\n"
        f"{best_models['metric_df'].describe()}\n")

    sns.set(style="ticks")

    if predictor_type == "classification":
        metric1 = "ROC-auc"
        metric2 = "MCC"
    elif predictor_type == "regression":
        metric1 = "pcc"
        metric2 = "scc"

    ### correlation coefficient distribution ###
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    sns.histplot(best_models["metric_df"][metric1], ax=axs[0], binrange=(-1, 1))
    sns.histplot(best_models["metric_df"][metric2], ax=axs[1], binrange=(-1, 1))
    median_value_metric1 = best_models["metric_df"][metric1].median()
    median_value_metric2 = best_models["metric_df"][metric2].median()
    axs[0].axvline(x=median_value_metric1, color='red', linestyle='dashed', linewidth=2, label='median')
    axs[1].axvline(x=median_value_metric2, color='red', linestyle='dashed', linewidth=2, label='median')
    axs[0].set_xlabel(metric1)
    axs[1].set_xlabel(metric2)
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

    ### metric2 vs variance ###
    if meta_data['metadata'].get('feature_type') == "fingerprints":
        metric2_val = best_models["metric_df"][metric2]
        metric1_val = best_models["metric_df"][metric1]
        drp = best_models["test_drp"]

        if predictor.task == "LPO":
            drp = best_models["test_drp"]
            drp = drp.pivot(index="Primary Cell Line Name", columns="Compound", values=best_models["metric"])
            var = drp.loc[metric2_val.index].var(axis=1)
        else:
            var = drp[metric2_val.index].var()

    elif meta_data["metadata"].get('feature_type') == "gene_expression":
        metric2_val = best_models["metric_df"][metric2]
        metric1_val = best_models["metric_df"][metric1]
        drp = best_models["test_drp"]

        if predictor.task == "LPO":
            drp = best_models["test_drp"].reset_index()
            drp = drp.pivot(index="Compound", columns="Primary Cell Line Name", values=best_models["metric"])
            var = drp.loc[metric2_val.index].var(axis=1)
        else:
            var = drp.loc[metric2_val.index].var(axis=1)

    fig, axs = plt.subplot_mosaic([['a)', 'b)']], figsize=(15, 10))
    axs['a)'].scatter(var, metric1_val)
    axs['b)'].scatter(var, metric2_val)
    plt.xlabel("variance")
    axs['a)'].set_ylabel(metric1)
    axs['b)'].set_ylabel(metric2)
    fig.suptitle(f"correlation coefficient vs variance "
                 f"{meta_data['metadata'].get('task')} - {meta_data['metadata'].get('feature_type')}",
                 fontsize=12, fontweight='bold')
    sns.despine(right=True)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig(Path(dir_path + "coefficient_variance_frequency.png"))
    plt.close()

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

    # checking for test set in balance in classification case
    if predictor_type == "classification":
        groups = best_models["pred_df"].groupby("target")
        value_counts_0 = groups.apply(lambda x: x.value_counts(subset=["y_true"])[0])
        value_counts_1 = groups.apply(lambda x: x.value_counts(subset=["y_true"])[1])
        in_balance = ((100 / (value_counts_0 + value_counts_1)) * value_counts_1).mean()
        logger.info(f"Test set in balance: {in_balance:.2f}% of the test set is of class label 1 (sensitive)")


def coef0_distribution(best_models, best_nfeatures, predictor_class, meta_data, dir_path):
    """
    Plot how often coefficients are set to 0. The plot is saved as a png file and the average number of coefficients set
    to 0 over all models is calculated and displayed.
    Note: This function is only applicable to linear models and SVMs with linear kernel.
    """
    ### analysing how many coef. set to 0 ###
    beta0_arr = []
    targets = []

    for target in best_models["models"]:
        if isinstance(best_models["models"].get(target), predictor_class):
            beta0_arr.append(np.reshape(best_models["models"].get(target).coef_ == 0), -1)
            targets.append(target)
        else:
            target_GCV = best_models["models"].get(target)
            beta0_arr.append(np.reshape(target_GCV.best_estimator_.coef_ == 0, -1))
            targets.append(target)

    beta0_df = pd.DataFrame(index=targets, data=beta0_arr)
    sns.barplot(x=beta0_df.sum().index, y=beta0_df.sum().values)
    plt.xlabel('coefficient number')
    plt.ylabel('count')
    plt.title(f'frequency of coefficients set to 0 ('
              f'{meta_data["metadata"]["task"]} - {meta_data["metadata"]["feature_type"]})', fontsize=12,
              fontweight='bold')
    sns.despine(right=True)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig(Path(dir_path + "coefficients0.png"))
    plt.close()

    logger.info(f"\nAverage number of coefficients set to 0 over all models: {beta0_df.T.sum().mean()}\n"
                f"Average number of coefficients not set to 0 over all models: {(best_nfeatures - beta0_df.T.sum()).mean()}\n"
                f"percentage of avg number of coefficients set to 0 over total coef: {beta0_df.T.sum().mean() / best_nfeatures * 100}\n"
                f"percentage of avg number of coefficients not set to 0 over total coef: {(best_nfeatures - beta0_df.T.sum()).mean() / best_nfeatures * 100}\n")


def scatter_predictions(best_models, dir_path):
    """
    Create a scatter plot of the predictions. The plot is interactive and allows the user to slide through the different
    models and see how well they performed. The plot is colored according to the target and the hover data shows the
    pcc and scc of the model. The plot is saved as a html file. The overall pcc and scc are also calculated and
    displayed in the title of the plot.
    """
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


def scores_clustering(best_models, dir_path, predictor_type):
    """
    Create a heatmap and clustermap of the model performance metrices calculated by the evaluate function. Clustering
    happens according to how well the models performed according to the scores in metric_df (ROC-auc, MCC, F1 etc.
    for classifier and pcc, scc, mse, rmse for regressor).
    """
    df = best_models["metric_df"].drop(columns=["nfeatures", "max_iter"])

    # dropping alpha since not really important in the context of the heatmap
    if "alpha" in df.columns:
        df = df.drop(columns=["alpha"])
    # dropping MCC since its value is between -1 and 1 and not between 0 and 1 like the other metrics
    # (and standardizing is diffucult since would have to remove outliers)
    if "MCC" in df.columns:
        df = df.drop(columns=["MCC"])

    df_median = df.median()

    formatted_strings = []
    for idx, val in df_median.items():
        formatted_string = f"{idx} = {val:.2f}"
        formatted_strings.append(formatted_string)
    title_string = "; ".join(formatted_strings)
    title_string = f"<br>Metric median: {title_string}"

    if predictor_type == "classification":
        # heatmap
        fig = px.imshow(df.sort_values(by='ROC-auc', ascending=True), text_auto=True, aspect="auto",
                        color_continuous_scale="plasma")
        fig.update_layout(
            title=title_string,
            xaxis_title="Metric",
            yaxis_title="target")

        fig.write_html(Path(dir_path + "scores_heatmap.html"))

        # cluster-map
        fig = go.Figure(dash_bio.Clustergram(
            data=df,
            cluster="row",
            # standardize="column",
            column_labels=list(df.columns.values),
            row_labels=list(df.index),
            height=1000,
            width=1600,
            # hidden_labels='row',
            center_values=False,
            color_map='plasma',
            display_ratio=[0.7]
        ))

        fig.update_layout(title=title_string)
        fig.write_html(Path(dir_path + "scores_clustermap.html"))

    if predictor_type == "regression":
        df_errors = df[["mse", "rmse"]].sort_values(by='mse', ascending=False)
        df_corrs = df[["pcc", "scc"]].sort_values(by='pcc', ascending=True)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Errors", "Correlations"), horizontal_spacing=0.15)

        heatmap_trace_error = go.Heatmap(z=df_errors.values, x=list(df_errors.columns), y=list(df_errors.index),
                                         colorscale="plasma", colorbar_x=0.44)
        heatmap_trace_corr = go.Heatmap(z=df_corrs.values, x=list(df_corrs.columns), y=list(df_corrs.index),
                                        colorscale="plasma", zmin=-1, zmax=1)

        fig = fig.add_traces([heatmap_trace_error, heatmap_trace_corr], rows=[1, 1], cols=[1, 2])
        fig.update_layout(height=1000, width=1600, title_text="Heatmap of the evaluation metrics" + title_string,
                          title_font_size=20, title_xanchor='center', title_x=0.5, title_font_family="Arial Black")

        fig.write_html(Path(dir_path + "scores_heatmap.html"))


def roc_plot(best_models, target_model):
    """
    Plot the ROC curve for a specific model. The ROC curve is created by plotting the true positive rate (TPR) against
    the false positive rate (FPR) at various threshold settings. The area under the curve (AUC) is also calculated.
    """
    df = best_models["pred_df"]
    groups = df.groupby("target")

    roc_curve_groups = groups.apply(lambda x: roc_curve(x["y_true"], x["y_prob"]))
    fpr, tpr, threshold = roc_curve_groups[target_model]

    if isinstance(target_model, int):
        model_name = roc_curve_groups.index[int(target_model)]
    else:
        model_name = target_model

    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, linestyle='--', color='black')
    plt.title("Receiver operating characteristic (ROC) curve")
    plt.legend([f"{model_name} curve (AUC = {roc_auc:.2f})", "random guess"], loc="lower right")
    plt.show()


def f_statistic(best_models, nfeatures):
    # compute F statistic to see if fit is significant
    """
    Compute the F-statistic and p-value for the models. The F-statistic is a measure of how well the independent
    variables explain the variance of the dependent variable. The p-value is the probability of observing an F-statistic
    as extreme as the one computed. The F-statistic is used to test the null hypothesis that the model with no
    independent variables fits the data as well as the model with the independent variables. If the p-value is less
    than the significance level, the null hypothesis is rejected. The F-statistic is computed as the ratio of the
    explained variance to the unexplained variance. The p-value is computed using the F-distribution.
    """
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
    """
    Plot the F-distribution and highlight the computed F-statistic and p-value.
    """
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


def decision_boundary(best_models, model_name, predictor_class):
    clf = best_models["models"][model_name]

    if not isinstance(clf, predictor_class):
        clf = clf.best_estimator_

    PC1 = best_models["data_dict"][model_name]["X_train"][:, 0]
    PC2 = best_models["data_dict"][model_name]["X_train"][:, 1]
    X = best_models["data_dict"][model_name]["X_train"][:, 0:2]
    y = best_models["data_dict"][model_name]["y_train"]
    # plt.scatter(PC1, PC2, c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()

    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        cmap=plt.cm.Paired,
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # plot support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=250,
        facecolors="none",
        edgecolors="k",
    )

    scatter = ax.scatter(PC1, PC2, c=y, s=150, edgecolors="k", cmap=plt.cm.Paired)
    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Maximum Margin Separating Hyperplane (2D PCA)')
    plt.show()
    plt.close()
