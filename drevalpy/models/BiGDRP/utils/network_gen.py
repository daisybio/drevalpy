import pandas as pd
import numpy as np
import dgl
import torch


def create_network(tuples, percentile=1):

    sen_edges = pd.DataFrame()
    res_edges = pd.DataFrame()
    for drug in tuples["drug"].unique():
        drug_edges = tuples.loc[tuples["drug"] == drug]
        thresh = np.percentile(drug_edges["response"], percentile)
        sen_edges = pd.concat(
            [sen_edges, drug_edges.loc[drug_edges["response"] < thresh]]
        )
        thresh = np.percentile(drug_edges["response"], (100 - percentile))
        res_edges = pd.concat(
            [res_edges, drug_edges.loc[drug_edges["response"] > thresh]]
        )

    print(
        "generated a network with %d sensitive edges and %d resistant edges "
        % (len(sen_edges), len(res_edges))
    )

    graph_data = {
        ("cell_line", "is_sensitive", "drug"): (
            sen_edges["cell_line"].values,
            sen_edges["drug"].values,
        ),
        ("drug", "is_effective", "cell_line"): (
            sen_edges["drug"].values,
            sen_edges["cell_line"].values,
        ),
        ("cell_line", "is_resistant", "drug"): (
            res_edges["cell_line"].values,
            res_edges["drug"].values,
        ),
        ("drug", "is_ineffective", "cell_line"): (
            res_edges["drug"].values,
            res_edges["cell_line"].values,
        ),
    }
    network = dgl.heterograph(graph_data)
    print(network)
    cl = list(set(sen_edges["cell_line"]).union(set(res_edges["cell_line"])))
    print("unique_CLs", len(cl))

    if len(cl) != tuples["cell_line"].nunique():
        network.add_nodes(tuples["cell_line"].max() - max(cl), ntype="cell_line")
    return network
