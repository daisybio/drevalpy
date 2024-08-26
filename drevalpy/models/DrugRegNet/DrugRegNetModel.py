import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
from scipy import stats


class DrugRegNetModel:
    def __init__(self, path_drug_response, path_dysregnet_scores, features):
        self.drug_response = pd.read_csv(path_drug_response, index_col=0).T
        self.dysregnet_scores = pd.read_feather(path_dysregnet_scores)
        self.dysregnet_scores = self.dysregnet_scores.set_index("patient id")
        self.features = features

    def create_train_data(self):
        all_data = dict()
        for drug in self.drug_response.columns:
            print("Creating train data for drug:", drug)
            drp = self.drug_response[drug]
            drp = drp[~drp.isna()]
            drp = drp[~drp.index.duplicated(keep="first")]
            drp = drp[drp.index.isin(self.dysregnet_scores.index)]
            x = self.dysregnet_scores.loc[drp.index]
            x = self.feature_selection(x)
            all_data[drug] = DrugRegNetDataset(drug, x, drp)
        setattr(self, "all_data", all_data)

    def feature_selection(self, x, n_features=300):
        if self.features == "topN":
            # get the n_features columns with the highest variance
            x = x.loc[:, x.var().nlargest(n_features).index]
        return x

    def train_model(self):
        for drug in self.all_data.keys():
            print("Training model for drug:", drug)
            x = self.all_data[drug].x
            y = self.all_data[drug].y
            # TODO: cross validation?
            model = Lasso(alpha=0.1)
            model.fit(x, y)
            # get p-values for coefficients
            p_values = self.calculate_pvalues(model, x, y)
            # do Bonferroni correction by getting minimum of p-value * number of features and 1
            p_adj = np.minimum(p_values * x.shape[1], 1)
            result_df = pd.DataFrame(
                {
                    "edge": x.columns,
                    "coef": model.coef_,
                    "p_val": p_values,
                    "p_adj": p_adj,
                }
            )
            setattr(model, "results", result_df)
            setattr(self, drug, model)

    @staticmethod
    def calculate_pvalues(model, x, y):
        params = np.append(model.intercept_, model.coef_)
        predictions = model.predict(x)
        newX = pd.DataFrame({"Constant": np.ones(len(x))}, index=x.index).join(x)
        MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))
        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b
        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]
        p_values = np.round(p_values, 3)
        p_values = p_values[1:]
        return p_values

    def export_results(self, path):
        for drug in self.all_data.keys():
            result_df = getattr(self, drug).results
            # order by p-value
            result_df = result_df.sort_values("p_val")
            result_df.to_csv(path + "/results_" + drug + ".csv")
            drug_specific_network = result_df[result_df["p_val"] < 0.5]
            # only get edge column
            if not drug_specific_network.empty:
                drug_specific_network = drug_specific_network["edge"]
                # split column such that (g1, g2) becomes g1 and g2
                drug_specific_network = drug_specific_network.str.replace(
                    "(", ""
                ).str.replace(")", "")
                drug_specific_network = drug_specific_network.str.replace("'", "")
                drug_specific_network = drug_specific_network.str.split(
                    ", ", expand=True
                )
                drug_specific_network.columns = ["intA", "intB"]
                drug_specific_network.to_csv(
                    path + "/network_" + drug + ".csv", index=False
                )


class DrugRegNetDataset:
    def __init__(self, drug, x, y):
        self.drug = drug
        self.x = x
        self.y = y


if __name__ == "__main__":
    model = DrugRegNetModel(
        "../../data/response_output/CCLE/curve_curator_pEC50_CCLE.csv",
        "../../data/cell_line_input/DysRegNet/ccle_fake.fea",
        features="topN",
    )
    model.create_train_data()
    model.train_model()
    model.export_results("results")
