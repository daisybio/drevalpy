import os
import pytest
import pandas as pd
import numpy as np
import tempfile

from drevalpy.models import MODEL_FACTORY
from drevalpy.models.utils import (
    load_cl_ids_from_csv,
    load_and_reduce_gene_features,
    iterate_features,
    load_drug_ids_from_csv,
    load_drug_fingerprint_features,
    get_multiomics_feature_dataset,
    unique,
)


def test_factory():
    assert "SimpleNeuralNetwork" in MODEL_FACTORY
    assert "MultiOmicsNeuralNetwork" in MODEL_FACTORY
    assert "ElasticNet" in MODEL_FACTORY
    assert "RandomForest" in MODEL_FACTORY
    assert "MultiOmicsRandomForest" in MODEL_FACTORY
    assert "SVR" in MODEL_FACTORY
    assert "NaivePredictor" in MODEL_FACTORY
    assert "NaiveDrugMeanPredictor" in MODEL_FACTORY
    assert "NaiveCellLineMeanPredictor" in MODEL_FACTORY
    assert "SingleDrugRandomForest" in MODEL_FACTORY
    assert len(MODEL_FACTORY) == 10


def test_load_cl_ids_from_csv():
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    temp_file = os.path.join(temp.name, "GDSC1_small", "cell_line_names.csv")
    with open(temp_file, "w") as f:
        f.write("CELL_LINE_NAME\n201T\n22RV1\n23132/87\n42-MG-BA\n")

    cl_ids_gdsc1 = load_cl_ids_from_csv(temp.name, "GDSC1_small")
    assert len(cl_ids_gdsc1.features) == 4
    assert cl_ids_gdsc1.identifiers[0] == "201T"


def write_gene_list(temp_dir, gene_list):
    os.mkdir(os.path.join(temp_dir.name, "GDSC1_small", "gene_lists"))
    temp_file = os.path.join(
        temp_dir.name, "GDSC1_small", "gene_lists", f"{gene_list}.csv"
    )
    if gene_list == "landmark_genes":
        with open(temp_file, "w") as f:
            f.write(
                "Entrez ID\tSymbol\tName\tGene Family\tType\tRNA-Seq Correlation\tRNA-Seq Correlation Self-Rank\n"
                "3638\tINSIG1\tinsulin induced gene 1\t\tlandmark\t\t\n"
                "2309\tFOXO3\tforkhead box O3\tForkhead boxes\tlandmark\t\t\n"
                "7105\tTSPAN6\ttetraspanin 6\tTetraspanins\tlandmark\t\t\n"
                "57147\tSCYL3\tSCY1 like pseudokinase 3\tSCY1 like pseudokinases\tlandmark\t\t"
            )
    elif gene_list == "drug_target_genes_all_drugs":
        with open(temp_file, "w") as f:
            f.write(
                ",Symbol\n"
                "0,EGFR\n"
                "1,MTOR\n"
                "2,KIT\n"
                "3,FLT3\n"
                "4,RET\n"
                "5,BRCA1\n"
            )
    elif gene_list == "gene_list_paccmann_network_prop":
        with open(temp_file, "w") as f:
            f.write(
                ",Symbol\n"
                "0,HDAC1\n"
                "1,ALS2CR12\n"
                "2,BFAR\n"
                "3,ZCWPW1\n"
                "4,ZP1\n"
                "5,PDZD7"
            )


@pytest.mark.parametrize(
    "gene_list",
    [
        None,
        "landmark_genes",
        "drug_target_genes_all_drugs",
        "gene_list_paccmann_network_prop",
    ],
)
def test_load_and_reduce_gene_features(gene_list):
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    temp_file = os.path.join(temp.name, "GDSC1_small", "gene_expression.csv")
    with open(temp_file, "w") as f:
        f.write(
            "CELL_LINE_NAME,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1\n"
            "CAL-120,7.632023171463389,2.9645851205892404,10.3795526353077,3.61479404843988,3.38068143582194\n"
            "DMS 114,7.54867116637172,2.77771614989839,11.807341248845802,4.066886747621,3.73248465377029\n"
            "CAL-51,8.71233752103624,2.6435077554121,9.88073281995499,3.95622995046262,3.23662007804984\n"
            "NCI-H2869,7.79714221650204,2.8179230218265,9.88347076381233,4.0637013909818505,3.55841402145301\n"
            "22Rv1,4.8044868436701,2.84812776692645,10.3319941550002,5.14538669275316,3.54519297942073\n"
        )
    if gene_list is not None:
        write_gene_list(temp, gene_list)

    gene_features_gdsc1 = load_and_reduce_gene_features(
        "gene_expression", gene_list, temp.name, "GDSC1_small"
    )
    if gene_list is None:
        assert len(gene_features_gdsc1.features) == 5
        assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 5
        assert np.all(
            gene_features_gdsc1.meta_info["gene_expression"]
            == ["TSPAN6", "TNMD", "BRCA1", "SCYL3", "HDAC1"]
        )
    elif gene_list == "landmark_genes":
        assert len(gene_features_gdsc1.features) == 5
        assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 2
        colnames = gene_features_gdsc1.meta_info["gene_expression"]
        colnames.sort()
        assert np.all(colnames == ["SCYL3", "TSPAN6"])
    elif gene_list == "drug_target_genes_all_drugs":
        assert len(gene_features_gdsc1.features) == 5
        assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 1
        assert np.all(gene_features_gdsc1.meta_info["gene_expression"] == ["BRCA1"])
    elif gene_list == "gene_list_paccmann_network_prop":
        assert len(gene_features_gdsc1.features) == 5
        assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 1
        assert np.all(gene_features_gdsc1.meta_info["gene_expression"] == ["HDAC1"])


def test_iterate_features():
    df = pd.DataFrame(
        {"GeneA": [1, 2, 3, 2], "GeneB": [4, 5, 6, 2], "GeneC": [7, 8, 9, 2]}
    )
    df.index = ["CellLine1", "CellLine2", "CellLine3", "CellLine1"]
    with pytest.warns(UserWarning):
        features = iterate_features(df, "gene_expression")
    assert len(features) == 3
    assert np.all(features["CellLine1"]["gene_expression"] == [1, 4, 7])


def test_load_drug_ids_from_csv():
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    temp_file = os.path.join(temp.name, "GDSC1_small", "drug_names.csv")
    with open(temp_file, "w") as f:
        f.write("DRUG_NAME\n(5Z)-7-Oxozeaenol\n5-Fluorouracil\nA-443654\nA-770041\n")
    drug_ids_gdsc1 = load_drug_ids_from_csv(temp.name, "GDSC1_small")
    assert len(drug_ids_gdsc1.features) == 4
    assert drug_ids_gdsc1.identifiers[0] == "(5Z)-7-Oxozeaenol"


def test_load_drugs_from_fingerprints():
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    os.mkdir(os.path.join(temp.name, "GDSC1_small", "drug_fingerprints"))
    temp_file = os.path.join(
        temp.name,
        "GDSC1_small",
        "drug_fingerprints",
        "drug_name_to_demorgan_128_map.csv",
    )
    with open(temp_file, "w") as f:
        f.write(
            ",Zibotentan,AZD1208,CI-1040,A-83-01,GSK269962A\n"
            "0,1,1,1,1,1\n"
            "1,1,1,0,0,1\n"
            "2,0,1,1,0,1\n"
            "3,1,0,1,1,1\n"
            "4,1,1,0,1,1\n"
        )
    drug_features_gdsc1 = load_drug_fingerprint_features(temp.name, "GDSC1_small")
    assert len(drug_features_gdsc1.features) == 5
    assert drug_features_gdsc1.features.keys() == {
        "Zibotentan",
        "AZD1208",
        "CI-1040",
        "A-83-01",
        "GSK269962A",
    }
    assert np.all(
        drug_features_gdsc1.features["Zibotentan"]["fingerprints"] == [1, 1, 0, 1, 1]
    )


@pytest.mark.parametrize(
    "gene_list",
    [
        None,
        "landmark_genes",
        "drug_target_genes_all_drugs",
        "gene_list_paccmann_network_prop",
    ],
)
def test_get_multiomics_feature_dataset(gene_list):
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    # gene expression
    temp_file = os.path.join(temp.name, "GDSC1_small", "gene_expression.csv")
    with open(temp_file, "w") as f:
        f.write(
            "CELL_LINE_NAME,TSPAN6,BRCA1,DPM1,SCYL3,HDAC1\n"
            "CAL-120,7.632023171463389,2.9645851205892404,10.3795526353077,3.61479404843988,3.38068143582194\n"
            "DMS 114,7.54867116637172,2.77771614989839,11.807341248845802,4.066886747621,3.73248465377029\n"
            "CAL-51,8.71233752103624,2.6435077554121,9.88073281995499,3.95622995046262,3.23662007804984\n"
            "PFSK-1,7.79714221650204,2.8179230218265,9.88347076381233,4.0637013909818505,3.55841402145301\n"
            "22Rv1,4.8044868436701,2.84812776692645,10.3319941550002,5.14538669275316,3.54519297942073\n"
        )

    # methylation
    temp_file = os.path.join(temp.name, "GDSC1_small", "methylation.csv")
    with open(temp_file, "w") as f:
        f.write(
            "CELL_LINE_NAME,chr1:10003165-10003585,chr1:100315420-100316009,chr1:100435297-100436070,chr1:100503482-100504404,chr1:10057121-10058108\n"
            "22Rv1,0.192212286,0.20381998,0.277913619,0.1909300789999999,0.544058696\n"
            "PFSK-1,0.1876026089999999,0.2076517789999999,0.400145531,0.195871473,0.76489757\n"
            "CAL-120,0.2101851619999999,0.222116189,0.264730199,0.243298011,0.415484752\n"
            "ES3,0.205613728,0.227570131,0.303640813,0.250454389,0.599274902\n"
        )
    # mutations
    temp_file = os.path.join(temp.name, "GDSC1_small", "mutations.csv")
    with open(temp_file, "w") as f:
        f.write(
            "CELL_LINE_NAME,TSPAN6,BRCA1,DPM1,SCYL3,HDAC1\n"
            "201T,False,False,False,False,False\n"
            "22Rv1,False,True,False,True,False\n"
            "23132/87,False,False,True,True,False\n"
            "CAL-120,False,False,False,False,False\n"
        )

    # copy number variation
    temp_file = os.path.join(
        temp.name, "GDSC1_small", "copy_number_variation_gistic.csv"
    )
    with open(temp_file, "w") as f:
        f.write(
            "CELL_LINE_NAME,TSPAN6,BRCA1,DPM1,SCYL3,HDAC1\n"
            "201T,0.0,0.0,-1.0,0.0,0.0\n"
            "TE-12,-1.0,-1.0,0.0,1.0,1.0\n"
            "CAL-120,0.0,0.0,0.0,-1.0,-1.0\n"
            "STS-0421,0.0,0.0,1.0,0.0,0.0\n"
            "22Rv1,1.0,1.0,-1.0,1.0,1.0\n"
        )
    if gene_list is not None:
        write_gene_list(temp, gene_list)
    dataset = get_multiomics_feature_dataset(
        data_path=temp.name, dataset_name="GDSC1_small", gene_list=gene_list
    )
    assert len(dataset.features) == 2
    common_cls = dataset.get_ids()
    common_cls.sort()
    assert common_cls == ["22Rv1", "CAL-120"]
    assert len(dataset.meta_info) == 4
    if gene_list is None:
        assert np.all(
            dataset.meta_info["gene_expression"]
            == ["TSPAN6", "BRCA1", "DPM1", "SCYL3", "HDAC1"]
        )
        for key in dataset.meta_info:
            assert len(dataset.meta_info[key]) == 5
    elif gene_list == "landmark_genes":
        feature_names = []
        for key in dataset.meta_info:
            if key == "methylation":
                assert len(dataset.meta_info[key]) == 5
            else:
                assert len(dataset.meta_info[key]) == 2
                if len(feature_names) == 0:
                    feature_names = dataset.meta_info[key]
                else:
                    assert np.all(dataset.meta_info[key] == feature_names)
    elif gene_list == "drug_target_genes_all_drugs":
        for key in dataset.meta_info:
            if key == "methylation":
                assert len(dataset.meta_info[key]) == 5
            else:
                assert len(dataset.meta_info[key]) == 1
                assert np.all(dataset.meta_info[key] == ["BRCA1"])
    elif gene_list == "gene_list_paccmann_network_prop":
        for key in dataset.meta_info:
            if key == "methylation":
                assert len(dataset.meta_info[key]) == 5
            else:
                assert len(dataset.meta_info[key]) == 1
                assert np.all(dataset.meta_info[key] == ["HDAC1"])


def test_unique():
    array = np.array([1, 9, 3, 2, 1, 4, 5, 6, 7, 8, 9, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    unique_array = unique(array)
    assert np.all(unique_array == np.array([1, 9, 3, 2, 4, 5, 6, 7, 8]))


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
