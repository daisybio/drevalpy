import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from drevalpy.models import MODEL_FACTORY
from drevalpy.models.utils import (
    get_multiomics_feature_dataset,
    iterate_features,
    load_and_reduce_gene_features,
    load_cl_ids_from_csv,
    load_drug_fingerprint_features,
    load_drug_ids_from_csv,
    unique,
)


def test_factory():
    assert "NaivePredictor" in MODEL_FACTORY
    assert "NaiveDrugMeanPredictor" in MODEL_FACTORY
    assert "NaiveCellLineMeanPredictor" in MODEL_FACTORY
    assert "ElasticNet" in MODEL_FACTORY
    assert "RandomForest" in MODEL_FACTORY
    assert "SVR" in MODEL_FACTORY
    assert "SimpleNeuralNetwork" in MODEL_FACTORY
    assert "MultiOmicsNeuralNetwork" in MODEL_FACTORY
    assert "MultiOmicsRandomForest" in MODEL_FACTORY
    assert "SingleDrugRandomForest" in MODEL_FACTORY
    assert "SRMF" in MODEL_FACTORY
    assert "GradientBoosting" in MODEL_FACTORY
    assert len(MODEL_FACTORY) == 12


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
    temp_file = os.path.join(temp_dir.name, "GDSC1_small", "gene_lists", f"{gene_list}.csv")
    if gene_list == "landmark_genes":
        with open(temp_file, "w") as f:
            f.write(
                "Entrez ID,Symbol,Name,Gene Family,Type,RNA-Seq Correlation,RNA-Seq Correlation Self-Rank\n"
                "3638,INSIG1,insulin induced gene 1,,landmark,,\n"
                "2309,FOXO3,forkhead box O3,Forkhead boxes,landmark,,\n"
                '672,BRCA1,"BRCA1, DNA repair associated","Ring finger proteins, Fanconi anemia complementation groups'
                ',Protein phosphatase 1 regulatory subunits, BRCA1 A complex, BRCA1 B complex, BRCA1 C complex",'
                "landmark,,\n57147,SCYL3,SCY1 like pseudokinase 3,SCY1 like pseudokinases,landmark,,"
            )
    elif gene_list == "drug_target_genes_all_drugs":
        with open(temp_file, "w") as f:
            f.write("Symbol\n" "TSPAN6\n" "SCYL3\n" "BRCA1\n")
    elif gene_list == "gene_list_paccmann_network_prop":
        with open(temp_file, "w") as f:
            f.write("Symbol\n" "HDAC1\n" "ALS2CR12\n" "BFAR\n" "ZCWPW1\n" "ZP1\n" "PDZD7")


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
            "CELL_LINE_NAME,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
            "CAL-120,7.632023171463389,2.9645851205892404,10.3795526353077,3.61479404843988,"
            "3.38068143582194,7.09344749430946,3.0222634357817597\n"
            "DMS 114,7.54867116637172,2.77771614989839,11.807341248845802,4.066886747621,"
            "3.73248465377029,2.8016127581695,6.07851099764176\n"
            "CAL-51,8.71233752103624,2.6435077554121,9.88073281995499,3.95622995046262,"
            "3.23662007804984,11.394340478134598,4.22471584953505\n"
            "NCI-H2869,7.79714221650204,2.8179230218265,9.88347076381233,4.0637013909818505,"
            "3.55841402145301,8.76055372116888,4.33420904819493\n"
            "22Rv1,4.8044868436701,2.84812776692645,10.3319941550002,5.14538669275316,"
            "3.54519297942073,3.9337949618623704,2.8629939819029904\n"
        )
    if gene_list is not None:
        write_gene_list(temp, gene_list)

    if gene_list == "gene_list_paccmann_network_prop":
        with pytest.raises(ValueError) as valerr:
            gene_features_gdsc1 = load_and_reduce_gene_features("gene_expression", gene_list, temp.name, "GDSC1_small")
    else:
        gene_features_gdsc1 = load_and_reduce_gene_features("gene_expression", gene_list, temp.name, "GDSC1_small")
    if gene_list is None:
        assert len(gene_features_gdsc1.features) == 5
        assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 7
        gene_names = ["TSPAN6", "TNMD", "BRCA1", "SCYL3", "HDAC1", "INSIG1", "FOXO3"]
        assert np.all(gene_features_gdsc1.meta_info["gene_expression"] == gene_names)
    elif gene_list == "landmark_genes":
        assert len(gene_features_gdsc1.features) == 5
        assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 4
        colnames = gene_features_gdsc1.meta_info["gene_expression"]
        colnames.sort()
        assert np.all(colnames == ["BRCA1", "FOXO3", "INSIG1", "SCYL3"])
    elif gene_list == "drug_target_genes_all_drugs":
        assert len(gene_features_gdsc1.features) == 5
        assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 3
        colnames = gene_features_gdsc1.meta_info["gene_expression"]
        colnames.sort()
        assert np.all(colnames == ["BRCA1", "SCYL3", "TSPAN6"])
    elif gene_list == "gene_list_paccmann_network_prop":
        assert "The following genes are missing from the dataset GDSC1_small" in str(valerr.value)


def test_iterate_features():
    df = pd.DataFrame({"GeneA": [1, 2, 3, 2], "GeneB": [4, 5, 6, 2], "GeneC": [7, 8, 9, 2]})
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
    assert np.all(drug_features_gdsc1.features["Zibotentan"]["fingerprints"] == [1, 1, 0, 1, 1])


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
            "CELL_LINE_NAME,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
            "CAL-120,7.632023171463389,2.9645851205892404,10.3795526353077,3.61479404843988,"
            "3.38068143582194,7.09344749430946,3.0222634357817597\n"
            "DMS 114,7.54867116637172,2.77771614989839,11.807341248845802,4.066886747621,"
            "3.73248465377029,2.8016127581695,6.07851099764176\n"
            "CAL-51,8.71233752103624,2.6435077554121,9.88073281995499,3.95622995046262,"
            "3.23662007804984,11.394340478134598,4.22471584953505\n"
            "NCI-H2869,7.79714221650204,2.8179230218265,9.88347076381233,4.0637013909818505,"
            "3.55841402145301,8.76055372116888,4.33420904819493\n"
            "22Rv1,4.8044868436701,2.84812776692645,10.3319941550002,5.14538669275316,"
            "3.54519297942073,3.9337949618623704,2.8629939819029904\n"
        )

    # methylation
    temp_file = os.path.join(temp.name, "GDSC1_small", "methylation.csv")
    with open(temp_file, "w") as f:
        f.write(
            "CELL_LINE_NAME,chr1:10003165-10003585,chr1:100315420-100316009,"
            "chr1:100435297-100436070,chr1:100503482-100504404,chr1:10057121-10058108,"
            "chr11:107728949-107729586,chr11:107798958-107799980\n"
            "22Rv1,0.192212286,0.20381998,0.277913619,0.1909300789999999,0.544058696\n"
            "PFSK-1,0.1876026089999999,0.2076517789999999,0.400145531,0.195871473,0.76489757\n"
            "CAL-120,0.2101851619999999,0.222116189,0.264730199,0.243298011,0.415484752\n"
            "ES3,0.205613728,0.227570131,0.303640813,0.250454389,0.599274902\n"
        )
    # mutations
    temp_file = os.path.join(temp.name, "GDSC1_small", "mutations.csv")
    with open(temp_file, "w") as f:
        f.write(
            "CELL_LINE_NAME,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
            "201T,False,False,False,False,False,True,True\n"
            "22Rv1,False,True,False,True,False,False,True\n"
            "23132/87,False,False,True,True,False,False,False\n"
            "CAL-120,False,False,False,False,False,True,False\n"
        )

    # copy number variation
    temp_file = os.path.join(temp.name, "GDSC1_small", "copy_number_variation_gistic.csv")
    with open(temp_file, "w") as f:
        f.write(
            "CELL_LINE_NAME,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
            "201T,0.0,0.0,-1.0,0.0,0.0,1.0,-1.0\n"
            "TE-12,-1.0,-1.0,0.0,1.0,1.0,0.0,0.0\n"
            "CAL-120,0.0,0.0,0.0,-1.0,-1.0,1.0,0.0\n"
            "STS-0421,0.0,0.0,1.0,0.0,0.0,-1.0,0.0\n"
            "22Rv1,1.0,1.0,-1.0,1.0,1.0,1.0,1.0\n"
        )
    if gene_list is not None:
        write_gene_list(temp, gene_list)
    if gene_list == "gene_list_paccmann_network_prop":
        with pytest.raises(ValueError) as valerr:
            dataset = get_multiomics_feature_dataset(
                data_path=temp.name,
                dataset_name="GDSC1_small",
                gene_list=gene_list,
            )
    else:
        dataset = get_multiomics_feature_dataset(
            data_path=temp.name,
            dataset_name="GDSC1_small",
            gene_list=gene_list,
        )
        assert len(dataset.features) == 2
        common_cls = dataset.get_ids()
        common_cls.sort()
        assert np.all(common_cls == ["22Rv1", "CAL-120"])
        assert len(dataset.meta_info) == 4
    if gene_list is None:
        assert np.all(
            dataset.meta_info["gene_expression"] == ["TSPAN6", "TNMD", "BRCA1", "SCYL3", "HDAC1", "INSIG1", "FOXO3"]
        )
        for key in dataset.meta_info:
            assert len(dataset.meta_info[key]) == 7
    elif gene_list == "landmark_genes":
        feature_names = []
        for key in dataset.meta_info:
            if key == "methylation":
                assert len(dataset.meta_info[key]) == 7
            else:
                assert len(dataset.meta_info[key]) == 4
                if len(feature_names) == 0:
                    feature_names = dataset.meta_info[key]
                else:
                    assert np.all(dataset.meta_info[key] == feature_names)
    elif gene_list == "drug_target_genes_all_drugs":
        feature_names = []
        for key in dataset.meta_info:
            if key == "methylation":
                assert len(dataset.meta_info[key]) == 7
            else:
                assert len(dataset.meta_info[key]) == 3
                if len(feature_names) == 0:
                    feature_names = dataset.meta_info[key]
                else:
                    assert np.all(dataset.meta_info[key] == feature_names)
    elif gene_list == "gene_list_paccmann_network_prop":
        assert "The following genes are missing from the dataset GDSC1_small" in str(valerr.value)


def test_unique():
    array = np.array([1, 9, 3, 2, 1, 4, 5, 6, 7, 8, 9, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    unique_array = unique(array)
    assert np.all(unique_array == np.array([1, 9, 3, 2, 4, 5, 6, 7, 8]))


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
