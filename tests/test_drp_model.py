"""Tests for the DRPModel."""

import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from drevalpy.datasets.loader import load_toyv1, load_toyv2
from drevalpy.datasets.utils import TISSUE_IDENTIFIER
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.utils import (
    get_multiomics_feature_dataset,
    iterate_features,
    load_and_select_gene_features,
    load_cl_ids_from_csv,
    load_drug_fingerprint_features,
    load_drug_ids_from_csv,
    load_tissues_from_csv,
    unique,
)


def test_factory() -> None:
    """Test the model factory."""
    assert "NaivePredictor" in MODEL_FACTORY
    assert "NaiveDrugMeanPredictor" in MODEL_FACTORY
    assert "NaiveCellLineMeanPredictor" in MODEL_FACTORY
    assert "NaiveMeanEffectsPredictor" in MODEL_FACTORY
    assert "ElasticNet" in MODEL_FACTORY
    assert "RandomForest" in MODEL_FACTORY
    assert "SVR" in MODEL_FACTORY
    assert "SimpleNeuralNetwork" in MODEL_FACTORY
    assert "MultiOmicsNeuralNetwork" in MODEL_FACTORY
    assert "MultiOmicsRandomForest" in MODEL_FACTORY
    assert "SingleDrugRandomForest" in MODEL_FACTORY
    assert "SRMF" in MODEL_FACTORY
    assert "GradientBoosting" in MODEL_FACTORY
    assert "MOLIR" in MODEL_FACTORY
    assert "SuperFELTR" in MODEL_FACTORY
    assert "DIPK" in MODEL_FACTORY


def test_load_cl_ids_from_csv() -> None:
    """Test the loading of cell line identifiers from a CSV file."""
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    temp_file = os.path.join(temp.name, "GDSC1_small", "cell_line_names.csv")
    with open(temp_file, "w") as f:
        f.write(
            "cellosaurus_id,CELL_LINE_NAME\nCVCL_X481,201T\nCVCL_1045,22Rv1\n"
            "CVCL_1046,23132/87\nCVCL_1798,42-MG-BA\n"
        )

    cl_ids_gdsc1 = load_cl_ids_from_csv(temp.name, "GDSC1_small")
    assert len(cl_ids_gdsc1.features) == 4
    assert cl_ids_gdsc1.identifiers[0] == "201T"


def test_load_tissues_from_csv() -> None:
    """Test the loading of tissues from a CSV file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.mkdir(os.path.join(temp_dir, "GDSC1_small"))
        temp_file = os.path.join(temp_dir, "GDSC1_small", "cell_line_names.csv")
        with open(temp_file, "w") as f:
            f.write(
                "cellosaurus_id,CELL_LINE_NAME,tissue\n"
                "CVCL_X481,201T,lung\n"
                "CVCL_1045,22Rv1,breast\n"
                "CVCL_1046,23132/87,liver\n"
                "CVCL_1798,42-MG-BA,kidney\n"
            )

        tissues_gdsc1 = load_tissues_from_csv(temp_dir, "GDSC1_small")
        assert len(tissues_gdsc1.features) == 4

        expected = {
            "201T": "lung",
            "22Rv1": "breast",
            "23132/87": "liver",
            "42-MG-BA": "kidney",
        }

        for cl_name, expected_tissue in expected.items():
            tissue_value = tissues_gdsc1.features[cl_name][TISSUE_IDENTIFIER]
            assert isinstance(tissue_value, np.ndarray)
            assert tissue_value.shape == (1,)
            assert tissue_value[0] == expected_tissue


def _write_gene_list(temp_dir: tempfile.TemporaryDirectory, gene_list: Optional[str] = None) -> None:
    """
    Write a gene list to a temporary directory.

    :param temp_dir: temporary directory
    :param gene_list: either None, landmark_genes, drug_target_genes_all_drugs, or gene_list_paccmann_network_prop
    """
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
def test_load_and_select_gene_features(gene_list: Optional[str]) -> None:
    """
    Test the loading and reduction of gene features.

    :param gene_list: either None, landmark_genes, drug_target_genes_all_drugs, or gene_list_paccmann_network_prop
    """
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    temp_file = os.path.join(temp.name, "GDSC1_small", "gene_expression.csv")
    with open(temp_file, "w") as f:
        f.write(
            "cellosaurus_id,CELL_LINE_NAME,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
            "CVCL_1104,CAL-120,7.632023171463389,2.9645851205892404,10.3795526353077,3.61479404843988,"
            "3.38068143582194,7.09344749430946,3.0222634357817597\n"
            "CVCL_1174,DMS 114,7.54867116637172,2.77771614989839,11.807341248845802,4.066886747621,"
            "3.73248465377029,2.8016127581695,6.07851099764176\n"
            "CVCL_1110,CAL-51,8.71233752103624,2.6435077554121,9.88073281995499,3.95622995046262,"
            "3.23662007804984,11.394340478134598,4.22471584953505\n"
            "CVCL_V001,NCI-H2869,7.79714221650204,2.8179230218265,9.88347076381233,4.0637013909818505,"
            "3.55841402145301,8.76055372116888,4.33420904819493\n"
            "CVCL_1045,22Rv1,4.8044868436701,2.84812776692645,10.3319941550002,5.14538669275316,"
            "3.54519297942073,3.9337949618623704,2.8629939819029904\n"
        )
    if gene_list is not None:
        _write_gene_list(temp, gene_list)

    if gene_list == "gene_list_paccmann_network_prop":
        with pytest.raises(ValueError) as valerr:
            gene_features_gdsc1 = load_and_select_gene_features("gene_expression", gene_list, temp.name, "GDSC1_small")
    else:
        gene_features_gdsc1 = load_and_select_gene_features("gene_expression", gene_list, temp.name, "GDSC1_small")
    if gene_list is None:
        assert len(gene_features_gdsc1.features) == 5
        assert gene_features_gdsc1.meta_info is not None
        assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 7
        gene_names = ["TSPAN6", "TNMD", "BRCA1", "SCYL3", "HDAC1", "INSIG1", "FOXO3"]
        assert np.all(gene_features_gdsc1.meta_info["gene_expression"] == gene_names)
    elif gene_list == "landmark_genes":
        assert len(gene_features_gdsc1.features) == 5
        assert gene_features_gdsc1.meta_info is not None
        assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 4
        colnames = gene_features_gdsc1.meta_info["gene_expression"]
        colnames.sort()
        assert np.all(colnames == ["BRCA1", "FOXO3", "INSIG1", "SCYL3"])
    elif gene_list == "drug_target_genes_all_drugs":
        assert len(gene_features_gdsc1.features) == 5
        assert gene_features_gdsc1.meta_info is not None
        assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 3
        colnames = gene_features_gdsc1.meta_info["gene_expression"]
        colnames.sort()
        assert np.all(colnames == ["BRCA1", "SCYL3", "TSPAN6"])
    elif gene_list == "gene_list_paccmann_network_prop":
        assert "The following genes are missing from the dataset GDSC1_small" in str(valerr.value)


def test_order_load_and_select_gene_features() -> None:
    """Test the order of the features after loading and reducing gene features. it should be maintained."""
    path_data = os.path.join("..", "data")

    load_toyv1(path_data)
    load_toyv2(path_data)
    gene_list = "gene_expression_intersection"
    a = load_and_select_gene_features("gene_expression", gene_list, path_data, "TOYv1")
    b = load_and_select_gene_features("gene_expression", gene_list, path_data, "TOYv2")
    # assert the meta info (=gene names) are the same
    assert np.all(a.meta_info["gene_expression"] == b.meta_info["gene_expression"])
    # assert the shape of the features for a random cell line is actually the same
    random_cell_line_a = np.random.choice(a.identifiers)
    random_cell_line_b = np.random.choice(b.identifiers)
    assert (
        a.features[random_cell_line_a]["gene_expression"].shape
        == b.features[random_cell_line_b]["gene_expression"].shape
    )


def test_iterate_features() -> None:
    """Test the iteration over features."""
    df = pd.DataFrame({"GeneA": [1, 2, 3, 2], "GeneB": [4, 5, 6, 2], "GeneC": [7, 8, 9, 2]})
    df.index = ["CellLine1", "CellLine2", "CellLine3", "CellLine1"]
    features = iterate_features(df, "gene_expression")
    assert len(features) == 3
    assert np.all(features["CellLine1"]["gene_expression"] == [1.5, 3, 4.5])


def test_load_drug_ids_from_csv() -> None:
    """Test the loading of drug identifiers from a CSV file."""
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    temp_file = os.path.join(temp.name, "GDSC1_small", "drug_names.csv")
    with open(temp_file, "w") as f:
        f.write("DRUG_NAME\n(5Z)-7-Oxozeaenol\n5-Fluorouracil\nA-443654\nA-770041\n")
    drug_ids_gdsc1 = load_drug_ids_from_csv(temp.name, "GDSC1_small")
    assert len(drug_ids_gdsc1.features) == 4
    assert drug_ids_gdsc1.identifiers[0] == "(5Z)-7-Oxozeaenol"


def test_load_drugs_from_fingerprints() -> None:
    """Test the loading of drugs from fingerprints."""
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    os.mkdir(os.path.join(temp.name, "GDSC1_small", "drug_fingerprints"))
    temp_file = os.path.join(
        temp.name,
        "GDSC1_small",
        "drug_fingerprints",
        "pubchem_id_to_demorgan_128_map.csv",
    )
    with open(temp_file, "w") as f:
        f.write(
            "3827738,5311510,46883536,73707530,16720766\n"
            "1,1,1,1,1\n"
            "1,1,0,0,1\n"
            "0,1,1,0,1\n"
            "1,0,1,1,1\n"
            "1,1,0,1,1\n"
        )
    drug_features_gdsc1 = load_drug_fingerprint_features(temp.name, "GDSC1_small")
    assert len(drug_features_gdsc1.features) == 5
    assert drug_features_gdsc1.features.keys() == {
        "3827738",
        "5311510",
        "46883536",
        "73707530",
        "16720766",
    }
    assert np.all(drug_features_gdsc1.features["3827738"]["fingerprints"] == [1, 1, 0, 1, 1])


@pytest.mark.parametrize(
    "gene_list",
    [
        None,
        "landmark_genes",
        "drug_target_genes_all_drugs",
        "gene_list_paccmann_network_prop",
    ],
)
def test_get_multiomics_feature_dataset(gene_list: Optional[str]) -> None:
    """
    Test the loading of multiomics features.

    :param gene_list: list of genes to keep
    """
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    # gene expression
    temp_file = os.path.join(temp.name, "GDSC1_small", "gene_expression.csv")
    with open(temp_file, "w") as f:
        f.write(
            "cellosaurus_id,CELL_LINE_NAME,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
            "CVCL_1104,CAL-120,7.632023171463389,2.9645851205892404,10.3795526353077,3.61479404843988,"
            "3.38068143582194,7.09344749430946,3.0222634357817597\n"
            "CVCL_1174,DMS 114,7.54867116637172,2.77771614989839,11.807341248845802,4.066886747621,"
            "3.73248465377029,2.8016127581695,6.07851099764176\n"
            "CVCL_1110,CAL-51,8.71233752103624,2.6435077554121,9.88073281995499,3.95622995046262,"
            "3.23662007804984,11.394340478134598,4.22471584953505\n"
            "CVCL_V001,NCI-H2869,7.79714221650204,2.8179230218265,9.88347076381233,4.0637013909818505,"
            "3.55841402145301,8.76055372116888,4.33420904819493\n"
            "CVCL_1045,22Rv1,4.8044868436701,2.84812776692645,10.3319941550002,5.14538669275316,"
            "3.54519297942073,3.9337949618623704,2.8629939819029904\n"
        )

    # methylation
    temp_file = os.path.join(temp.name, "GDSC1_small", "methylation.csv")
    with open(temp_file, "w") as f:
        f.write(
            "cellosaurus_id,CELL_LINE_NAME,chr1:10003165-10003585,chr1:100315420-100316009,"
            "chr1:100435297-100436070,chr1:100503482-100504404,chr1:10057121-10058108,"
            "chr11:107728949-107729586,chr11:107798958-107799980\n"
            "CVCL_1045,22Rv1,0.192212286,0.20381998,0.277913619,0.1909300789999999,0.544058696\n"
            "CVCL_1642,PFSK-1,0.1876026089999999,0.2076517789999999,0.400145531,0.195871473,0.76489757\n"
            "CVCL_1104,CAL-120,0.2101851619999999,0.222116189,0.264730199,0.243298011,0.415484752\n"
            "CVCL_1199,ES3,0.205613728,0.227570131,0.303640813,0.250454389,0.599274902\n"
        )
    # mutations
    temp_file = os.path.join(temp.name, "GDSC1_small", "mutations.csv")
    with open(temp_file, "w") as f:
        f.write(
            "cellosaurus_id,CELL_LINE_NAME,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
            "CVCL_X481,201T,False,False,False,False,False,True,True\n"
            "CVCL_1045,22Rv1,False,True,False,True,False,False,True\n"
            "CVCL_1046,23132/87,False,False,True,True,False,False,False\n"
            "CVCL_1104,CAL-120,False,False,False,False,False,True,False\n"
        )

    # copy number variation
    temp_file = os.path.join(temp.name, "GDSC1_small", "copy_number_variation_gistic.csv")
    with open(temp_file, "w") as f:
        f.write(
            "cellosaurus_id,CELL_LINE_NAME,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
            "CVCL_X481,201T,0.0,0.0,-1.0,0.0,0.0,1.0,-1.0\n"
            "CVCL_1762,TE-12,-1.0,-1.0,0.0,1.0,1.0,0.0,0.0\n"
            "CVCL_1104,CAL-120,0.0,0.0,0.0,-1.0,-1.0,1.0,0.0\n"
            "CVCL_X508,STS-0421,0.0,0.0,1.0,0.0,0.0,-1.0,0.0\n"
            "CVCL_1045,22Rv1,1.0,1.0,-1.0,1.0,1.0,1.0,1.0\n"
        )
    if gene_list is not None:
        _write_gene_list(temp, gene_list)
    omics = ["gene_expression", "methylation", "mutations", "copy_number_variation_gistic"]
    gene_lists = {o: gene_list for o in omics}
    gene_lists["methylation"] = None
    if gene_list == "gene_list_paccmann_network_prop":
        with pytest.raises(ValueError) as valerr:
            dataset = get_multiomics_feature_dataset(
                data_path=temp.name,
                dataset_name="GDSC1_small",
                gene_lists=gene_lists,
                omics=omics,
            )
    else:
        dataset = get_multiomics_feature_dataset(
            data_path=temp.name,
            dataset_name="GDSC1_small",
            gene_lists=gene_lists,
            omics=omics,
        )
        assert len(dataset.features) == 2
        common_cls = dataset.identifiers
        common_cls.sort()
        assert np.all(common_cls == ["22Rv1", "CAL-120"])
        assert dataset.meta_info is not None
        assert len(dataset.meta_info) == 4
    if gene_list is None:
        assert dataset.meta_info is not None
        assert np.all(
            dataset.meta_info["gene_expression"] == ["TSPAN6", "TNMD", "BRCA1", "SCYL3", "HDAC1", "INSIG1", "FOXO3"]
        )
        for key in dataset.meta_info:
            assert len(dataset.meta_info[key]) == 7
    else:
        feature_names: list[str] = []
        if gene_list == "landmark_genes":
            assert dataset.meta_info is not None
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
            assert dataset.meta_info is not None
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


def test_unique() -> None:
    """Test the unique function."""
    array = np.array([1, 9, 3, 2, 1, 4, 5, 6, 7, 8, 9, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    unique_array = unique(array)
    assert np.all(unique_array == np.array([1, 9, 3, 2, 4, 5, 6, 7, 8]))
