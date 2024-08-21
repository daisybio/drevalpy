import os
import pytest
import pandas as pd
import numpy as np
import tempfile

from drevalpy.datasets.utils import download_dataset
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.utils import load_cl_ids_from_csv, load_and_reduce_gene_features, iterate_features, load_drug_ids_from_csv, load_drug_features_from_fingerprints


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


def test_load_and_reduce_gene_features():
    temp = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(temp.name, "GDSC1_small"))
    temp_file = os.path.join(temp.name, "GDSC1_small", "gene_expression.csv")
    with open(temp_file, "w") as f:
        f.write("CELL_LINE_NAME,TSPAN6,TNMD,DPM1,SCYL3,C1orf112\n"
                "201T,1,2,3,4,17\n"
                "22RV1,5,6,7,8,18\n"
                "23132/87,9,10,11,12,19\n"
                "42-MG-BA,13,14,15,16,20\n")
    gene_features_gdsc1 = load_and_reduce_gene_features("gene_expression", None, temp.name, "GDSC1_small")
    assert len(gene_features_gdsc1.features) == 4
    assert len(gene_features_gdsc1.meta_info["gene_expression"]) == 5
    assert gene_features_gdsc1.meta_info["gene_expression"][0] == "TSPAN6"


def test_iterate_features():
    df = pd.DataFrame({"GeneA": [1, 2, 3, 2], "GeneB": [4, 5, 6, 2], "GeneC": [7, 8, 9, 2]})
    df.index = ["CellLine1", "CellLine2", "CellLine3", "CellLine1"]
    with pytest.warns(UserWarning):
        features = iterate_features(df, "gene_expression")
    assert len(features) == 3
    assert np.all(features["CellLine1"]["gene_expression"] == [1, 4, 7])



# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
