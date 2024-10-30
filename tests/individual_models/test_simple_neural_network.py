import pytest

from drevalpy.evaluation import evaluate
from drevalpy.models import MODEL_FACTORY

from .conftest import sample_dataset
from .utils import call_save_and_load


@pytest.mark.parametrize("test_mode", ["LPO"])
@pytest.mark.parametrize("model_name", ["SRMF", "SimpleNeuralNetwork", "MultiOmicsNeuralNetwork"])
def test_simple_neural_network(sample_dataset, model_name, test_mode):
    drug_response, cell_line_input, drug_input = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=5,
        mode=test_mode,
    )
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_es_dataset = split["validation_es"]
    es_dataset = split["early_stopping"]

    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers

    len_train_before = len(train_dataset)
    len_pred_before = len(val_es_dataset)
    len_es_before = len(es_dataset)
    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    print(f"Reduced training dataset from {len_train_before} to {len(train_dataset)}")
    print(f"Reduced val_es dataset from {len_pred_before} to {len(val_es_dataset)}")
    print(f"Reduced es dataset from {len_es_before} to {len(es_dataset)}")

    model = MODEL_FACTORY[model_name]()
    hpams = model.get_hyperparameter_set()
    hpam_combi = hpams[0]
    model.build_model(hyperparameters=hpam_combi)
    model.train(
        output=train_dataset,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
        output_earlystopping=es_dataset,
    )

    val_es_dataset.predictions = model.predict(
        drug_ids=val_es_dataset.drug_ids,
        cell_line_ids=val_es_dataset.cell_line_ids,
        drug_input=drug_input,
        cell_line_input=cell_line_input,
    )

    metrics = evaluate(val_es_dataset, metric=["Pearson"])
    print(f"{test_mode}: Performance of {model}, hpams: {hpam_combi}: PCC = {metrics['Pearson']}")
    if test_mode == "LDO":
        assert metrics["Pearson"] > 0.0
    else:
        assert metrics["Pearson"] > 0.5

    call_save_and_load(model)
