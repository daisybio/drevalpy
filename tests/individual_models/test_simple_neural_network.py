import pytest

from .utils import sample_dataset, call_save_and_load
from drevalpy.models import MODEL_FACTORY
from drevalpy.evaluation import evaluate


@pytest.mark.parametrize("test_mode", ["LPO", "LCO", "LDO"])
@pytest.mark.parametrize(
    "model_name", ["SimpleNeuralNetwork", "MultiOmicsNeuralNetwork"]
)
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
    print(
        f"{test_mode}: Performance of {model}, hpams: {hpam_combi}: PCC = {metrics['Pearson']}"
    )
    if test_mode == "LDO":
        assert metrics["Pearson"] > 0.0
    else:
        assert metrics["Pearson"] > 0.5

    call_save_and_load(model)
