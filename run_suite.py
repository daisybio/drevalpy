from models_code import SimpleNeuralNetwork
from suite.data_wrapper import DrugResponseDataset
from suite.experiment import drug_response_experiment
import pandas as pd
neural_net_baseline = SimpleNeuralNetwork("smpl", target="IC50")

models = [neural_net_baseline]

response_data = pd.read_csv("data/GDSC/response_GDSC2.csv")
output = response_data["LN_IC50"].values
cell_line_ids = response_data["CELL_LINE_NAME"].values
drug_ids = response_data["DRUG_NAME"].values
response_data = DrugResponseDataset(
    response=output, cell_line_ids=cell_line_ids, drug_ids=drug_ids
)
result = drug_response_experiment(models, response_data, multiprocessing=True, randomization_test_views={"randomize_gene_expression": ["gene_expression"]})
print(result)


