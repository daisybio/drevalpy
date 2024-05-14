import argparse
from ..models import MODELFACTORY

"""def train_and_predict(
    model_name: DRPModel,
    hpams: Dict[str, List],
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    response_transformation: Optional[TransformerMixin] = None,
    cl_features: Optional[FeatureDataset] = None,
    drug_features: Optional[FeatureDataset] = None,"""
def get_parser():
     parser = argparse.ArgumentParser(
        description="Run the drug response prediction model test suite."
    )
    parser.add_argument(
        "--run_id", type=str, default="my_run", help="identifier to save the results"
    )
    parser.add_argument(
        "--models", nargs="+", help="model to evalaute or list of models to compare"


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()




def main(args):
    model = MODELFACTORY[model_name](target="IC50")
    model.build_model(hyperparameters=hpams)

    if cl_features is None:
        cl_features = model.load_cell_line_features(path=hpams["feature_path"])
    if drug_features is None:
        drug_features = model.load_drug_features(path=hpams["feature_path"])
    # making sure there are no missing features:
    train_dataset.reduce_to(
        cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
    )

    prediction_dataset.reduce_to(
        cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
    )

    inputs = model.get_feature_matrices(
                    cell_line_ids=train_dataset.cell_line_ids,
                    drug_ids=train_dataset.drug_ids,
                    cell_line_input=cl_features,
                    drug_input=drug_features)
    prediction_inputs = model.get_feature_matrices(
                    cell_line_ids=prediction_dataset.cell_line_ids,
                    drug_ids=prediction_dataset.drug_ids,
                    cell_line_input=cl_features,
                    drug_input=drug_features)
    if early_stopping_dataset is not None:
        early_stopping_dataset.reduce_to(
            cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
        )
        early_stopping_inputs = model.get_feature_matrices(
            cell_line_ids=early_stopping_dataset.cell_line_ids,
            drug_ids=early_stopping_dataset.drug_ids,
            cell_line_input=cl_features,
            drug_input=drug_features,
        )
        for key in early_stopping_inputs:
            inputs[key + "_earlystopping"] = early_stopping_inputs[key]

    if response_transformation:
        response_transformation.fit(train_dataset.response.reshape(-1, 1))
        train_dataset.response = response_transformation.transform(train_dataset.response.reshape(-1, 1)).squeeze()
        early_stopping_dataset.response = response_transformation.transform(early_stopping_dataset.response.reshape(-1, 1)).squeeze()
        prediction_dataset.response = response_transformation.transform(prediction_dataset.response.reshape(-1, 1)).squeeze()

    if model.early_stopping:
        model.train(
            output=train_dataset,
            output_earlystopping=early_stopping_dataset,
            **inputs
        )
    else: 
        model.train(
            output=train_dataset,
            **inputs
        )

    prediction_dataset.predictions = model.predict(**prediction_inputs)

    if response_transformation:
        prediction_dataset.response = response_transformation.inverse_transform(prediction_dataset.response)
        
    return prediction_dataset
