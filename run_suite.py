# Main file for running the whole suite of tests

import os
import argparse

from suite.data_wrapper import DrugResponseDataset, DrugFeatureDataset


def get_parser():
    parser = argparse.ArgumentParser(description='Run the drug response prediction model test suite')
    parser.add_argument('--test_mode', type=str, default='LPO', help='Which tests to run (LPO=Leave-random-Pairs-Out, '
                                                                     'LCO=Leave-Cell-line-Out, LDO=Leave-Drug-Out)')
    parser.add_argument('--path_dr', type=str, default='data/drug_response/', help='Path to the drug response dataset '
                                                                                   '/ directory containing all '
                                                                                   'relevant files')
    parser.add_argument('--name_dr', type=str, default='CCLE', help='Name of the drug response '
                                                                    'dataset')
    parser.add_argument('--target', type=str, default='IC50', help='Target variable to predict (AUC, IC50, EC50, '
                                                                   'classification into sensitive/resistant)')
    parser.add_argument('--path_df', type=str, default='data/drug_features/', help='Path to the drug feature dataset '
                                                                                   '/ directory containing all '
                                                                                   'relevant files')
    parser.add_argument('--name_df', type=str, default='CCLE', help='Name of the drug feature '
                                                                    'dataset')
    parser.add_argument('--path_cf', type=str, default='data/cell_line_features/', help='Path to the cell line '
                                                                                        'feature dataset')
    parser.add_argument('--name_cf', type=str, default='CCLE', help='Name of the cell line feature '
                                                                    'dataset')
    parser.add_argument('--path_out', type=str, default='results/', help='Path to the output directory')
    parser.add_argument('--curve_curator', action=argparse.BooleanOptionalAction, default=False, help='Whether to run '
                                                                                                      'CurveCurator '
                                                                                                      'to sort out '
                                                                                                      'non-reactive '
                                                                                                      'curves')
    parser.add_argument('--custom_models', action=argparse.BooleanOptionalAction, default=False, help='Whether to '
                                                                                                      'train a custom'
                                                                                                      ' model '
                                                                                                      'provided by '
                                                                                                      'the user via '
                                                                                                      'the wrapper')

    return parser


def compare_models(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset):
    """
    Compare the models in the list models on the given datasets.
    """
    for model in models:
        # Train the model
        # Test the model
        # Save the results
        pass


def robustness_tests(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset):
    """
    Run the randomization tests.
    """
    train_with_different_seed(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset)
    train_with_permuted_input(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset)


def train_with_different_seed(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset):
    """
    Train the models with different random seeds.
    """
    pass


def train_with_permuted_input(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset):
    """
    Train the models with input in permuted order.
    """
    pass


def randomization_tests(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset):
    """
    Run the randomization tests.
    """
    train_with_shuffled_input(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset)
    train_with_zero_inputs(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset)


def train_with_shuffled_input(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset):
    """
    Train the models with shuffled input.
    """
    pass


def train_with_zero_inputs(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset):
    """
    Train the models with zero vectors as inputs.
    """
    pass


def visualize_results(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset):
    """
    Visualize the results.
    """
    pass


if __name__ == '__main__':
    os.chdir('suite')
    args = get_parser().parse_args()
    print(args)

    # Get which models should be run
    if args.target == 'classification':
        models = ['Baselines', 'DeepCDR', 'MOLI', 'Super.FELT']
        print(f'Running classification tests on the models {models}')
    else:
        if args.target == 'AUC':
            models = ['Baselines', 'DrugCell']
        elif args.target == 'IC50':
            models = ['Baselines', 'BiGDRP', 'DeepCDR', 'PaccMan', 'SRMF']
        elif args.target == 'EC50':
            models = ['Baselines']
        else:
            raise ValueError('Target variable not recognized')
        print(f'Running regression tests on the models {models} with target variable {args.target}')

    # Make DRP dataset from path and split it according to test mode
    drp_dataset = DrugResponseDataset(path=args.path_dr, name=args.name_dr, target_type=args.target)
    drp_dataset.split_dataset(mode=args.test_mode)

    # Make drug feature dataset from path
    drug_feature_dataset = DrugFeatureDataset(path=args.path_df, name=args.name_df)
    # Make cell line feature dataset from path
    cell_line_feature_dataset = DrugFeatureDataset(path=args.path_cf, name=args.name_cf)

    if args.curve_curator:
        # Run CurveCurator
        pass

    if args.custom_models:
        # Run custom models
        pass

    # Compare the models
    compare_models(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset)

    # Run the robustness tests
    robustness_tests(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset)

    # Run the randomization tests
    randomization_tests(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset)

    # Run the visualization
    visualize_results(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset)