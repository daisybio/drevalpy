"""For the nf-core/drugresponseeval subworkflow preprocess_custom."""

import argparse
from pathlib import Path


def preprocess_raw_viability():
    """CLI for preprocessing raw viability data."""
    from drevalpy.datasets.curvecurator import preprocess

    # define parser
    parser = argparse.ArgumentParser(description="Preprocess CurveCurator viability data.")
    parser.add_argument(
        "--path_data",
        type=str,
        default="./data",
        help="Path to base folder containing datasets, in particular dataset_name/dataset_name_raw.csv, "
        "default: ./data.",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name, e.g., MyCustomDataset.")
    parser.add_argument(
        "--cores", type=int, default=4, help="The number of cores used for CurveCurator fitting, default: 4."
    )
    args = parser.parse_args()
    # get the raw data and preprocess
    input_file = Path(args.path_data).resolve() / args.dataset_name / f"{args.dataset_name}_raw.csv"
    output_dir = input_file.parent
    preprocess(input_file=str(input_file), output_dir=str(output_dir), dataset_name=args.dataset_name, cores=args.cores)


def postprocess_viability():
    """CLI for postprocessing viability data."""
    from drevalpy.datasets.curvecurator import postprocess

    # define parser
    parser = argparse.ArgumentParser(
        description="Postprocess CurveCurator viability data, combines everything in one <dataset_name>.csv file."
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name, e.g., MyCustomDataset.")
    parser.add_argument(
        "--path_data",
        type=str,
        default="./",
        help="Path to output folder of CurveCurator containing the curves.txt file, default: './'.",
    )
    args = parser.parse_args()
    output_folder = Path(args.path_data).resolve() / args.dataset_name
    # postprocess the curves.txt files and saves to the dataset_name.csv file
    postprocess(output_folder=str(output_folder), dataset_name=args.dataset_name)
