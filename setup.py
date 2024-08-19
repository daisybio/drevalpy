from setuptools import setup, find_packages

setup(
    name="drevalpy",
    version="1.0.4",
    author=" ",
    description="Drug Response Evaluation of cancer cell line drug response models in a fair setting.",
    long_description="<h1>Drug Response Evaluation of cancer cell line drug response models in a fair setting</h1>"
    "<p>drevalpy is a Python package that provides a framework for evaluating cancer cell line drug "
    "response models in a fair setting. The package includes the functionality to load common drug response "
    "datasets and train pre-implemented models with hyperparamter tuning. It also contains robustness "
    "and randomization tests and functions to evaluate and visualize the results. The package is the basis "
    "for an associated Nextflow pipeline. </p>",
    long_description_content_type="text/markdown",
    url="https://github.com/daisybio/drevalpy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=[
        "importlib_resources",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "pingouin",
        "plotly",
        "pytorch-lightning",
        "ray[tune]",
        "scikit-learn",
        "scipy",
    ],
    include_package_data=True,
    package_data={
        "": [
            "models/Baselines/hyperparameters.yaml",
            "models/SimpleNeuralNetwork/hyperparameters.yaml",
            "visualization/style_utils/favicon.png",
            "visualization/style_utils/index_layout.html",
            "visualization/style_utils/LCO.png",
            "visualization/style_utils/LDO.png",
            "visualization/style_utils/LPO.png",
            "visualization/style_utils/nf-core-drugresponseeval_logo_light.png",
            "visualization/style_utils/page_layout.html",
        ]
    },
)
