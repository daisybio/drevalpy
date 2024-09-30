from setuptools import setup, find_packages

setup(
    name="drevalpy",
    version="1.0.5",
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
        "flaky",
        "importlib_resources",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "pingouin",
        "plotly",
        "pytorch-lightning",
        "pytest",
        "ray[tune]",
        "requests",
        "scikit-learn",
        "scipy",
    ],
    entry_points= {'console_scripts': [
            'run_suite=drevalpy.run_suite:main', 'create_report=drevalpy.create_report:main'
        ]},
    include_package_data=True,
    package_data={
        "": [
            "models/baselines/hyperparameters.yaml",
            "models/simple_neural_network/hyperparameters.yaml",
            "visualization/style_utils/favicon.png",
            "visualization/style_utils/index_layout.html",
            "visualization/style_utils/LCO.png",
            "visualization/style_utils/LDO.png",
            "visualization/style_utils/LPO.png",
            "visualization/style_utils/nf-core-drugresponseeval_logo_light.png",
            "visualization/style_utils/page_layout.html",
            "assets/dreval.png"
        ]
    },
)
