from setuptools import setup, find_packages

setup(
    name='dreval',
    version='1.0.0',
    author=' ',
    description='Drug Response Evaluation of cancer cell line drug response models in a fair setting.',
    long_description='todo',
    long_description_content_type='text/markdown',
    url='https://github.com/biomedbigdata/drp_model_suite',
    packages=find_packages(), 
    classifiers=[
        'Development Status :: Alpha',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.9', 
    install_requires=[
        'pandas',  
        'numpy',     
        'ray[tune]',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'plotly'
    ],
    include_package_data=True,
    package_data={'': ['models/Baselines/hyperparameters.yaml',
                       'models/SimpleNeuralNetwork/hyperparameters.yaml']},
)
