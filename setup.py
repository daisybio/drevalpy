from setuptools import setup, find_packages

setup(
    name='drp_model_suite',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Evaluation of cancer cell line drug response models in a fair setting.',
    long_description='todo',
    long_description_content_type='text/markdown',
    url='https://github.com/biomedbigdata/drp_model_suite',
    packages=find_packages(), 
    classifiers=[
        'Development Status :: Alpha',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.9',  # Specify the minimum Python version required
    install_requires=[
        'pandas',  
        'numpy',     
        'ray[tune]'
        'scikit-learn'
    ],
)
