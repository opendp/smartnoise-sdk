# Differentially Private Synthetic Data Gym (DPSDGym)
Differentially Private Synthetic Data Gym (DPSDGym) provides infrastructure to evaluate differentially private synthetic data generators on tabular datasets using best methods from recent literature (see citations). As of now, this includes:
1. [1]() propensity Mean Squared Error (pMSE) 
2. [1]() Wasserstein Randomization
3. [2]() Synthetic Ranking Agreement (SRA) 
4. [3]() Aggregate AUROC/Machine learning accuracy across different epsilons for:
    * AdaBoost
    * Bagging Classifier
    * Logistic Regression
    * Multilayer Perceptron
    * Decision Tree
    * Naive Bayes (Gaussian, etc.)
    * Random Forest
    * Extra Trees
## Datasets
DPSDGym contains a data loader function that retrieves datasets from the [UCI ML Repository] (https://archive.ics.uci.edu/ml/datasets.php). Currently, DPSDGym has been tested with:
* [Car Evaluation Data Set] (https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
* [Mushroom Data Set] (https://archive.ics.uci.edu/ml/datasets/Mushroom)
* [Wine Quality Data Set] (https://archive.ics.uci.edu/ml/datasets/wine+quality)
* [Nursery Data Set] (https://archive.ics.uci.edu/ml/datasets/nursery)

These datasets and their specifications are included already in the datasets.json file. They should work outright.

When adding a new UCI dataset specification to the datasets.json file, use the following format:
```json
"dataset": {
"name": "dataset_name",
"url": "https://archive.ics.uci.edu/ml/machine-learning-databases/direct/link/to/data.data",
"categorical_columns": "col1,col2,col4",
"columns":"col1,col2,col3,col4",
"target": "col4",
"header": "f",
"sep": ","
}
```
The "columns" are assumed to be in order and include all of the tabular columns, but "categorical_columns" does not need to be ordered. Note that the url links directly to a csv/tsv file hosted by UCI. Also, you must specify whether the dataset has a header (t/f) and what the separator is (will default to ',' but sometimes UCI datasets have weird separators).

UCI datasets are not standardized, although many look similar. If you add a dataset and find there are errors in loading or parsing, you may have to modify load_data.py to accomodate the conversion of the unique dataset you are attempting to add into a pandas DataFrame object.
## Requirements
[Python 3.5+] (https://www.python.org/downloads/)
## Usage

## Other Resources