# Differentially Private Synthetic Data Gym (DPSDGym)
## TLDR
Pip install requirements
```
pip install -r requirements.txt
```
Run evaluation suite on a dataset with the command
```
python main.py [dataset]
ex: (python main.py car)
```
## More info
Differentially Private Synthetic Data Gym (DPSDGym) provides infrastructure to evaluate differentially private synthetic data generators on tabular datasets using best methods from recent literature (see citations). As of now, this includes:
1. propensity Mean Squared Error (pMSE) [(1)](https://arxiv.org/pdf/2004.07740.pdf)
2. Wasserstein Randomization [(1)](https://arxiv.org/pdf/2004.07740.pdf)
3. Synthetic Ranking Agreement (SRA) [(2)](https://arxiv.org/pdf/1806.11345.pdf) 
4. Aggregate AUROC/Machine learning accuracy across different epsilons. [(3)](https://openreview.net/pdf?id=S1zk9iRqF7) The current models supporting aggregation are as follows. Note that any sklearn model can be added simply, using the KNOWN_MODELS list in conf.py.
    * AdaBoost
    * Bagging Classifier
    * Logistic Regression
    * Multilayer Perceptron
    * Random Forest

    
## Datasets
DPSDGym contains a data loader function that retrieves datasets from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets.php). Currently, DPSDGym has been tested with:
* [Car Evaluation Data Set](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
* [Mushroom Data Set](https://archive.ics.uci.edu/ml/datasets/Mushroom)
* [Adult Data Set](http://archive.ics.uci.edu/ml/datasets/Adult)
* [Online Shoppers Data Set](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset#)
* [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#)

These datasets and their specifications are included already in the datasets.json file. They should work outright.

### Adding a new dataset
#### 1. Add new JSON specification in `datasets.json`
When adding a new UCI dataset specification to the datasets.json file, use the following format:
```json
"dataset": {
"name": "dataset_name",
"url": "https://archive.ics.uci.edu/ml/machine-learning-databases/direct/link/to/data.data",
"categorical_columns": "col1,col2,col4",
"columns":"col1,col2,col3,col4",
"target": "col4",
"header": "f",
"sep": ",",
"imbalanced": "f"
}
```
The "columns" are assumed to be in order and include all of the tabular columns, but "categorical_columns" does not need to be ordered. Note that the url links directly to a csv/tsv file hosted by UCI. Also, you must specify whether the dataset has a header (t/f) and what the separator is (will default to ',' but sometimes UCI datasets have weird separators). If you would like for an unbalanced dataset to be rebalanced using imblearn's SMOTE implementation, specify imbalanced as "t".

UCI datasets are not standardized, although many look similar. If you add a dataset and find there are errors in loading or parsing, you may have to modify load_data.py to accomodate the conversion of the unique dataset you are attempting to add into a pandas DataFrame object. Datasets that are formatted similarly to mushroom, adult, etc... should work outright.

#### 2. Add new configuration in `conf.py`
If order to use the added dataset in evaluation, you must add it to the list of `KNOWN_DATASETS`. 

You must also add a new configuration for every synthesizer intended to be evaluated with a new dataset. The configuration is simply the hyperparameters to be used with the synthesizer for that dataset. Inside conf.py, add a new entry under the synthesizer you intend on using in the `SYNTH_SETTINGS` dictionary. Specify the synthesizer's hyperparameters to be used in conjunction with the new dataset.
## Requirements
Make sure you have the following installed in a conda env, before running the evaluation.

[python=3.5+](https://www.python.org/downloads/)
[pyemd](https://github.com/wmayner/pyemd)
[imblearn](https://pypi.org/project/imbalanced-learn/)
[scipy](https://www.scipy.org/scipylib/download.html)
[mlflow==1.4.0](https://pypi.org/project/mlflow/)
[diffprivlib](https://github.com/IBM/differential-privacy-library)
[pandas==0.25.3](https://pypi.org/project/pandas/)
## Usage
Once the configuration/datasets are set up, the entire evaluation pipeline, with all metrics, can be run through the eval script
```
python main.py [dataset]
```
<!-- Flags can be used to specify which metrics to run in the evaluation. You could equivalently run
```
python main.py wasserstein pmse ml_eval sra
```
(Note that `sra` can only be run if `ml_eval` is also run.) -->

The default epsilon increments for the evaluation (different privacy settings):
```python
epsilons=[0.01, 0.1, 0.5, 1.0, 3.0, 6.0, 9.0]
```
Epsilon here is usually graphed on a log scale, hence the big jumps in privacy. Adding more fine grained epsilons, especially in the earlier values, is probably a good idea.

The output of the evaluation pipeline is recorded using mlflow. After a run, you can view the mlflow ui to begin your analysis.
```
mlflow ui
```
Due to the unique nature of evaluating differential privacy across privacy budgets, it can sometimes be difficult to analyze the special case metrics using the mlflow ui. We've added an accompanying notebook which helps with visualizing the metrics across privacy values.
```
jupyter notebook plot-v2.ipynb
```

## Other Resources
DPSDGym draws inspiration from [SDGYM](https://github.com/sdv-dev/SDGym) - it is also designed to play nicely with SDGYM, and so synthesizers that work for DPSDGym should also work with SDGym.