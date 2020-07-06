# Differentially Private Synthetic Data Gym (DPSDGym)
Differentially Private Synthetic Data Gym (DPSDGym) provides infrastructure to evaluate differentially private synthetic data generators on tabular datasets using best methods from recent literature (see citations). As of now, this includes:
1. propensity Mean Squared Error (pMSE) [1](https://arxiv.org/pdf/2004.07740.pdf)
2. Wasserstein Randomization [1](https://arxiv.org/pdf/2004.07740.pdf)
3. Synthetic Ranking Agreement (SRA) [2](https://arxiv.org/pdf/1806.11345.pdf) 
4. Aggregate AUROC/Machine learning accuracy across different epsilons for:
    * AdaBoost
    * Bagging Classifier
    * Logistic Regression
    * Multilayer Perceptron
    * Decision Tree
    * Naive Bayes (Gaussian, etc.)
    * Random Forest
    * Extra Trees
    [3](https://openreview.net/pdf?id=S1zk9iRqF7) 
## Datasets
DPSDGym contains a data loader function that retrieves datasets from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets.php). Currently, DPSDGym has been tested with:
* [Car Evaluation Data Set](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
* [Mushroom Data Set](https://archive.ics.uci.edu/ml/datasets/Mushroom)
* [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)
* [Nursery Data Set](https://archive.ics.uci.edu/ml/datasets/nursery)

These datasets and their specifications are included already in the datasets.json file. They should work outright.

### Adding a new dataset
#### Add new JSON specification in `datasets.json`
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

#### Add new configuration in `conf.py`
If you wish to use the added dataset in evaluation, you must add it to the list of `KNOWN_DATASETS`. 

You must also add a new configuration for every synthesizer intended to be evaluated with a new dataset. The configuration is simply the hyperparameters to be used with the synthesizer for that dataset. Inside conf.py, add a new entry under the synthesizer you intend on using in the `SYNTH_SETTINGS` dictionary. Specify the synthesizer's hyperparameters to be used in conjunction with the new dataset.
## Requirements
[Python 3.5+](https://www.python.org/downloads/)
## Usage
Once the configuration/datasets are set up, the entire evaluation pipeline, with all metrics, can be run through the eval script
```
python eval.py all
```
Flags can be used to specify which metrics to run in the evaluation. You could equivalently run
```
python eval.py wasserstein ml_eval sra
```
(Note that `sra` can only be run if `ml_eval` is also run.)

The default epsilon increments for the evaluation (different privacy settings):
```python
epsilons=[0.01, 0.1, 1.0, 9.0, 45.0, 95.0]
```

The output of the evaluation pipeline is an `artifact.json` file, that includes all the results from metrics. With this file, you can run the `visualization.py` to produce a graph.
## Other Resources
"""
    {
    "mushroom":{
        "data":{},
        "target":"edible",
        "name":"mushroom",
        "mwem":{},
        "target_synth":{},
        "AdaBoostClassifier":{
            "mwem":{
                "TRTR":{},
                "TSTR":{},
                "TSTS":{}
            },
            "target_synth":{
                "TRTR":{},
                "TSTR":{},
                "TSTS":{}
            }
        },
        "BaggingClassifier":{},
        "LogisticRegression":{},
        "MLPClassifier":{},
        "DecisionTreeClassifier":{},
        "GaussianNB":{},
        "BernoulliNB":{},
        "MultinomialNB":{},
        "RandomForestClassifier":{},
        "ExtraTreesClassifier":{},
        "trtr_sra":{
            "mwem":[],
            "target_synth":[]
        },
        "tsts_sra":{},
        "tstr_avg":{},
        "dumb_predictor":{}
    }
    }
    """