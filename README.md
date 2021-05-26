[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a href="https://smartnoise.org"><img src="images/SmartNoise/SVG/Logo Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>

## SmartNoise System: Tools for Differential Privacy


<br />Please see the accompanying [SmartNoise Documentation](https://docs.opendp.org/en/latest/smartnoise/), [SmartNoise Core repository](https://github.com/opendp/smartnoise-core) and [SmartNoise Samples repository](https://github.com/opendp/smartnoise-samples) for this system. 


##

The tools SmartNoise SDK allows researchers and analysts to:

* Use SQL dialect to create differentially private results over tabular data stores
* Host a service to compose queries from heterogeneous differential privacy modules (including non-SQL) against shared privacy budget
* Perform privacy algorithm stochastic testing against differential privacy modules

This SmartNoise System is currently aimed at scenarios where differential privacy can be used in scenarios where the researcher is trusted by the data owner.  Future releases will focus on hardened scenarios where the researcher or analyst is untrusted.  


## Data Access

The data access library intercepts SQL queries and processes the queries to return differentially private results.  It is implemented in Python and designed to operate like any ODBC or DBAPI source.  We provide support for PostgreSQL, SQL Server, Spark, Presto, and Pandas.

Detailed documentation, as well as information about plugging in to other database backends, can be found [here](https://github.com/opendp/smartnoise-samples/tree/master/docs).

## Service

The reference execution service provides a REST endpoint that can serve requests against shared data sources.  It is designed to allow pluggable composition of many heterogeneous differential privacy modules.  Heterogeneous requests against the same data source will compose privacy budget.  We include SQL dialect, differentially-private graph (core), and a Logistic Regression module from IBM's diffprivlib.

More information, including information about creating and integrating your own privacy modules, can be found [here](https://github.com/opendp/smartnoise-sdk/tree/master/service).

## Evaluator

The stochastic evaluator drives privacy algorithms, checking for privacy violations, accuracy, and bias.  It was inspired by Google's stochastic evaluator, and is implemented in Python.  Future releases will support more intelligent search of query input and data input space.

Notebooks illustrating the use of the evaluator can be found [here](https://github.com/opendp/smartnoise-samples/tree/master/evaluator).

## Installation:
The system's SmartNoise Core library can be installed from PyPi:
> pip install opendp-smartnoise

## Documentation
Documentation for SDK functionality: [here](https://opendp.github.io/smartnoise-samples/docs/api/system/)

### Experimental
Service API specification: [here](https://github.com/opendp/smartnoise-sdk/blob/master/service/openapi/swagger.yml)

## Getting started
### venv setup
```shell
virtualenv -p `which python3` venv
source venv/bin/activate
pip3 install -U scikit-learn scipy matplotlib
pip3 install opendp-smartnoise
```
## conda setup
```shell
conda create -n dev_smartnoise python=3.7
conda activate dev_smartnoise
pip install -U scikit-learn scipy matplotlib
pip install opendp-smartnoise
```
### Script
```python
import sklearn.datasets
import pandas as pd

from opendp.smartnoise.sql import execute_private_query, PandasReader
from opendp.smartnoise.metadata import CollectionMetadata
from opendp.smartnoise.metadata.collection import Table, Float

sklearn_dataset = sklearn.datasets.load_iris()
sklearn_df = pd.DataFrame(data=sklearn_dataset.data, columns=sklearn_dataset.feature_names)


iris = Table("dbo", "iris", [
    Float("sepal length (cm)", 4, 8),
    Float("sepal width (cm)", 2, 5),
    Float("petal length (cm)", 1, 7),
    Float("petal width (cm)", 0, 3)
], 150)
schema = CollectionMetadata([iris], "csv")

reader = PandasReader(sklearn_df, schema)
df = execute_private_query(reader, schema, 0.3, 'SELECT AVG("petal width (cm)") AS petal FROM dbo.iris')
with pd.option_context('display.max_rows', None, 'display.max_columns', 3): print(df)
```
## SmartNoise Samples
Samples of DP SQL functionality: [here](https://github.com/opendp/smartnoise-samples/blob/master/data/README.md)

## Communication

- Please use [GitHub issues](https://github.com/opendp/smartnoise-sdk/issues) for bug reports, feature requests, install issues, and ideas.
- [Gitter](https://gitter.im/opendifferentialprivacy/SmartNoise) is available for general chat and online discussions.
- For other requests, please contact us at [smartnoise@opendp.org](mailto:smartnoise@opendp.org).
  - _Note: We encourage you to use [GitHub issues](https://github.com/opendp/smartnoise-sdk/issues), especially for bugs._

## Releases and Contributing

Please let us know if you encounter a bug by [creating an issue](https://github.com/opendp/smartnoise-sdk/issues).

We appreciate all contributions. We welcome pull requests with bug-fixes without prior discussion.

If you plan to contribute new features, utility functions or extensions to this system, please first open an issue and discuss the feature with us.
  - Sending a PR without discussion might end up resulting in a rejected PR, because we may be taking the system in a different direction than you might be aware of.
