[**Please note that we are renaming the toolkit and will be introducing the new name in the coming weeks.**](https://projects.iq.harvard.edu/opendp/blog/building-inclusive-community)

<a href="https://opendifferentialprivacy.github.io"><img src="images/WhiteNoise Logo/SVG/Logo Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>
## System: Tools for Differential Privacy
See also the accompanying [Core repository](https://github.com/opendifferentialprivacy/whitenoise-core) and [Samples repository](https://github.com/opendifferentialprivacy/whitenoise-samples) for this system. </br>


##

The tools repository for this system allow researchers and analysts to: 

* Use SQL dialect to create differentially private results over tabular data stores
* Host a service to compose queries from heterogeneous differential privacy modules (including non-SQL) against shared privacy budget
* Perform black-box stochastic testing against differential privacy modules

This differential privacy system is currently aimed at scenarios where the researcher is trusted by the data owner.  Future releases will focus on hardened scenarios where the researcher or analyst is untrusted.  

New mechanisms and algorithms will be available in coming weeks.


## Data Access

The data access library intercepts SQL queries and processes the queries to return differentially private results.  It is implemented in Python and designed to operate like any ODBC or DBAPI source.  We provide support for PostgreSQL, SQL Server, Spark, Presto, and Pandas. 

Detailed documentation, as well as information about plugging in to other database backends, can be found [here](https://github.com/opendifferentialprivacy/whitenoise-samples/tree/master/docs).

## Service

The reference execution service provides a REST endpoint that can serve requests against shared data sources.  It is designed to allow pluggable composition of many heterogeneous differential privacy modules.  Heterogeneous requests against the same data source will compose privacy budget.  We include SQL dialect, differentially-private graph (core), and a Logistic Regression module from IBM's diffprivlib.

More information, including information about creating and integrating your own privacy modules, can be found [here](https://github.com/opendifferentialprivacy/whitenoise-system/tree/master/service).

## Evaluator

The stochastic evaluator drives black-box privacy algorithms, checking for privacy violations, accuracy, and bias.  It was inspired by Google's stochastic evaluator, and is implemented in Python.  Future releases will support more intelligent search of query input and data input space.

Notebooks illustrating the use of the evaluator can be found [here](https://github.com/opendifferentialprivacy/whitenoise-samples/tree/master/evaluator).

## Installation:
The system's Core library can be installed from PyPi:
> pip install opendp-whitenoise

## Documentation
Documentation for SDK functionality: [here](https://opendifferentialprivacy.github.io/whitenoise-samples/docs/api/system/)

### Experimental
Service API specification: [here](https://github.com/opendifferentialprivacy/whitenoise-system/blob/master/service/openapi/swagger.yml)

## Getting started
```python
import sklearn.datasets
import pandas as pd

from opendp.whitenoise.sql import execute_private_query, PandasReader
from opendp.whitenoise.metadata import CollectionMetadata
from opendp.whitenoise.metadata.collection import Table, Float

sklearn_dataset = sklearn.datasets.load_iris()
sklearn_df = pd.DataFrame(data=sklearn_dataset.data, columns=sklearn_dataset.feature_names)


iris = Table("dbo", "iris", 150, [
    Float("sepal length (cm)", 4, 8),
    Float("sepal width (cm)", 2, 5),
    Float("petal length (cm)", 1, 7),
    Float("petal width (cm)", 0, 3)
])
schema = CollectionMetadata([iris], "csv")

reader = PandasReader(schema, sklearn_df)
rowset = execute_private_query(schema, reader, 0.3, 'SELECT AVG("petal width (cm)") FROM dbo.iris')
df = pd.DataFrame(rowset[1:], columns=rowset[0])
```
## Samples
Samples of DP SQL functionality: [here](https://github.com/opendifferentialprivacy/whitenoise-samples/blob/master/data/README.md)

## Communication

- Please use [GitHub issues](https://github.com/opendifferentialprivacy/whitenoise-system/issues) for bug reports, feature requests, install issues, and ideas.
- [Gitter](https://gitter.im/opendifferentialprivacy/WhiteNoise) is available for general chat and online discussions.
- For other requests, please contact us at [whitenoise@opendp.io](mailto:whitenoise@opendp.io).
  - _Note: We encourage you to use [GitHub issues](https://github.com/opendifferentialprivacy/whitenoise-system/issues), especially for bugs._

## Releases and Contributing

Please let us know if you encounter a bug by [creating an issue](https://github.com/opendifferentialprivacy/whitenoise-system/issues).

We appreciate all contributions. We welcome pull requests with bug-fixes without prior discussion.

If you plan to contribute new features, utility functions or extensions to this system, please first open an issue and discuss the feature with us.
  - Sending a PR without discussion might end up resulting in a rejected PR, because we may be taking the system in a different direction than you might be aware of.
