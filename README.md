# WhiteNoise System: Tools for Differential Privacy

<a href="https://www.linkedin.com/pulse/microsoft-harvards-institute-quantitative-social-science-john-kahan/"><img src="images/WhiteNoise Logo/SVG/Full_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>
## WhiteNoise Samples: Differential Privacy Examples, Notebooks and Documentation
See also the accompanying [WhiteNoise-core](https://github.com/opendifferentialprivacy/whitenoise-core) and [WhiteNoise-Samples](https://github.com/opendifferentialprivacy/whitenoise-samples) repositories for this system.

The WhiteNoise tools allow researchers and analysts to: 

* Use SQL dialect to ceate differentially private results over tabular data stores
* Host a service to compose queries from heterogeneous differential privacy modules (including non-SQL) against shared privacy buget
* Perform black-box stochastic testing against differential privacy modules

The WhiteNoise system is currently aimed at scenarios where the researcher is trusted by the data owner.  Future releases will focus on hardened scenarios where the researcher or analyst is untrusted.  

New mechanisms and algorithms will be available in coming weeks.


## Data Access

The data access library intercepts SQL queries and processes the queries to return differentially private results.  It is implemented in Python and designed to operate like any ODBC or DBAPI source.  We provide support for PostgreSQL, SQL Server, Spark, Presto, and Pandas. Detailed documentation, as well as information about plugging in to other database backends, can be found here.

## Service

The reference execution service provides a REST endpoint that can serve requests against shared data sources.  It is designed to allow pluggable composition of many heterogeneous differential privacy modules.  Heterogeneous requests against the same data source will compose privacy budget.  We include SQL dialect, differentially-private graph (core), and a Logistic Regression module from IBM's diffprivlib.  More information, including information about creating and integrating your own privacy modules, can be found here.

## Evaluator

The stochastic evaluator drives black-box privacy algorithms, checking for privacy violations, accuracy, and bias.  It was inspired by Google's stochastic evaluator, and is implemented in Python.  Future releases will support more intelligent search of query input and data input space.

## Installation:
The burdock library can be installed from PyPi:
> pip install opendp-whitenoise

## Documentation
Documentation for SDK functionality: <here>
Service API documentation: <here>

## Getting started

## Samples
Samples of DP SQL functionality: <here>
Samples of interacting with the burdock service: <here>
Working with SQL Engines: <exit>
Working with Spark DataFrames: <exit>
  
## Experimental
### Services getting started
Running opendp-whitenoise code through a service layer: <here>
  
### Service samples
TODO
  

  

  
  
  
