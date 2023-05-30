# SmartNoise Evaluator

The SmartNoise Evaluator is designed to help assess the privacy and accuracy of differentially private queries. It includes:

* Analyze: Analyze a dataset and provide information about cardinality, data types, independencies, and other information that is useful for creating a privacy pipeline
* Runner: A tool that allows a privacy pipeline to be repeatedly run against a dataset, including with different parameters
* Evaluator: Compares the privatized results to the true results and provides information about the accuracy and bias
* Attack: Attempts powerful membership inference attacks against privatized results.  The attack depends on running privatized results against many neighboring datasets, with member population chosen to maximize the attack's power.

These tools are currently available as code samples, but may be released as a Python package as the interface stabilizes.

The Analyze step can run against supported databases, text files or Parquet files, and can use a supplied TableTransformer.  The runner can run against any data sources, but must output CSV, TSV, or Parquet, since the evaluator and attack tools require these formats.

```python
from smartnoise.evaluation import Analyze

analyze = Analyze(
    source = "data.csv", # can be text file or df or parquet, or a spark session or database connection
    table = None, # optional table name - required if source is a database or spark session
    workload = [], # optional list of important queries
    transformer = None, # optional TableTransformer
    timeout = 60, # optional timeout for any analysis step
    max_errors = 50, # optional maximum number of errors to ignore before failing
    output_path = "analysis.json", # optional path to write analysis results
    metadata = None, # optional metadata describing the columns

)

analyze.run()

```

```python
from smartnoise.evaluation import Runner

