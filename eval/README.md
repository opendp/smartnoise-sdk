# SmartNoise Evaluator

The SmartNoise Evaluator is designed to help assess the privacy and accuracy of differentially private queries. It includes:

* Analyze: Analyze a dataset and provide information about cardinality, data types, independencies, and other information that is useful for creating a privacy pipeline
* Evaluator: Compares the privatized results to the true results and provides information about the accuracy and bias

These tools currently require PySpark.

## Analyze

Analyze provides metrics about a single dataset.

* Percent of all dimension combinations that are unique, k < 5 and k < 10 (Count up to configurable “reporting length”)
* Report which columns are “most linkable”
* Marginal histograms up to n-way -- choose default with reasonable size (e.g. 10 per marginal, and up to 20 marginals -- allow override).  Trim and encode labels.
* Number of rows
* Number of distinct rows
* Count, Mean, Variance, Min, Max, Median, Percentiles for each marginal
* Classification AUC
* Individual Cardinalities
* Dimensionality, Sparsity
* Independencies
* t-sne


```python
from sneval import Analyze

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

## Evaluate

Evaluate compares an original data file with one or more comparison files.  It can compare any of the single-file metrics computed in `Analyze` as well as a number of metrics that involve two datasets.  When more than one comparison dataset is provided, we can provide all of the two-way comparisons with the original, and allow the consumer to combine these measures (e.g. average over all datasets)

* How many dimension combinations are suppressed 
* How many dimension combinations are fabricated 
* How many redacted rows (fully redacted vs. partly redacted) 
* Mean absolute error by 1-way, 2-way, etc. up to reporting length
* Also do for user specified dimension combinations 
* Report by bin size (e.g., < 1000, >= 1000) 
* Mean proportional error by 1-way, 2-way, etc. 


## Run

```python
from smartnoise.evaluation import Runner

