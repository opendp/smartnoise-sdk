# SmartNoise Evaluator

The SmartNoise Evaluator is designed to help assess the privacy and accuracy of differentially private queries. It includes:

* Analyze: Analyze a dataset and provide information about cardinality, data types, independencies, and other information that is useful for creating a privacy pipeline
* Evaluate: Compares the privatized results to the true results and provides information about the accuracy and bias

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


## Evaluate

Evaluate compares an original data file with one or more comparison files.  It can compare any of the single-file metrics computed in `Analyze` as well as a number of metrics that involve two datasets.  When more than one comparison dataset is provided, we can provide all of the two-way comparisons with the original, and allow the consumer to combine these measures (e.g. average over all datasets)

* How many dimension combinations are suppressed 
* How many dimension combinations are fabricated 
* How many redacted rows (fully redacted vs. partly redacted)
* Mean error in the count across categories by 1-way, 2-way, etc.
* Mean absolute error by 1-way, 2-way, etc. up to reporting length
  * Also do for user specified dimension combinations 
  * Report by bin size (e.g., < 1000, >= 1000) 
* Mean proportional error by 1-way, 2-way, etc. 
