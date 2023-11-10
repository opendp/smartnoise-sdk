# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['sneval', 'sneval.metrics', 'sneval.metrics.basic', 'sneval.metrics.compare']

package_data = \
{'': ['*']}

install_requires = \
['numpy>=1.26.1,<2.0.0', 'pyspark>=3.5.0,<4.0.0']

setup_kwargs = {
    'name': 'smartnoise-eval',
    'version': '0.3.0',
    'description': 'Evaluation of differentially private tabular data',
    'long_description': '# SmartNoise Evaluator\n\nThe SmartNoise Evaluator is designed to help assess the privacy and accuracy of differentially private queries. It includes:\n\n* Analyze: Analyze a dataset and provide information about cardinality, data types, independencies, and other information that is useful for creating a privacy pipeline\n* Evaluator: Compares the privatized results to the true results and provides information about the accuracy and bias\n\nThese tools currently require PySpark.\n\n## Analyze\n\nAnalyze provides metrics about a single dataset.\n\n* Percent of all dimension combinations that are unique, k < 5 and k < 10 (Count up to configurable “reporting length”)\n* Report which columns are “most linkable”\n* Marginal histograms up to n-way -- choose default with reasonable size (e.g. 10 per marginal, and up to 20 marginals -- allow override).  Trim and encode labels.\n* Number of rows\n* Number of distinct rows\n* Count, Mean, Variance, Min, Max, Median, Percentiles for each marginal\n* Classification AUC\n* Individual Cardinalities\n* Dimensionality, Sparsity\n* Independencies\n\n\n## Evaluate\n\nEvaluate compares an original data file with one or more comparison files.  It can compare any of the single-file metrics computed in `Analyze` as well as a number of metrics that involve two datasets.  When more than one comparison dataset is provided, we can provide all of the two-way comparisons with the original, and allow the consumer to combine these measures (e.g. average over all datasets)\n\n* How many dimension combinations are suppressed \n* How many dimension combinations are fabricated \n* How many redacted rows (fully redacted vs. partly redacted) \n* Mean absolute error by 1-way, 2-way, etc. up to reporting length\n* Also do for user specified dimension combinations \n* Report by bin size (e.g., < 1000, >= 1000) \n* Mean proportional error by 1-way, 2-way, etc. \n',
    'author': 'SmartNoise Team',
    'author_email': 'smartnoise@opendp.org',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://smartnoise.org',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.13',
}


setup(**setup_kwargs)
