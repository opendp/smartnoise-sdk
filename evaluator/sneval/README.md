## Introduction

Evaluation is one of core components in the development and use of differentially private algorithms. Any privacy algorithm claiming to be differentially private (mechanism, SQL query) can tested against various properties they promise -
* **Privacy**: DP algorithms claim to adhere to the fundamental promise of bounding privacy loss as per the (ε, δ)-DP condition
* **Accuracy**: DP algorithms should add the minimal amount of noise needed to actual responses for bounding privacy loss
* **Utility**: The error / confidence bounds for the responses from DP algorithms should be small for the results to have utility
* **Bias**: DP algorithms on repeated runs should have a mean signed deviation close to zero and not have a statistically significant deviation greater or lower than zero.

## DP Evaluator
As part of the evaluation suite, we compute a set of metrics corresponding to these promises via a single call to `evaluate` function. The interface based design of this suite allows for evaluation of state-of-the-art DP implementations like DP-SQL queries with GROUP BY, JOINs, RANK operators and testing of τ-thresholding.

### Metrics Available

 | Metric  | Promise | Description |
 |---------|--------------|---------------------|
 |  `dp_res` | Privacy | (ε, δ)-DP bounded histogram test on neighboring datasets D1 and D2. Returns True or False         |
 |  `jensen_shannon_divergence` | Privacy | Track JS Divergence applying DP algorithm on neighboring datasets D1 and D2 |
 |  `kl_divergence` | Privacy | Same as above |
 |  `wasserstein_distance` | Privacy | Same as above (statistical measure for distance between probability distributions) |
 |  `mse` | Accuracy | Mean squared error between repeated DP responses vs actual response |
 |  `std` | Accuracy | Standard deviation between repeated DP responses vs actual response |
 |  `msd` | Bias | Mean signed deviation between repeated DP responses vs actual response |
 |  `bias_res` | Bias | 1 sample t-test to check if difference in actual and noisy responses is not statistically significant. Returns True or False |

 There are more metrics planned to be added with availability of error bounds in DP responses like `within_bounds` and `outside_bounds` for testing utility promise.  

### Example Code

This [unit test](https://github.com/opendp/smartnoise-sdk/blob/060ead584360f6e8c16db12d9e7c9eb8e59e687f/tests/sdk/evaluation/test_interface.py#L47) is a good example of how to specify a PrivacyAlgorithm interface for the DP implementation that needs to be tested and pass it through the evaluator to fetch evaluation metrics listed above as one metrics object.

#### Privacy Algorithm Interface Requirements
Privacy Algorithm needs to support `prepare`, `release` and `actual_release` functions.

* `prepare` acts similar to a constructor taking in the privacy algorithm object and evaluation parameters as input and initializing them
* `release` applies the private algorithm on the input dataset repeatedly based on the `repeat_count` in evaluation params. It returns a dictionary of key-value pairs. If it is a SQL query with multiple row response, key is a hash or concatenation of dimension column values. If it is a single row response, then one should specifiy a dummy key `__key__`. Value is a list of DP noisy numerical responses based on the `repeat_count` evaluator paramater specified as input to the `release` function. The evaluator needs to be passed in neighboring datasets `d1` and `d2` which differ by 1 user's record (either by deleting a row or updating it) for it to apply the Privacy test and check if it passes.
* `actual_release` works very similar to the `release` function but instead of a list of noisy responses, it returns the actual response of non-private algorithm corresponding to the DP algorithm being tested.

A sample PrivacyAlgorithm interface is implemented [here](https://github.com/opendp/smartnoise-sdk/blob/master/tests/sdk/evaluation/dp_algorithm.py).

#### Input
```python
from sneval.params._privacy_params import PrivacyParams
from sneval.params._eval_params import EvaluatorParams
from sneval.report._report import Report
from sneval.privacyalgorithm._base import PrivacyAlgorithm
from sneval.evaluator._dp_evaluator import DPEvaluator
from sneval.metrics._metrics import Metrics
from dp_lib import DPSampleLibrary
from dp_algorithm import DPSample
import pandas as pd
import numpy as np
import random
import pytest

lib = DPSampleLibrary()
pa = DPSample()
metrics = Metrics()
pp = PrivacyParams(epsilon=1.0)
ev = EvaluatorParams(repeat_count=500)
# Creating neighboring datasets
d1 = pd.DataFrame(random.sample(range(1, 1000), 100), columns = ['Usage'])
drop_idx = np.random.choice(d1.index, 1, replace=False)
d2 = d1.drop(drop_idx)
# Call evaluate
eval = DPEvaluator()
```

#### Evaluation Call
```python
key_metrics = eval.evaluate(d1, d2, pa, lib.dp_count, pp, ev)
```

#### Print Metrics output
```python
for key, metrics in key_metrics.items():
    print(str(metrics.dp_res))
    print(str(metrics.wasserstein_distance))
    print(str(metrics.jensen_shannon_divergence))
    print(str(metrics.kl_divergence))
    print(str(metrics.mse))
    print(str(metrics.std))
    print(str(metrics.msd))
```

## DP Benchmarking
While building DP algorithms, researchers need to benchmark DP evaluation metrics against a set of input parameters like a range of epsilon values or various dataset sizes. Our benchmarking capabilities via a single `benchmark` call built on top of evaluation suite enable this scenario. This is also helpful for visualizing the properties of any new DP algorithm to end user base for gaining confidence.

### Example Code
This [unit test](https://github.com/opendp/smartnoise-sdk/blob/060ead584360f6e8c16db12d9e7c9eb8e59e687f/tests/sdk/evaluation/test_interface.py#L86) is a good example of how to setup benchmarking of an implementation claiming to be differentially private and interface-able via the PrivacyAlgorithm interface exposed by the researcher or end user. Benchmarking calls the evaluate function underneath iterating through every epsilon value, dataset size and privacy algorithm that needs to be benchmarked.

#### Input
```python
from sneval.params._privacy_params import PrivacyParams
from sneval.params._eval_params import EvaluatorParams
from sneval.params._benchmark_params import BenchmarkParams
from sneval.report._report import Report
from sneval.privacyalgorithm._base import PrivacyAlgorithm
from sneval.evaluator._dp_evaluator import DPEvaluator
from sneval.benchmarking._dp_benchmark import DPBenchmarking
from sneval.metrics._metrics import Metrics
from dp_lib import DPSampleLibrary
from dp_algorithm import DPSample
import pandas as pd
import numpy as np
import random
import pytest

logging.getLogger().setLevel(logging.DEBUG)
epsilon_list = [0.001, 0.5, 1.0, 2.0, 4.0]
lib = DPSampleLibrary()
pa = DPSample()
ev = EvaluatorParams(repeat_count=500)
# Creating neighboring datasets
d1 = pd.DataFrame(random.sample(range(1, 1000), 100), columns = ['Usage'])
drop_idx = np.random.choice(d1.index, 1, replace=False)
d2 = d1.drop(drop_idx)
benchmarking = DPBenchmarking()
# Preparing benchmarking params
pa_algorithms = {pa : [lib.dp_count]}
privacy_params_list = []
for epsilon in epsilon_list:
    pp = PrivacyParams()
    pp.epsilon = epsilon
    privacy_params_list.append(pp)
d1_d2_list = [[d1, d2]]
benchmark_params = BenchmarkParams(pa_algorithms, privacy_params_list, d1_d2_list, ev)
```

#### Benchmarking Call
```python
benchmark_metrics_list = benchmarking.benchmark(benchmark_params)
```

#### Read benchmarking output
```python
for bm in benchmark_metrics_list:
    for key, metrics in bm.key_metrics.items():
        test_logger.debug("Epsilon: " + str(bm.privacy_params.epsilon) + \
            " MSE:" + str(metrics.mse) + \
            " Privacy Test: " + str(metrics.dp_res))
        assert(metrics.dp_res == True)
```

This [Jupyter notebook](https://github.com/opendp/smartnoise-sdk/blob/main/tests/sdk/evaluation/DPBenchmarkingFramework.ipynb) demonstrates how one can use the benchmarking output for various kind of visualizations comparing DP algorithms for the same purpose.

## RL based DP-SQL fuzzing
One of the most commonly used scenarios that are apt for adoption of differential privacy is data aggregation, reporting and data analytics. To enable this, SmartNoise SDK provides a DP-SQL implementation. This enables a large set of aggregation operators available for both end users to leverage and potential adversaries to exploit. In order to test this functionality, manual testing or random testing is not sufficient as the search space of queries possible to write is virtually infinite. We need a smart way to test DP-SQL which is self-optimizing towards the objective of finding bugs. We used contextual bandits and reinforcement learning to achieve this for automatically creating smart queries that have high likelihood to fail the fundamental privacy promise.
