## Introduction

Evaluation is one of core components in the development and use of differentially private algorithms. Any privacy algorithm claiming to be differentially private (mechanism, SQL query) can tested against various properties they promise - 
* **Privacy**: DP algorithms claim to adhere to the fundamental promise of bounding privacy loss as per the (ε, δ)-DP condition
* **Accuracy**: DP algorithms should add the minimal amount of noise needed to actual responses for bounding privacy loss
* **Utility**: The error / confidence bounds for the responses from DP algorithms should be small for the results to have utility
* **Bias**: DP algorithms on repeated runs should have a mean signed deviation close to zero and not have a statistically significant deviation greater or lower than zero. 

## DP Evaluation
As part of the evaluation suite, we compute a set of metrics corresponding to these promises via a single call to `evaluate` function. The interface based design of this suite allows for evaluation of state-of-the-art DP implementations like DP-SQL queries with GROUP BY, JOINs, RANK operators and testing of τ-thresholding. 

 | Metric  | Promise Type | Description |
 |---------|--------------|---------------------|
 |  `dp_res` | Privacy | (ε, δ)-DP bounded histogram test on neighboring datasets D1 and D2. Returns True or False         |
 |  `jensen_shannon_divergence` | Privacy | Track JS Divergence applying DP algorithm on neighboring datasets D1 and D2 |
 |  `kl_divergence` | Privacy | Same as above |
 |  `wasserstein_distance` | Privacy | Same as above (statistical measure for distance between probability distributions) |
 |  `mse` | Accuracy | Mean squared error between repeated DP responses vs actual response |
 |  `std` | Accuracy | Standard deviation between repeated DP responses vs actual response |
 |  `msd` | Bias | Mean signed Deviation between repeated DP responses vs actual response |
 |  `bias_res` | Bias | 1 sample t-test to check if difference in actual and noisy responses is not statistically significant. Returns True or False |

 There are more metrics planned to be added with availability of error bounds in DP responses like `within_bounds` and `outside_bounds` for testing utility promise.  

## DP Benchmarking
While building DP algorithms, researchers need to benchmark DP evaluation metrics against a set of input parameters like a range of epsilon values or various dataset sizes. Our benchmarking capabilities via a single `benchmark` call built on top of evaluation suite enable this scenario. This is also helpful for visualizing the properties of any new DP algorithm to end user base for gaining confidence. 

## RL based DP-SQL fuzzing
One of the most commonly used scenarios that are apt for adoption of differential privacy is data aggregation, reporting and data analytics. To enable this, SmartNoise SDK provides a DP-SQL implementation. This enables a large set of aggregation operators available for both end users to leverage and potential adversaries to exploit. In order to test this functionality, manual testing or random testing is not sufficient as the search space of queries possible to write is virtually infinite. We need a smart way to test DP-SQL which is self-optimizing towards the objective of finding bugs. We used contextual bandits and reinforcement learning to achieve this for automatically creating smart queries that have high likelihood to fail the fundamental privacy promise. 

