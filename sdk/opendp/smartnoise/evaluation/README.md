## Introduction

Evaluation is one of core components in the development and use of differentially private algorithms. Any privacy algorithm claiming to be differentially private (mechanism, SQL query) can tested against various properties they promise - 
* **Privacy**: DP algorithms claim to adhere to the fundamental promise of bounding privacy loss as per the (ε, δ)-DP condition
* **Accuracy**: DP algorithms should add the minimal amount of noise needed to actual responses for bounding privacy loss
* **Utility**: The error / confidence bounds for the responses from DP algorithms should be small for the results to have utility
* **Bias**: DP algorithms on repeated runs should have a mean signed deviation close to zero and not have a statistically significant deviation greater or lower than zero. 

## DP Evaluation
As part of the evaluation suite, we provide a large set of metrics corresponding to these promises via a single call to evaluate function. 

## DP Benchmarking
While building DP algorithms, researchers need to benchmark DP evaluation metrics against a set of input parameters like a range of epsilon values or various dataset sizes. Our benchmarking capabilities built on top of evaluation suite enable this scenario. This is also helpful for visualizing the properties of any new DP algorithm to end user base for gaining confidence. 

## RL based DP-SQL fuzzing
One of the most commonly used scenarios that are apt for adoption of differential privacy is data aggregation, reporting and analytics. To enable this, SmartNoise SDK provides a DP-SQL implementation. This enables a large set of aggregation operators available for both end users to leverage and potential adversaries to exploit. In order to test this functionality, manual testing or random testing is not sufficient as the search space of queries possible to write is virtually infinite. We need a smart way to test DP-SQL which is self-optimizing towards the objective of finding bugs. We used contextual bandits and reinforcement learning to achieve this for automatically creating smart queries that have high likelihood to fail the fundamental privacy promise. 

