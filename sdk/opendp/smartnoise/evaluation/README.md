Evaluation is one of fundamental components in the development and use of differentially private algorithms. Any privacy algorithm claiming to be differentially private (mechanism, SQL query) can tested against various properties they promise via this evaluation suite - 
* Privacy promise: DP algorithms commit to adhering the DP definition and bound privacy loss
* Accuracy promise: DP algorithms should add the minimal amount of noise needed to actual responses for bounding privacy loss
* Utility promise: The error / confidence bounds for the responses from DP algorithms should be small for the results to have utility
* Bias promise: DP algorithms on repeated runs should have a mean signed deviation close to zero and not have a statistically significant MSD != 0. 

## Evaluation
As part of the evaluation suite, we provide a large set of metrics corresponding to these promises via a single call to evaluate function. 

## Benchmarking
Researchers while building DP algorithms need to benchmark these metrics against a set of input parameters like a range of epsilon values or various simulated dataset sizes. Our benchmarking capabilities on top of stochastic evaluation enable this scenario. This is also helpful for showcasing the properties of any new DP algorithm to end user base. 