# Sample Noteboks and Code for Stochastic Evaluator

The Differential Privacy Verification notebook in this folder showcases various tests applied on different layers of Differential Privacy implementation. The layers include - 

* Noise adding Mechanisms like Gaussian / Laplace
* Aggregation functions like SUM, COUNT, MEAN, VAR
* Rich SQL Queries with multiple measures grouped by various dimensions

The tests include - 
* Privacy Test - verifies whether a given differential private query response is adhering to the condition of differential privacy trying out across various neighboring pairs
* Accuracy Test - given confidence level like 95%, on repeatedly querying the responses fall within the confidence interval 95% of the times
* Utility Test - ensures that the confidence interval bounds reported are not too wide and (1-confidence level) of the DP responses do fall outside the bounds
* Bias Test - reports the mean signed deviation of noisy responses as a percentage of actual response on repeated querying