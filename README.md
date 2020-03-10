# WhiteNoise: A library and service for creating Differentially Private Releases from data <may be worth breaking the below paragraph into bullet points>
Core to the burdock library is our DP SQL functionality, where SQL queries<make link> are executed with pre and post processing that makes the released Rowset Differentially Private.Burdock also provides a client service architecture for running prepackaged Modules<link to modules> on a Dataset to make a Differentially Private Release. The architecture allows for service interactions with the SQL functionality, but also extends support for external libraries including yarrow<link> a DP execution engine that supports validation of DP gaurantees and diffprivlib<link>  which is referenced <here> for a DP LogisticRegression Module that releases a DP Model.
  
While architecting burdock we realized the need for a tracking store, to store the Release, Parameters, and Metrics from the DP  Modules. Given the success of Mlflow, we have integrated with it's tracking store and provide a reference implementation of the Execution Service component that utilizes mlflow's project run paradigm.

## Installation:
The burdock library can be installed from PyPi:
> pip install -i https://test.pypi.org/simple/ burdock

## Documentation
Documentation for SDK fucntionality: <here>
Service API documentation: <here>

## Getting started

## Samples
Samples of DP SQL functionality: <here>
Samples of interacting with the burdock service: <here>
Working with SQL Engines: <exit>
Working with Spark DataFrames: <exit>
  
## Experimental
### Services getting started
Running burdock code through a service layer: <here>
  
### Service samples
TODO
  

  

  
  
  
