# Extending SQL functionality
## Architecture Overview
## Setup
For now setup and e2e setup are the same because the dataset service feature is not mocked. Once we can mock dataset service features for local csvs(inputs for tests) we can move a smaller subset of dependencies here.
## E2e Setup
For the SQL module we need 3 core components: dataset service for reading the data, the ability to run a module, and the burdock library/dependencies.
Steps:
1, create a conda environment: conda create -n oss_dp python
2, git clone https://privacytoolsproject/burdock.git
3, pip install -e sdk/
4, pip install -r tests/requirements.txt 
5, pip install -r service/requirements.txt
6, pytest tests/


## Adding tests
For tests 
