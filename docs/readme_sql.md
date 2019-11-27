# Extending SQL functionality
## Architecture Overview
## Setup
For now setup and e2e setup are the same because the dataset service feature is not mocked. Once we can mock dataset service features for local csvs(inputs for tests) we can move a smaller subset of dependencies here.
## E2e Setup
For the SQL module we need 3 core components: dataset service for reading the data, the ability to run a module, and the burdock library/dependencies.
Steps:
create a conda environment: conda create -n oss_dp python
conda activate oss_dp
git clone https://github.com/privacytoolsproject/burdock.git
pip install -e sdk/
pip install -r tests/requirements.txt
pip install -r service/requirements.txt
pytest tests/


## Adding tests
For tests 
