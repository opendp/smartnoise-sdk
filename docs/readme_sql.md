# Extending SQL functionality

## Architecture Overview
The entrypoint for sql functionality is found at:

sdk/burdock/query/sql/private/query.py

The general architecture starts off with a SQL query, dataset name, and budget received from the user as inputs. The sql module code can be found in service/modules/sql-module.
SQL parsing includes reading the user SQL into an AST, morphing the AST for DP related preprocessing. The preprocessed query is compiled back into a string and sent to the reader. We have a local CSV reader and support a few odbc based readers. After the result of the preprocessed query is received, a final post processing layer applies the mechanisms to privatize the results.

Test datasets are found at service/datasets/, queries currently must name tables with db name and table name, for simplicity most schemas(under datasets but .yaml instead of .csv) use dataset_name.dataset_name.
## Setup
For now setup and e2e setup are the same because the dataset service feature does not have a mock. Once we can mock dataset service features for local csvs(inputs for tests) we can move a smaller subset of dependencies here.
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
