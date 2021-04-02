Contributing to SmartNoise
=============================
Contributions to SmartNoise are welcome from all members of the community. This document is here to simplify the onboarding experience for contributors, contributions to this document are also welcome.

System requirements
=============================
SmartNoise-sdk is python based. The initial setup will require a python
environment, ideally isolated with conda or venv. Below we have a conda based example for setting up the SDK and Service requirements

.. code-block:: bash

    create a conda environment: conda create -n smartnoise python
    conda activate smartnoise
    git clone https://github.com/opendp/smartnoise-sdk.git
    cd smartnoise-sdk
    conda install -c anaconda sqlite
    python -m pip install -e sdk/
    python -m pip install -r tests/requirements.txt

To run the unit tests, we will need to copy the wheel files from smartnoise.core into your cloned repository.  This is because we installed the cloned repository in --editable mode, so all files for the smartnoise namespace (including smartnoise.core) will be searched from our cloned repository.  First, find the wheel location where smartnoise.core has been installed:

.. code-block:: bash

    pip show opendp-smartnoise-core

This will show a path, for example Location: /Users/youraccount/miniconda3/lib/python3.7/site-packages

Use this path to copy the files from opendp/smartnoise/core:

.. code-block:: bash

    mkdir sdk/opendp/smartnoise/core
    cp -R /Users/youraccount/miniconda3/lib/python3.7/site-packages/opendp/smartnoise/core sdk/opendp/smartnoise/core

Verifying your SDK installation is running:

.. code-block:: bash

    pytest tests/sdk

Additional requirements are needed for contributing to the SmartNoise Service:

.. code-block:: bash

    python -m pip install -r service/requirements.txt

The sdk tests should pass if not, please check github issues for known build failures and report a new issue if not recorded.

Adding new tests
===============================
Testing against new datasets is common since most functionality can only be evaluated with certain datasets. The general practice around
new datasets is to auto download them in conftest.py. If the amount grows too large it would be worth wrapping the idempotent download in fixtures. We will leave that as a possible contribution for now.
Tests for the SDK should be quick. If not, pleaes mark the test with @pytest.mark.slow.

There are existing datasets used for the service found in service/datasets/,
we tend to use the git relative path for access. As you can see below, the metadata tends to be in
a .yaml of the same name as the test dataset, schema.yaml files are needed for most supported DP algorithms.

.. code-block:: python

    import subprocess
    git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
    meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "reddit.yaml"))
    csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "reddit.csv"))

Contributing a Patch or Feature
===============================

If you will be making regular contributions, or need to participate in code reviews of other people's contributions, your GitHub ID should be added to the "Contributors" group.  Regular contributors submit patches or features via a feature branch:

.. code-block:: bash
    git checkout main
    git pull
    git checkout -b branchname
    git push -u origin branchname

Branches should be scoped tightly to work that can be delivered and reviewed in a manageable amount of time.  Feature branches should align with only one pull request, and should be deleted after the PR is approved and merged.  Larger or more open-ended work items should be broken into more scoped pull requests, with one feature branch per pull request.  Naming should be moderately descriptive (e.g. `bugfix_double_spend`) but can be short.

From your new feature branch, make all of your changes.  You can check in changes and use `git push` to periodically synchronize local changes with the feature branch in GitHub.

If other patches or feature branches have been merged to main while you are working, your branch may be out of sync with main.  This is usually not a risk with small patches, but is more likely as development takes longer.

You will need to make sure your branch includes latest changes to main before submitting the pull request.  To do this, you commit any uncommited changes, switch to main and pull, then switch back to your branch and merge.

.. code-block:: bash
    git commit -m "saving changes before merge"
    git push # optional
    git checkout main
    git pull
    git branch branchname # switch back to your branch
    git merge

If there are no changes that conflict with your branch, the merge will automatically succeed, and you can check it in, push, and move on to the pull request.  If there are merge conflicts, you will need to review and resolve the conflicts first.  Visual Studio Code has nice support for reviewing merge conflicts.

When the patch or feature is ready to submit, run the unit tests to make sure there are no regressions:

.. code-block:: bash
    pytest tests/sdk

Fix any regressions before creating a pull request.  Make sure that GitHub has the latest copy of your local changes:

.. code-block:: bash
    git push

To create the pull request, use your Web browser to navigate to the "pull requests" tab on github.com.  Assign the pull request to someone on the development team for code review.

Once the pull request is submitted, some automated integration tests will run to check for regressions.  These tests can take several minutes to complete, and results will be shown in the "Automation" tab.

If there are comments or questions during code review, they will be shown in-line on the PR review page.  Code changes updates to the PR can be added automatically by changing the code in your local branch and runnning `git push` to move commits into the open pull request.  Pushing new commits into the pull request will trigger the integration tests to run again.

Once the PR has been approved, an approver will merge it into main.  After the code is merged to main, you can delete the feature branch.

Contributing from a fork:
=========================

If you are submitting a one-time patch or feature, you can submit a pull request from your own fork.  Create and test your patch as above.  When it's time to submit the pull request, navigate your Web browser to the GitHub page for your fork, and go to the "pull requests" tab.  You will have the option to create a new pull request, and GitHub should automatically select base: opendp/smartnoise-sdk/main for the destination, and your fork and branch as the source. 


Service setup and test validation:
============================
To setup the local flask service run the below command within the previously made conda environment:

.. code-block:: bash

    python service/application.py


In a different shell, within the same conda environment, run the test suite:
.. code-block:: bash

    pytest tests/service -m "not dataverse_token"


Modules can be run directly, without going through the execution service for easier debugging:
.. code-block:: bash

    python service/modules/sql-module/run_query.py "example" .3 "SELECT COUNT(A) from example.example"

Datasets in service/datasets/ can be accessed through SQL queries with table name "file_name.file_name", for example "example.csv" -> "example.example".

Enough with examples :)

Adding a new Service API:
=============================
In order to add dataset.py service calls (to be supported by both the client and server), complete the following steps. One can also modify existing calls by following portions of the below steps.

Adding functionality to the service:

1. Under service/dataset.py, add a function with a new name (one that matches the new api path), and takes in a dictionary.
2. Write in your functionality, operating on that dictionary as if the server has received it from the client. For example: register(dataset): dataset['dataset_name']...
3. Verify that your functionality is correct by running dataset.py with a mock dictionary, and by writing tests under tests/service. Once the additional service functionality appears operational,
move on to adding a client call with the next steps.

Swagger+autorest steps:

1. Follow the steps in the swagger readme (under service/openapi/readme.md) and ensure that you can regenerate the restclient from the swagger.yml file using the autorest npm package. Regenerating it once ensures that any future errors are your fault : )
2. Add a new path to the service/openapi/swagger.yml file. You can use one of the existing paths as a template, but make sure to modify each field and specify the api functionality carefully. Refer to online documentation for examples (https://swagger.io/docs/specification/describing-request-body/ is a good place to start)
3. As you define your new path, make sure to add schema definition that fits your specific use case under "definitions:" in the swagger.yml file. For example, /register takes in a very specific schema, which is defined in "DatasetPutDocument".
4. Regenerate the swagger, and verify that the sdk/opendp/smartnoise/client/restclient contains your new path definition and sdk/opendp/smartnoise/client/restclient/models contains your new schema definition. (the names should be reflective of specifications from the swagger.yml file)

Putting the two together:

Now we must link together the service functionality you wrote with the api call.

1. Navigate to sdk/opendp/smartnoise/client/__init__.py.
2. Inside __init__.py, you'll see classes for the various Clients that the service supports. If you are adding a dataset.py function, you will add a new definition under the DatasetClient class.
3. Note (Ignore if adding to existing client): if you are adding an entirely new client, you will need to make a new client class, and add a "get"-er for that class, to be called in the service module you expose.
4. Use the existing client methods as a template, and perform an additional processing to the dictionary received from the module. For example, in the case of /register, the dictionary passed in by the user is unpacked to fit the DatasetPutDocument schema specified in the swagger.
5. Make sure you call the exposed client method (which is autogenerated by autorest). For example, in the case of register, the method is called datasetregister.

Testing

To verify a working end-to-end:
1. Refer to the Service setup and test validation section.
2. In the other shell, verify that everything is working by running an existing module. Inside your own module, you should then be able to call your new client side function with a call like so:
response = get_dataset_client().your_new_function_name(your_new_function_parameters)
Make sure to swap in a new "get"-er if you've written one, in place of get_dataset_client().
