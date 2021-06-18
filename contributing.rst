Contributing to SmartNoise
=============================
Contributions to SmartNoise are welcome from all members of the community. This document is here to simplify the onboarding experience for contributors, contributions to this document are also welcome.

System requirements
=============================
SmartNoise-sdk is python based. The initial setup will require a python
environment, ideally isolated with conda or venv. Below we have a conda based example for setting up the SDK requirements

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

The sdk tests should pass if not, please check github issues for known build failures and report a new issue if not recorded.

Adding new tests
===============================
Testing against new datasets is common since most functionality can only be evaluated with certain datasets. The general practice around
new datasets is to auto download them in conftest.py. If the amount grows too large it would be worth wrapping the idempotent download in fixtures. We will leave that as a possible contribution for now.
Tests for the SDK should be quick. If not, pleaes mark the test with @pytest.mark.slow.

There are existing datasets found in datasets/,
we tend to use the git relative path for access. As you can see below, the metadata tends to be in
a .yaml of the same name as the test dataset, schema.yaml files are needed for most supported DP algorithms.

.. code-block:: python

    import subprocess
    git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
    meta_path = os.path.join(git_root_dir, os.path.join("datasets", "reddit.yaml"))
    csv_path = os.path.join(git_root_dir, os.path.join("datasets", "reddit.csv"))

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
    git checkout branchname # switch back to your branch
    git merge main

If there are no changes that conflict with your branch, the merge will automatically succeed, and you can check it in, push, and move on to the pull request.  If there are merge conflicts, you will need to review and resolve the conflicts first.  Visual Studio Code has nice support for reviewing merge conflicts.

When the patch or feature is ready to submit, run the unit tests to make sure there are no regressions:

.. code-block:: bash

    pytest tests/sdk

Fix any regressions before creating a pull request.  Make sure that GitHub has the latest copy of your local changes:

.. code-block:: bash

    git push

To create the pull request, use your Web browser to navigate to the "pull requests" tab on github.com.  

.. image:: images/doc/Recent_pushes.png

.. image:: images/doc/PR_from_repo.png

Assign the pull request to someone on the development team for code review.  Once the pull request is submitted, some automated integration tests will run to check for regressions.  These tests can take several minutes to complete, and results will be shown in the "Automation" tab.

If there are comments or questions during code review, they will be shown in-line on the PR review page.  Code changes updates to the PR can be added automatically by changing the code in your local branch and runnning `git push` to move commits into the open pull request.  Pushing new commits into the pull request will trigger the integration tests to run again.

When the PR has been approved, an approver will merge it into main.  After the code is merged to main, you can delete the feature branch.

Contributing from a fork:
=========================

If you are submitting a one-time patch or feature, you can submit a pull request from your own fork.  Create and test your patch as above.  When it's time to submit the pull request, navigate your Web browser to the GitHub page for your fork, and go to the "pull requests" tab.  You will have the option to create a new pull request, and GitHub should automatically select base: opendp/smartnoise-sdk/main for the destination, and your fork and branch as the source. 

.. image:: images/doc/PR_from_fork.png