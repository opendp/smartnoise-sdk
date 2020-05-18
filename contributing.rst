Contributing to WhiteNoise
=============================
Contributions to WhiteNoise are welcome from all members of the community. This document is here to simplify the onboarding experience for contributors, contributions to this document are also welcome.

System requirements
=============================
WhiteNoise-system is python based. The initial setup will require a python
environment, ideally isolated with conda or venv. Below we have a conda based example for setting up the SDK and Service requirements

.. code-block:: bash

    create a conda environment: conda create -n whitenoise python
    conda activate whitenoise
    git clone https://github.com/opendifferentialprivacy/whitenoise-system.git
    cd whitenoise-system
    conda install -c anaconda sqlite
    python -m pip install -e sdk/
    python -m pip install -r tests/requirements.txt

Verifying your SDK installation is running:

.. code-block:: bash

    pytest tests/sdk

Additional requirements are needed for contributing to the WhiteNoise Service:

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
4. Regenerate the swagger, and verify that the sdk/opendp/whitenoise/client/restclient contains your new path definition and sdk/opendp/whitenoise/client/restclient/models contains your new schema definition. (the names should be reflective of specifications from the swagger.yml file)

Putting the two together:

Now we must link together the service functionality you wrote with the api call.

1. Navigate to sdk/opendp/whitenoise/client/__init__.py.
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
