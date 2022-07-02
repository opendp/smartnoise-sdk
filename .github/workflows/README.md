# Continuous Integration Tests

GitHub Actions for continuous integration.

You can run CI tests locally before initiating a pull request, using `act`.  First, [install act](http://github.com/nektos/act).  Then, from the root of the repository, run `act pull_request`.  CI actions can be initiated individually by passing the workflow name: `act pull_request -W .github/workflows/postgres.yml`

Note that `act` requires docker, and can only run actions built on Ubuntu images.  These actions can run on Linux, Windows, and MacOS.  Actions based on non-Ubuntu images can still run in cloud CI, but do not run locally.  To run locally on MacOS, you might need to edit the matrix to use only one python version, since MacOS seems to have trouble when multiple dockers are running pytorch or spark.

When running the `postgres.yml` CI using `act` on Windows, make sure the `install.sh` in `tests/setup/postgres/PUMS` has not been checked out with Windows CRLF line endings.  This file gets copied into the Ubuntu container to run against the Postgres engine, and will fail if there are CRs.  If this happens, you can load `install.sh` in VS Code, press (CTRL+SHIFT+P) and type "Change End Of Line Sequence" to switch to Unix style newlines.

When running the CI for the `bigquery` engine, you will need to supply GCP credentials in the `.secrets` file.  The credentials will be used to create the ephemeral testing environment in the cloud.  You can see the demo of this file in [`.secrets-demo`](../../sql/tests/setup/bigquery/.secrets-demo).  Duplicate this file, rename it to `.secrets` and supply a value for each of the variables.  In order to generate the necessary GCP credentials, you'll have to:
1. [Create a Service Account](https://cloud.google.com/iam/docs/creating-managing-service-accounts#creating)
2. [Grant the Service Account](https://cloud.google.com/iam/docs/granting-changing-revoking-access#single-role) with `BigQuery Admin` and `Storage Admin` roles
3. [Create a key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating) for the service Account and download as a `.json` file
4. Remove line breaks `/n` from the `.json` file with the permissions and insert it into `.secrets` file

With these, you can trigger the CI pipeline locally with `act pull_request -W .github/workflows/bigquery.yml --secret-file "sql/.secrets"`
