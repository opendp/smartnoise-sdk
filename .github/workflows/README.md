# Continuous Integration Tests

GitHub Actions for continuous integration.

You can run CI tests locally before initiating a pull request, using `act`.  First, [install act](http://github.com/nektos/act).  Then, from the root of the repository, run `act pull_request`.  CI actions can be initiated individually by passing the workflow name: `act pull_request .github/workflows/sql/engines/postgres.yml`

Note that `act` requires docker, and can only run actions built on Ubuntu images.  These actions can run on Linux, Windows, and MacOS.  Actions based on non-Ubuntu images can still run in cloud CI, but do not run locally.  To run locally on MacOS, you might need to edit the matrix to use only one python version, since MacOS seems to have trouble when multiple dockers are running pytorch or spark.

