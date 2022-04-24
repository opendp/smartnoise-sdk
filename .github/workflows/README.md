# Continuous Integration Tests

GitHub Actions for continuous integration.

You can run CI tests locally before initiating a pull request, using `act`.  First, [install act](http://github.com/nektos/act).  Then, from the root of the repository, run `act pull_request`.  CI actions can be initiated individually by passing the workflow name: `act pull_request -W .github/workflows/postgres.yml`

Note that `act` requires docker, and can only run actions built on Ubuntu images.  These actions can run on Linux, Windows, and MacOS.  Actions based on non-Ubuntu images can still run in cloud CI, but do not run locally.  To run locally on MacOS, you might need to edit the matrix to use only one python version, since MacOS seems to have trouble when multiple dockers are running pytorch or spark.

When running the `postgres.yml` CI using `act` on Windows, make sure the `install.sh` in `tests/setup/postgres/PUMS` has not been checked out with Windows CRLF line endings.  This file gets copied into the Ubuntu container to run against the Postgres engine, and will fail if there are CRs.  If this happens, you can load `install.sh` in VS Code, press (CTRL+SHIFT+P) and type "Change End Of Line Sequence" to switch to Unix style newlines.