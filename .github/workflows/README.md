# Continuous Integration Tests

GitHub Actions for continuous integration.

## Repo-wide CI contract

- Supported CI Python window: **3.9-3.12**
- Default single-version runtime for jobs that do not benefit from a matrix: **3.12**
- Shared bootstrap surface: **`.github/actions/setup-python-uv`**
- Preferred install pattern:
  1. Install local repo packages with `editable-projects`, in dependency order.
  2. Install third-party test dependencies with `requirements-files`.
  3. Install one-off tools such as linters with `package-specs`.

The shared action replaces Conda-first bootstrapping with `actions/setup-python` plus `astral-sh/setup-uv`, and enables one shared cache contract across workflows. New or modernized workflows should use it rather than hand-rolling Python setup.

### Example

```yaml
- uses: actions/checkout@v6
- uses: ./.github/actions/setup-python-uv
  with:
    python-version: "3.12"
    editable-projects: |
      sql
      synth
    requirements-files: |
      synth/tests/requirements.txt
    package-specs: |
      flake8
```

`ci_contract.yml` is the proving workflow for this contract. It keeps 3.12 as the default runtime while using an explicit 3.9-3.12 matrix only where validating the support window adds value.

## Running workflows locally with act

You can run CI tests locally before initiating a pull request by using `act`. First, [install act](http://github.com/nektos/act). Then, from the repository root, run `act pull_request`. CI actions can be initiated individually by passing the workflow name: `act pull_request -W .github/workflows/postgres.yml`

`act` requires Docker, and can only run actions built on Ubuntu images. These actions can run on Linux, Windows, and macOS in GitHub-hosted CI, but non-Ubuntu jobs will not run locally with `act`. On macOS, you may need to temporarily reduce large Python matrices when local Docker resources are tight.

When running the `postgres.yml` CI using `act` on Windows, make sure `tests/setup/postgres/PUMS/install.sh` has not been checked out with CRLF line endings. That file is copied into the Ubuntu container and will fail if carriage returns are present. If needed, switch the file back to Unix newlines in your editor before running the workflow.

## BigQuery local secrets

When running the CI for the `bigquery` engine, you will need to supply GCP credentials in a `.secrets` file. The credentials are used to create the ephemeral testing environment in the cloud. You can see the template in [`.secrets-demo`](../../sql/tests/setup/bigquery/.secrets-demo). Duplicate that file, rename it to `.secrets`, and supply values for each variable.

To generate the necessary GCP credentials:

1. [Create a Service Account](https://cloud.google.com/iam/docs/creating-managing-service-accounts#creating)
2. [Grant the Service Account](https://cloud.google.com/iam/docs/granting-changing-revoking-access#single-role) the `BigQuery Admin` and `Storage Admin` roles
3. [Create a key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating) for the service account and download it as a `.json` file
4. Remove line breaks from the `.json` contents and place the single-line value into `.secrets`

With those values in place, you can trigger the CI pipeline locally with `act pull_request -W .github/workflows/bigquery.yml --secret-file "sql/.secrets"`
