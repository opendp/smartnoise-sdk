# Continuous Integration Tests

GitHub Actions for continuous integration.

## Repo-wide CI contract

- Supported CI Python window: **3.10-3.14**
- Default single-version runtime for jobs that do not benefit from a matrix: **3.12**
- Shared bootstrap surface: **`.github/actions/setup-python-uv`**
- Preferred install pattern:
  1. Install local repo packages with `editable-projects`, in dependency order.
  2. Install third-party test dependencies with `requirements-files`.
  3. Install one-off tools such as linters with `package-specs`.

The shared action replaces Conda-first bootstrapping with `actions/setup-python` plus `astral-sh/setup-uv`, and enables one shared cache contract across workflows. New or modernized workflows should use it rather than hand-rolling Python setup.

The current core PR gates (`sql.yml`, `synth.yml`, and `lint_sql.yml`) already use this helper, `synth_lint.yml` uses the same bootstrap for manual lint runs, the refreshed database integration workflows (`postgres.yml`, `mysql.yml`, `windows.yml`, and `bigquery.yml`) now follow the same Python 3.12 default/runtime contract, and `spark.yml` now does the same before layering on `actions/setup-java` plus `pyspark==3.5.0`. `synth.yml` installs local `sql` before local `synth` so cross-package pull requests exercise the checkout instead of resolving `smartnoise-sql` from PyPI.

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

`ci_contract.yml` is the proving workflow for this contract. It keeps 3.12 as the default runtime while using an explicit 3.10-3.14 matrix only where validating the support window adds value.

## Reproducing the shared bootstrap locally

The composite action is deliberately simple: select the job's Python version, install local packages in dependency order, install any requirements files, then install one-off tools. To mirror a workflow locally without `act`, create a virtual environment with the same Python version and run the same `uv pip` install order.

For example, the synth test workflow maps to:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --editable sql --editable synth
uv pip install -r synth/tests/requirements.txt
pytest synth/tests
```

The action itself uses `uv pip install --system` because GitHub-hosted runners install directly into the job interpreter. When reproducing locally inside a virtual environment, omit `--system` and keep the same package ordering.

## Running workflows locally with act

You can run CI tests locally before initiating a pull request by using `act`. First, [install act](http://github.com/nektos/act). Then, from the repository root, run `act pull_request`. Individual pull-request workflows can be targeted with `act pull_request -W .github/workflows/postgres.yml`, while manual-only workflows use `act workflow_dispatch -W .github/workflows/samples-test.yml`.

`act` requires Docker, and can only run actions built on Ubuntu images. These actions can run on Linux, Windows, and macOS in GitHub-hosted CI, but non-Ubuntu jobs will not run locally with `act`. On macOS, you may need to temporarily reduce large Python matrices when local Docker resources are tight.

The Ubuntu database workflows now start their own service containers:

1. `postgres.yml` uses `postgres:16` and loads the PUMS fixtures inside the service container.
2. `mysql.yml` uses `mysql:8.0`, stages fixture CSVs into `/var/lib/mysql-files`, and creates a local `runner` database user inside the service container.

Those workflows no longer depend on repository secrets for local database passwords. `windows.yml` still relies on the hosted `windows-2022` image's LocalDB tooling, so it remains GitHub-hosted only.

The Spark workflow now uses Temurin 17 with `pyspark==3.5.0` on Python 3.12, and sets `TEST_SPARK`, `SKIP_PANDAS`, `PYSPARK_DRIVER_PYTHON`, `PYSPARK_PYTHON`, `SPARK_LOCAL_HOSTNAME`, and `SPARK_LOCAL_IP` explicitly in the job environment so the Spark-only path does not depend on inherited shell state.

## Current workflow and check names

The branch-protection API for `main` currently reports no required status-check contexts. If required checks are configured or re-enabled later, use the exact job names below rather than the YAML file names.

| Workflow file | Actions workflow name | Job check name(s) | Trigger | Notes |
| --- | --- | --- | --- | --- |
| `ci_contract.yml` | `CI bootstrap contract` | `Default runtime install smoke (sql)`; `Default runtime install smoke (synth)`; `Default runtime install smoke (eval)`; `Support window smoke (Python 3.10)`; `Support window smoke (Python 3.11)`; `Support window smoke (Python 3.12)`; `Support window smoke (Python 3.13)`; `Support window smoke (Python 3.14)` | `pull_request`, `workflow_dispatch` | Bootstrap proving workflow for the shared Python/uv contract |
| `sql.yml` | `SQL on Pandas` | `SQL on Pandas (Python 3.10)`; `SQL on Pandas (Python 3.11)`; `SQL on Pandas (Python 3.12)`; `SQL on Pandas (Python 3.13)`; `SQL on Pandas (Python 3.14)` | `pull_request`, `workflow_dispatch` | Core SQL PR gate |
| `lint_sql.yml` | `SQL code linter` | `SQL code linter` | `pull_request`, `workflow_dispatch` | Fast SQL lint gate |
| `synth.yml` | `Synthesizers Unit Tests` | `Synthesizers Unit Tests` | `pull_request`, `workflow_dispatch` | Installs local `sql` before local `synth` |
| `postgres.yml` | `PostgreSQL and SQLite Integration Tests` | `PostgreSQL and SQLite Integration Tests` | `pull_request`, `workflow_dispatch` | Uses a `postgres:16` service container |
| `mysql.yml` | `MySQL Integration Tests` | `MySQL Integration Tests` | `pull_request`, `workflow_dispatch` | Uses a `mysql:8.0` service container |
| `windows.yml` | `SQL on Pandas and LocalDB on Windows` | `SQL on Pandas and LocalDB on Windows` | `pull_request`, `workflow_dispatch` | Hosted Windows only |
| `spark.yml` | `SQL on Spark` | `SQL on Spark` | `pull_request`, `workflow_dispatch` | Canonical Spark runtime is Python 3.12 + Temurin 17 |
| `bigquery.yml` | `GCP BigQuery Integration Tests` | `GCP BigQuery Integration Tests` | `pull_request`, `workflow_dispatch` | Skips cleanly when required GCP secrets are unavailable |
| `docs.yml` | `Test Documentation Build` | `Build docs` | `pull_request`, `workflow_dispatch` | Runs on docs and autodoc-relevant package changes |
| `synth_lint.yml` | `Lint Synthesizers` | `Lint Synthesizers` | `workflow_dispatch` only | Manual-only until the remaining unrelated synth flake8 debt is cleaned up |
| `samples-test.yml` | `Run Sample Notebooks` | `Sample notebooks` | `workflow_dispatch` only | Manual-only curated notebook smoke set |

## Docs and sample workflows

`docs.yml` now builds the local docs trees from this checkout (`docs/`, `sql/docs/`, `synth/docs/`, and `eval/docs/`) on Python 3.12 with the shared `setup-python-uv` helper, rather than cloning the external `opendp-documentation` repository. It runs automatically on pull requests that touch docs sources, package code that feeds autodoc/apidoc, or the shared CI bootstrap.

`samples-test.yml` stays `workflow_dispatch`-only. The legacy external `smartnoise-samples` repository still targets the old `opendp.smartnoise.*` monolith, so the workflow now smoke-tests a curated set of notebooks that live in this repository instead:

1. `sql/samples/SQL Queries.ipynb`
2. `synth/samples/aggregate_seeded_sample/aggregate_seeded_short_sample.ipynb`
3. `synth/samples/mst_sample/mst_sample_pums.ipynb`
4. `synth/samples/mwem_sample/Visualizing MWEM.ipynb`

The workflow installs local `sql` and `synth`, strips notebook self-install cells in a temporary copy before execution, and keeps the heavier Spark / deep-learning / long-running notebooks out of the default smoke set.

## Manual-only workflows

1. `synth_lint.yml` remains manual-only because the existing `synth/snsynth` tree still has unrelated flake8 debt; keeping it on demand preserves a useful modernization target without making unrelated PRs fail.
2. `samples-test.yml` remains manual-only because the curated notebook smoke set is intentionally slower and less hermetic than the core package test gates, but still useful as an on-demand integration check.

## BigQuery workflow secrets

`bigquery.yml` now authenticates through `google-github-actions/auth@v3` and only runs the BigQuery integration path when all required secrets are present. On forks or other no-secret contexts, the job exits cleanly with a step-summary note instead of failing inside Terraform setup.

For GitHub-hosted CI, configure these repository secrets:

1. `GOOGLE_PROJECT_ID`
2. `GOOGLE_REGION`
3. `GOOGLE_BUCKET_NAME`
4. Either `GOOGLE_WORKLOAD_IDENTITY_PROVIDER` plus `GOOGLE_SERVICE_ACCOUNT` (**preferred**), or `GOOGLE_APPLICATION_CREDENTIALS` with the service account key JSON

The workflow prefers Workload Identity Federation when the provider and service account secrets are present, and otherwise falls back to the legacy service-account-key secret. In both cases, the auth action creates an ADC credentials file for Terraform and the BigQuery test harness.

No other refreshed workflow currently requires repository secrets. The Postgres and MySQL jobs provision their own service-container credentials inside the runner, and the SQL, synth, Spark, docs, and notebook jobs rely only on repository contents.

## BigQuery local secrets

When running the BigQuery workflow locally with `act`, you will still need to supply GCP credentials in a `.secrets` file. The credentials are used to create the ephemeral testing environment in the cloud. You can see the template in [`.secrets-demo`](../../sql/tests/setup/bigquery/.secrets-demo). Duplicate that file, rename it to `.secrets`, and supply values for each variable.

To generate the necessary GCP credentials:

1. [Create a Service Account](https://cloud.google.com/iam/docs/creating-managing-service-accounts#creating)
2. [Grant the Service Account](https://cloud.google.com/iam/docs/granting-changing-revoking-access#single-role) the `BigQuery Admin` and `Storage Admin` roles
3. [Create a key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating) for the service account and download it as a `.json` file
4. Remove line breaks from the `.json` contents and place the single-line value into `.secrets`

With those values in place, you can trigger the CI pipeline locally with `act pull_request -W .github/workflows/bigquery.yml --secret-file "sql/.secrets"`

## Dependency maintenance

`.github/dependabot.yml` now provides a lightweight maintenance loop for this CI surface:

1. Monthly GitHub Actions updates for `.github/workflows/**` and the shared composite action.
2. Monthly Python dependency updates for `sql/`, `synth/`, `eval/`, and each docs dependency directory.
