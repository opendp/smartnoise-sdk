# CI/CD modernization plan

This plan is scoped so that **one agent can complete one phase in a single session**. Use the checkboxes to track progress as work lands.

## Planning baseline

- [x] Audited the current workflow set in `.github/workflows/`.
- [x] Confirmed the package support window overlap is **Python 3.10-3.14** (`sql`, `synth`, and `eval` now all allow `>=3.10,<3.15`).
- [x] Confirmed the workflows still rely heavily on outdated Actions and bootstrap steps (`actions/checkout@v2/v3`, `actions/setup-python@v2`, `conda-incubator/setup-miniconda@v2`, `hashicorp/setup-terraform@v1`, manual `conda install pip`, and legacy `pip` flows).
- [x] Confirmed recent failure signatures that should shape the first implementation passes:
- [x] `synth.yml` fails because `smartnoise-synth` resolves `smartnoise-sql>=1.0.7` from PyPI instead of the local checkout.
- [x] `spark.yml` fails in current PySpark handling around `pre_aggregated` DataFrame type checks.
- [x] `bigquery.yml` fails when repo secrets / ADC are unavailable and still surfaces deprecated Terraform action behavior.
- [x] Confirmed we want to **keep and modernize all existing workflows**, not retire them.

## Phase 1 - Define the repo-wide CI contract and uv bootstrap

**Goal:** establish a supported Python policy and one modern environment bootstrap pattern before editing individual workflows.

- [x] Set the canonical CI support window to **3.10-3.14** for the packages that intentionally changed metadata in this branch.
- [x] Decide the default single-version job runtime as **3.12**, with an expanded matrix used only where it adds value.
- [x] Establish the Conda replacement pattern with `actions/setup-python` plus `astral-sh/setup-uv` and shared caching.
- [x] Choose one reusable install pattern for local packages, test requirements, and editable installs across `sql`, `synth`, and `eval`.
- [x] Add and document a reusable CI helper surface so every job stops hand-rolling Python setup.

**Likely files:** `.github/workflows/*.yml`, `.github/workflows/README.md`, optional `.github/actions/...`

**Exit criteria:** every later phase can reuse one agreed Python/uv bootstrap pattern without re-deciding versions or install semantics.

**Phase 1 handoff:** the shared bootstrap now lives in `.github/actions/setup-python-uv/action.yml`, and `ci_contract.yml` is the proving workflow for the repo-wide contract. Phase 2 should migrate `sql.yml`, `synth.yml`, `lint_sql.yml`, and `synth_lint.yml` to this helper rather than re-implementing Python setup, caching, or editable install ordering.

## Phase 2 - Modernize the core PR gates for SQL and synth

**Goal:** fix the workflows most relevant to PR #626 and make coupled `sql` + `synth` changes test reliably.

- [x] Update `sql.yml`, `synth.yml`, `lint_sql.yml`, and `synth_lint.yml` to current major Actions versions.
- [x] Convert those workflows from ad hoc `pip` / Conda setup to the Phase 1 `uv` pattern.
- [x] Ensure `synth.yml` installs the local `sql` package from the checkout before testing `synth`, so cross-package PRs do not depend on PyPI state.
- [x] Revisit path filters so `synth` CI reruns when shared dependency surfaces in `sql` change in ways that can affect `synth`.
- [x] Keep the fast PR gates small enough to stay useful, but broad enough that PR #626 gets meaningful coverage in both packages.

**Likely files:** `.github/workflows/sql.yml`, `.github/workflows/synth.yml`, `.github/workflows/lint_sql.yml`, `.github/workflows/synth_lint.yml`

**Exit criteria:** the repo has a modern, uv-based PR gate for SQL and synth that can validate PR #626 without relying on stale package publishing behavior.

**Phase 2 handoff:** `sql.yml`, `synth.yml`, `lint_sql.yml`, and `synth_lint.yml` now use `.github/actions/setup-python-uv` and current checkout majors. The SQL test job now tracks the 3.10-3.14 support-window matrix, while the single-version synth and SQL lint jobs still default to 3.12. `synth.yml` installs local `sql` before `synth`, its pull request paths now include `sql/snsql/**` plus `sql/pyproject.toml`, `synth/pyproject.toml` now declares `disjoint-set` so MST tests install cleanly through package metadata, and `synth/tests/requirements.txt` now pins `scikit-learn<1.8` to stay compatible with the current `diffprivlib` path. Phase 2 also fixed an upper-bound off-by-one in `synth/snsynth/transform/bin.py` and added a regression test so the factory/MST coverage stays green under the modernized workflow. `synth_lint.yml` was modernized on the same helper but remains manual-only because the existing synth tree still has unrelated flake8 debt. Phase 3 should carry the same helper into `postgres.yml`, `mysql.yml`, and `windows.yml` while reevaluating database service/container setup instead of the old OS-level package bootstraps.

## Phase 3 - Refresh local database integration workflows

**Goal:** modernize the workflows that exercise local relational engines without mixing in Spark or cloud auth complexity.

- [x] Update `postgres.yml`, `mysql.yml`, and `windows.yml` to current supported Actions components.
- [x] Move those jobs to the shared Python/uv bootstrap while preserving DB-specific setup.
- [x] Replace brittle package-install and service-start steps with the most current supported runner/service patterns.
- [x] Normalize fixture setup, dataset downloads, and environment variable handling so the three workflows behave consistently.
- [x] Re-check whether any workflow should use service containers instead of manual OS-level package installs.

**Likely files:** `.github/workflows/postgres.yml`, `.github/workflows/mysql.yml`, `.github/workflows/windows.yml`

**Exit criteria:** Postgres, MySQL, and Windows SQL integration jobs are on supported Actions/tooling and behave consistently with the core PR gates.

**Phase 3 handoff:** `postgres.yml`, `mysql.yml`, and `windows.yml` now use current checkout/setup actions plus `.github/actions/setup-python-uv`, with Python 3.12 as the default single-version runtime. `postgres.yml` now provisions `postgres:16` as a service container and loads the PUMS fixtures inside the container instead of installing the server on the runner. `mysql.yml` now provisions `mysql:8.0` as a service container, stages fixture CSVs into `/var/lib/mysql-files`, and creates its CI test user locally so the workflow no longer depends on repo secrets. `windows.yml` now shares the same uv bootstrap and normalized `check_databases.py` / connection-fixture flow while still using hosted LocalDB on `windows-2022`. Phase 4 should carry the shared helper into `spark.yml`, then pin a known-good Java/PySpark combination and fix the `pre_aggregated` DataFrame path.

## Phase 4 - Repair and modernize the Spark workflow

**Goal:** make Spark green again on supported Python and current PySpark behavior.

- [x] Update `spark.yml` to the shared Python/uv bootstrap and current Actions versions.
- [x] Pin or validate a supported Java + PySpark combination for the selected Python version.
- [x] Fix the failing Spark path around `pre_aggregated` DataFrame handling if current PySpark type/module names differ from the code's assumptions.
- [x] Decide whether the Spark job should run on one canonical Python version or a wider matrix once it is stable.
- [x] Keep Spark-specific skips / env vars explicit and documented rather than relying on inherited shell state.

**Likely files:** `.github/workflows/spark.yml`, Spark-sensitive SQL test/code paths if needed

**Exit criteria:** Spark is no longer blocked by outdated environment setup or stale PySpark assumptions.

**Phase 4 handoff:** `spark.yml` now uses `actions/checkout@v6`, `actions/setup-java@v5`, and `.github/actions/setup-python-uv`, with Python 3.12 kept as the canonical single-version runtime for now instead of reintroducing a matrix before the job is stable. The workflow pins Temurin 17 plus `pyspark==3.5.0`, and sets `TEST_SPARK`, `SKIP_PANDAS`, `PYSPARK_DRIVER_PYTHON`, `PYSPARK_PYTHON`, `SPARK_LOCAL_HOSTNAME`, and `SPARK_LOCAL_IP` explicitly at the job level. `sql/snsql/sql/private_reader.py` now accepts both `pyspark.sql.dataframe.DataFrame` and the newer `pyspark.sql.classic.dataframe.DataFrame` module path for `pre_aggregated` inputs, and `sql/tests/private_reader/test_pre_aggregated.py` adds a regression test for the newer module path without requiring a live Spark runtime. Phase 5 should modernize `bigquery.yml`, move it to the current Google auth pattern, and make fork/no-secret behavior skip intentionally instead of failing during Terraform setup.

## Phase 5 - Modernize the BigQuery workflow and secret handling

**Goal:** keep BigQuery coverage while making auth, Terraform, and fork behavior sane.

- [x] Update `bigquery.yml` to current major Actions versions, including Terraform setup.
- [x] Replace the current credential flow with the recommended Google auth pattern for GitHub Actions.
- [x] Make the workflow behave intentionally when secrets are unavailable (for example, skip with a clear reason on forks instead of failing inside Terraform).
- [x] Remove deprecated Terraform action behavior and commit/provider pinning gaps if the workflow still generates new lock state at runtime.
- [x] Keep the full integration path active for trusted repo contexts where the required secrets exist.

**Likely files:** `.github/workflows/bigquery.yml`, `sql/tests/terraform/bigquery/*`, workflow docs for secrets

**Exit criteria:** BigQuery remains part of CI, but fails only for real integration issues rather than missing credentials or deprecated action plumbing.

**Phase 5 handoff:** `bigquery.yml` now uses `actions/checkout@v6`, `hashicorp/setup-terraform@v4`, `.github/actions/setup-python-uv`, and `google-github-actions/auth@v3`, with Python 3.12 as the default single-version runtime. The workflow now checks for `GOOGLE_PROJECT_ID`, `GOOGLE_REGION`, and `GOOGLE_BUCKET_NAME`, then prefers Workload Identity Federation when `GOOGLE_WORKLOAD_IDENTITY_PROVIDER` and `GOOGLE_SERVICE_ACCOUNT` are present, otherwise falling back to `GOOGLE_APPLICATION_CREDENTIALS`; if neither auth path is available, the job exits cleanly with a step-summary skip note instead of failing inside Terraform. `sql/tests/setup/dataloader/factories/bigquery.py` now accepts `GOOGLE_APPLICATION_CREDENTIALS` as either inline JSON or a generated credentials-file path so local `.secrets` and GitHub ADC both work, `sql/tests/terraform/bigquery/main.tf` now pins the Google provider major and relies on a committed lock file, and `.github/workflows/README.md` documents the new secret contract. Phase 6 should modernize `docs.yml` and `samples-test.yml`, remove stale Python 3.8-era assumptions, and decide whether those workflows should remain manual-only once they are green.

## Phase 6 - Refresh docs and sample-notebook workflows

**Goal:** bring the non-core workflows up to current Python/tooling and fix obvious repository drift.

- [x] Update `docs.yml` and `samples-test.yml` to current Actions versions and the shared Python/uv bootstrap.
- [x] Remove Python 3.8-era assumptions and keep these jobs inside the supported 3.10-3.14 window.
- [x] Reconcile stale repository assumptions in the sample workflow, including the current `./sdk` install path.
- [x] Revisit old Sphinx pins and Python-stdlib backports such as `pathlib` so docs install cleanly on supported Python.
- [x] Decide whether these workflows should stay manual-only or become PR / push checks after they are green.

**Likely files:** `.github/workflows/docs.yml`, `.github/workflows/samples-test.yml`, `docs/requirements.txt`, `sql/docs/requirements.txt`, `synth/docs/requirements.txt`, `.github/workflows/README.md`

**Exit criteria:** docs and sample workflows are modernized enough to run intentionally and no longer encode obviously obsolete repo layout or Python assumptions.

**Phase 6 handoff:** `docs.yml` now uses `actions/checkout@v6` plus `.github/actions/setup-python-uv`, builds the repo-local `docs/`, `sql/docs/`, `synth/docs/`, and `eval/docs/` trees on Python 3.12, and has pull-request path filters for docs sources plus package surfaces that feed autodoc/apidoc. The docs dependency stack is now pinned to a Python-3.12-compatible Sphinx 7 / `pydata-sphinx-theme` combination, `docs/requirements.txt` no longer installs the obsolete `pathlib` backport, and the docs Makefiles/configs now switch to non-versioned sidebars for local `make html` builds while leaving multiversion output intact. `samples-test.yml` also moved to the shared Python 3.12 bootstrap, but it stays `workflow_dispatch`-only: instead of cloning the legacy `smartnoise-samples` repo and trying to install the removed `./sdk` monolith, it now smoke-tests a curated set of notebooks from `sql/samples/` and `synth/samples/`, using temporary sanitized copies so notebook self-install cells do not replace the local editable packages. Phase 6 also updated `synth/samples/mst_sample/mst_sample_pums.ipynb` to the current `MSTSynthesizer` API so the curated notebook set runs cleanly. Phase 7 should run the refreshed workflow set together, confirm branch-protection check names, and decide whether any further documentation or dependency-maintenance automation belongs in the repo.

## Phase 7 - Full validation, branch protection alignment, and follow-through

**Goal:** make the refreshed CI usable for approving PR #626 and easier to keep current afterward.

- [ ] Run the full workflow set on a branch that includes the CI refresh and the PR #626 code path.
- [x] Confirm the status check names expected by branch protection still match the workflows after modernization.
- [x] Update `.github/workflows/README.md` with the new uv-based local workflow instructions and secret expectations.
- [x] Add a lightweight maintenance mechanism for GitHub Actions and Python dependencies (for example Dependabot or Renovate, if acceptable for this repo).
- [x] Record any intentionally manual-only workflows and the reason they are not blocking PRs.

**Likely files:** `.github/workflows/README.md`, workflow YAMLs, repository settings follow-up outside git if needed

**Exit criteria:** the repo has a credible path to getting PR #626 through green CI and a lower chance of drifting back into unsupported CI/CD state.

**Phase 7 handoff:** `.github/workflows/README.md` now documents the direct `uv` install order behind `.github/actions/setup-python-uv`, distinguishes `act pull_request` from `act workflow_dispatch`, lists the current workflow/job check names to use for branch protection, records the two intentional manual-only workflows (`synth_lint.yml` for existing unrelated flake8 debt and `samples-test.yml` for the curated notebook smoke set), and clarifies that BigQuery is now the only secret-dependent workflow. `.github/dependabot.yml` adds a lightweight monthly maintenance loop for GitHub Actions plus the Python dependency manifests used by the package and docs workflows. GitHub currently reports no required status-check contexts on `main`, and PR #626 currently shows no check runs, so the remaining follow-up is to push this CI refresh to a branch and run the refreshed workflow set in GitHub Actions before treating Phase 7 as fully validated.

## Notes for implementation agents

- Prefer **uv** for dependency installation and caching unless a phase uncovers a hard blocker.
- Treat **Python 3.10-3.14** as the current SQL, synth, and eval support window after the package metadata updates on this branch.
- Keep phases focused: do not mix BigQuery auth work into core PR gate cleanup, and do not expand Spark debugging into unrelated SQL workflow churn.
- When a phase finishes, update its completed tasks here from `[ ]` to `[x]` in the same PR so the next agent starts from the current state.
