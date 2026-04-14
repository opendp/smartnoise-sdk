# CI/CD modernization plan

This plan is scoped so that **one agent can complete one phase in a single session**. Use the checkboxes to track progress as work lands.

## Planning baseline

- [x] Audited the current workflow set in `.github/workflows/`.
- [x] Confirmed the package support window overlap is **Python 3.9-3.12** (`sql` allows `<3.14`, while `synth` and `eval` allow `<3.13`).
- [x] Confirmed the workflows still rely heavily on outdated Actions and bootstrap steps (`actions/checkout@v2/v3`, `actions/setup-python@v2`, `conda-incubator/setup-miniconda@v2`, `hashicorp/setup-terraform@v1`, manual `conda install pip`, and legacy `pip` flows).
- [x] Confirmed recent failure signatures that should shape the first implementation passes:
- [x] `synth.yml` fails because `smartnoise-synth` resolves `smartnoise-sql>=1.0.7` from PyPI instead of the local checkout.
- [x] `spark.yml` fails in current PySpark handling around `pre_aggregated` DataFrame type checks.
- [x] `bigquery.yml` fails when repo secrets / ADC are unavailable and still surfaces deprecated Terraform action behavior.
- [x] Confirmed we want to **keep and modernize all existing workflows**, not retire them.

## Phase 1 - Define the repo-wide CI contract and uv bootstrap

**Goal:** establish a supported Python policy and one modern environment bootstrap pattern before editing individual workflows.

- [x] Set the canonical CI support window to **3.9-3.12** unless package metadata is intentionally changed in the same PR.
- [x] Decide the default single-version job runtime as **3.12**, with an expanded matrix used only where it adds value.
- [x] Establish the Conda replacement pattern with `actions/setup-python` plus `astral-sh/setup-uv` and shared caching.
- [x] Choose one reusable install pattern for local packages, test requirements, and editable installs across `sql`, `synth`, and `eval`.
- [x] Add and document a reusable CI helper surface so every job stops hand-rolling Python setup.

**Likely files:** `.github/workflows/*.yml`, `.github/workflows/README.md`, optional `.github/actions/...`

**Exit criteria:** every later phase can reuse one agreed Python/uv bootstrap pattern without re-deciding versions or install semantics.

**Phase 1 handoff:** the shared bootstrap now lives in `.github/actions/setup-python-uv/action.yml`, and `ci_contract.yml` is the proving workflow for the repo-wide contract. Phase 2 should migrate `sql.yml`, `synth.yml`, `lint_sql.yml`, and `synth_lint.yml` to this helper rather than re-implementing Python setup, caching, or editable install ordering.

## Phase 2 - Modernize the core PR gates for SQL and synth

**Goal:** fix the workflows most relevant to PR #626 and make coupled `sql` + `synth` changes test reliably.

- [ ] Update `sql.yml`, `synth.yml`, `lint_sql.yml`, and `synth_lint.yml` to current major Actions versions.
- [ ] Convert those workflows from ad hoc `pip` / Conda setup to the Phase 1 `uv` pattern.
- [ ] Ensure `synth.yml` installs the local `sql` package from the checkout before testing `synth`, so cross-package PRs do not depend on PyPI state.
- [ ] Revisit path filters so `synth` CI reruns when shared dependency surfaces in `sql` change in ways that can affect `synth`.
- [ ] Keep the fast PR gates small enough to stay useful, but broad enough that PR #626 gets meaningful coverage in both packages.

**Likely files:** `.github/workflows/sql.yml`, `.github/workflows/synth.yml`, `.github/workflows/lint_sql.yml`, `.github/workflows/synth_lint.yml`

**Exit criteria:** the repo has a modern, uv-based PR gate for SQL and synth that can validate PR #626 without relying on stale package publishing behavior.

## Phase 3 - Refresh local database integration workflows

**Goal:** modernize the workflows that exercise local relational engines without mixing in Spark or cloud auth complexity.

- [ ] Update `postgres.yml`, `mysql.yml`, and `windows.yml` to current supported Actions components.
- [ ] Move those jobs to the shared Python/uv bootstrap while preserving DB-specific setup.
- [ ] Replace brittle package-install and service-start steps with the most current supported runner/service patterns.
- [ ] Normalize fixture setup, dataset downloads, and environment variable handling so the three workflows behave consistently.
- [ ] Re-check whether any workflow should use service containers instead of manual OS-level package installs.

**Likely files:** `.github/workflows/postgres.yml`, `.github/workflows/mysql.yml`, `.github/workflows/windows.yml`

**Exit criteria:** Postgres, MySQL, and Windows SQL integration jobs are on supported Actions/tooling and behave consistently with the core PR gates.

## Phase 4 - Repair and modernize the Spark workflow

**Goal:** make Spark green again on supported Python and current PySpark behavior.

- [ ] Update `spark.yml` to the shared Python/uv bootstrap and current Actions versions.
- [ ] Pin or validate a supported Java + PySpark combination for the selected Python version.
- [ ] Fix the failing Spark path around `pre_aggregated` DataFrame handling if current PySpark type/module names differ from the code's assumptions.
- [ ] Decide whether the Spark job should run on one canonical Python version or a wider matrix once it is stable.
- [ ] Keep Spark-specific skips / env vars explicit and documented rather than relying on inherited shell state.

**Likely files:** `.github/workflows/spark.yml`, Spark-sensitive SQL test/code paths if needed

**Exit criteria:** Spark is no longer blocked by outdated environment setup or stale PySpark assumptions.

## Phase 5 - Modernize the BigQuery workflow and secret handling

**Goal:** keep BigQuery coverage while making auth, Terraform, and fork behavior sane.

- [ ] Update `bigquery.yml` to current major Actions versions, including Terraform setup.
- [ ] Replace the current credential flow with the recommended Google auth pattern for GitHub Actions.
- [ ] Make the workflow behave intentionally when secrets are unavailable (for example, skip with a clear reason on forks instead of failing inside Terraform).
- [ ] Remove deprecated Terraform action behavior and commit/provider pinning gaps if the workflow still generates new lock state at runtime.
- [ ] Keep the full integration path active for trusted repo contexts where the required secrets exist.

**Likely files:** `.github/workflows/bigquery.yml`, `sql/tests/terraform/bigquery/*`, workflow docs for secrets

**Exit criteria:** BigQuery remains part of CI, but fails only for real integration issues rather than missing credentials or deprecated action plumbing.

## Phase 6 - Refresh docs and sample-notebook workflows

**Goal:** bring the non-core workflows up to current Python/tooling and fix obvious repository drift.

- [ ] Update `docs.yml` and `samples-test.yml` to current Actions versions and the shared Python/uv bootstrap.
- [ ] Remove Python 3.8-era assumptions and keep these jobs inside the supported 3.9-3.12 window.
- [ ] Reconcile stale repository assumptions in the sample workflow, including the current `./sdk` install path.
- [ ] Revisit old Sphinx pins and Python-stdlib backports such as `pathlib` so docs install cleanly on supported Python.
- [ ] Decide whether these workflows should stay manual-only or become PR / push checks after they are green.

**Likely files:** `.github/workflows/docs.yml`, `.github/workflows/samples-test.yml`, `docs/requirements.txt`, `sql/docs/requirements.txt`, `synth/docs/requirements.txt`, `.github/workflows/README.md`

**Exit criteria:** docs and sample workflows are modernized enough to run intentionally and no longer encode obviously obsolete repo layout or Python assumptions.

## Phase 7 - Full validation, branch protection alignment, and follow-through

**Goal:** make the refreshed CI usable for approving PR #626 and easier to keep current afterward.

- [ ] Run the full workflow set on a branch that includes the CI refresh and the PR #626 code path.
- [ ] Confirm the status check names expected by branch protection still match the workflows after modernization.
- [ ] Update `.github/workflows/README.md` with the new uv-based local workflow instructions and secret expectations.
- [ ] Add a lightweight maintenance mechanism for GitHub Actions and Python dependencies (for example Dependabot or Renovate, if acceptable for this repo).
- [ ] Record any intentionally manual-only workflows and the reason they are not blocking PRs.

**Likely files:** `.github/workflows/README.md`, workflow YAMLs, repository settings follow-up outside git if needed

**Exit criteria:** the repo has a credible path to getting PR #626 through green CI and a lower chance of drifting back into unsupported CI/CD state.

## Notes for implementation agents

- Prefer **uv** for dependency installation and caching unless a phase uncovers a hard blocker.
- Treat **Python 3.13+ and 3.14** as out of scope for CI until package metadata is intentionally updated.
- Keep phases focused: do not mix BigQuery auth work into core PR gate cleanup, and do not expand Spark debugging into unrelated SQL workflow churn.
- When a phase finishes, update its completed tasks here from `[ ]` to `[x]` in the same PR so the next agent starts from the current state.
