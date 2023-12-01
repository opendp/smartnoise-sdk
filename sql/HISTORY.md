# SmartNoise SQL v1.0.3 Release Notes

* Upgrade to OpenDP v0.8.0
* Better type hints (thanks, @mhauru!)

# SmartNoise SQL v1.0.2 Release Notes

* Fix privacy bug in approx_bounds (thanks, @TedTed)
* Remove buggy dead code for quantile computation

# SmartNoise SQL v1.0.1 Release Notes

* Upgrade to OpenDP v0.7.0
* Upgrade to PyYAML v6.0.1 (Thanks, @mhauru!)

# SmartNoise SQL v1.0.0 Release Notes

* Switch to use Pandas 2.0 and SQLAlchemy 2.0
* Remove dependency on pandasql
* Allow multiple dataframes to be passed to `from_df`

WARNING: While this release is intended to be backwards compatible with previous versions, this is a major change and may introduce bugs.  Please report any issues you find.

# SmartNoise SQL v0.2.12 Release Notes

* Fix bug where other counts would borrow `SELECT COUNT(*)` when `SELECT COUNT(*)` was used first
* Fix bug where queries would fail if multiple tables in a single metadata had different `max_contrib`.
* Change behavior of VAR and STDDEV to return zero if VAR is negative.

Thanks to @mhauru for reporting these issues!

# SmartNoise SQL v0.2.11 Release Notes

* Fix bug where EXTRACT function would lose datepart after symbols loaded

# SmartNoise SQL v0.2.10 Release Notes

* Fix bug where approximate bounds would fail in some cases

# SmartNoise SQL v0.2.9 Release Notes

* MySql and SQLite readers
* HAVING and ORDER BY allow expressions in addition to columns

# SmartNoise SQL v0.2.8 Release Notes

* Fix bug where integer sums can overflow i32.  All engines default to 64-bit integers now.

# SmartNoise SQL v0.2.7 Release Notes

* Fix Postgres Reader to rollback on failed transaction (thanks, @FishmanL!)

# SmartNoise SQL v0.2.6 Release Notes

* Update to use OpenDP v0.6

# SmartNoise SQL v0.2.5 Release Notes

* Update to use OpenDP v0.5, support for Mac Silicon
* Switch to use discrete Laplace for all integer sums and counts
* Enable discrete Gaussian option
* `get_privacy_cost` now allows lists of queries

# SmartNoise SQL v0.2.4 Release Notes

* Support for BigQuery (thanks, @oskarkocol!)
* Support for nullable and fixed imputation
* Allow override of sensitivity via metadata

# SmartNoise SQL v0.2.3 Release Notes

* Add scalar functions to AST
* Allow GROUP BY to group on complex expressions

# SmartNoise SQL v0.2.2 Release Notes

* Switch to `opendp` Geometric and Laplace for counts and sums (was previously Gaussian)
* Allow callers to override default mechanism choices
* Fix bug where `SELECT COUNT(*)` would fail in some cases
* Take accuracies from opendp rather than scipy
* New method, `get_simple_accuracies`, that reports accuracy for simple statistics that don't vary based on row value.  This supports getting accuracies before running the query.
* Add ability to pass in `_pre_aggregated` rows, allows queries against pre-aggregated data


# SmartNoise SQL v0.2.1.1 Release Notes

* Bug fix for CI


# SmartNoise SQL v0.2.1 Release Notes

* Initial release.  
* Split smartnoise-sdk into 3 packages, switch to use `opendp` instead of `smartnoise-core`.
