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
