# SmartNoise Synth v1.0.3 Release Notes

* Switch to use SmartNoise SQL v1.0.3

# SmartNoise Synth v1.0.2 Release Notes

* Switch to use SmartNoise SQL v1.0.2
* Synthesizers now convert integer epsilon to float rather than crashing (thanks, @lo2aayy)

# SmartNoise Synth v1.0.1 Release Notes

* Upgrade to OpenDP v0.7.0
* Upgrade to SmartNoise SQL v1.0.1

# SmartNoise Synth v1.0.0 Release Notes

* Update to SmartNoise SQL v1.0.0
* Prepare for breaking changes to remove transformers to separate package and update to Torch 2.0 in future release
* Fix MWEM samples


# SmartNoise Synth v0.3.7 Release Notes

* Update PAC-Synth to v0.0.8
* Update smartnoise-sql dependency to >=0.2.10

# SmartNoise Synth v0.3.6 Release Notes

* Add AIM Synthesizer (thanks, @aholovenko and @lurosenb!)
* Switch additive noise to use OpenDP v0.6

# SmartNoise Synth v0.3.5 Release Notes

* Callers can constrain inference with hints (thanks, @neat-web!)
* DropTransformer can now be used to drop columns

# SmartNoise Synth v0.3.4 Release Notes

* Anonymizing transformer for PII (thanks, @neat-web!)
* DateTime transformer
* Fix MinMax and StandardScaler to replace null with 0.0 when indicator column present

# SmartNoise Synth v0.3.3 Release Notes

* Support for conditional sampling
* Fix bug with onehot encoding on single-category column

Thanks to @neat-web for both contributions!

# SmartNoise Synth v0.3.2 Release Notes

* Update to OpenDP v0.6

# SmartNoise Synth v0.3.1 Release Notes

* Update PAC-Synth to v0.6
* Fix depenedencies to allow StandardScaler to run on Windows

# SmartNoise Synth v0.3.0 Release Notes

This release is a breaking change from v0.2.x.

* Add `Synthesizer.create()` factory method as preferred way to create synthesizers.  See [Getting Started](https://docs.smartnoise.org/synth/index.html#getting-started) for new factory syntax.
* Add library for differentially private reversible data transforms.  All synthesizers now accept a `TableTransformer` object, and infer one if none provided.  See [Data Transforms](https://docs.smartnoise.org/synth/index.html#data-transforms) for more information.
* All synthesizers use safe differentially private preprocessing by default.
* Removed option for `log_frequency` from CTGAN synthesizers.
* Support for Apple Silicon

# SmartNoise Synth v0.2.8.1 Release Notes

* Add diagnostics to GANs to show epsilon spend on preprocessor

# SmartNoise Synth v0.2.8 Release Notes

* Support for pac-synth DP Marginals synthesizer (thanks, @rracanicci!)
* Bump Python requirement to 3.7+

# SmartNoise Synth v0.2.7 Release Notes

## MWEM Updates

* Support for measuring Cuboids. Cuboids include multiple disjoint queries that can be measured under a single iteration.
* Default iterations and query count adapt based on dimensionality of source data
* Support for measure-only MWEM, for small cubes with optimal query workloads
* Basic accountant keeps track of spent epsilon
* Removed bin edge support, since we delegate to preprocessor now
* Better handles cases where exponential mechanism can't find a query. Should always find queries to measure now
* Debug flag prints trace information

# SmartNoise Synth v0.2.6 Release Notes

* Support for MST synthesizer.
* Re-enabled support for continuous values in GAN synthesizers.
* Fixed bug where MWEM was adding too much noise

# SmartNoise Synth v0.2.5 Release Notes

Bug fix where CTGAN synthesizers could silently use continuous column if called without PytorchDPSynthesizer wrapper.

# SmartNoise Synth v0.2.4 Release Notes

* Fixed bug in dpsgd synthesizers where final batch was not being counted against budget, potentially causing privacy leak
* Alert caller if continuous column is passed as a categorical column to CTGAN
* Warn if log_frequency for CTGAN is set to unsafe value.  Spend a small fraction of epsilon to estimate frequencies for conditional sampling.
* Fixed DPCTGAN regression that was impairing utility

# SmartNoise Synth v0.2.3 Release Notes

* MWEM allows sampling of arbitrary number of records
* Missing splits now handled correctly
* All synthesizers support numpy or pandas input

# SmartNoise Synth v0.2.2 Release Notes

* Update to use newest CTGAN

# SmartNoise Synth v0.2.1 Release Notes

* Initial release.  
* Split smartnoise-sdk into 3 packages, switch to use `opendp` instead of `smartnoise-core`.
