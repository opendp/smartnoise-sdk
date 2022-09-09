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
