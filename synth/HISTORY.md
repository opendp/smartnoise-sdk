# SmartNoise Synth v0.2.4 Release Notes

* Fixed bug in dpsgd synthesizers where final batch was not being counted against budget, potentially causing privacy leak
* Alert caller if continuous column is passed as a categorical column


# SmartNoise Synth v0.2.3 Release Notes

* MWEM allows sampling of arbitrary number of records
* Missing splits now handled correctly
* All synthesizers support numpy or pandas input

# SmartNoise Synth v0.2.2 Release Notes

* Update to use newest CTGAN

# SmartNoise Synth v0.2.1 Release Notes

* Initial release.  
* Split smartnoise-sdk into 3 packages, switch to use `opendp` instead of `smartnoise-core`.
