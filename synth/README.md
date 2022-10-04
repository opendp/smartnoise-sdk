[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://www.python.org/)

<a href="https://smartnoise.org"><img src="https://github.com/opendp/smartnoise-sdk/raw/main/images/SmartNoise/SVG/Logo%20Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>

# SmartNoise Synthesizers

Differentially private synthesizers for tabular data.  Package includes:
* MWEM
* MST
* QUAIL
* DP-CTGAN
* PATE-CTGAN
* PATE-GAN

## Installation

```
pip install smartnoise-synth
```

## Using

Please see the [SmartNoise synthesizers documentation](https://docs.smartnoise.org/synth/index.html) for usage examples.

## Note on Inputs

MWEM and MST require columns to be categorical. If you have columns with continuous values, you should discretize them before fitting.  Take care to discretize in a way that does not reveal information about the distribution of the data.

## Communication

- You are encouraged to join us on [GitHub Discussions](https://github.com/opendp/opendp/discussions/categories/smartnoise)
- Please use [GitHub Issues](https://github.com/opendp/smartnoise-sdk/issues) for bug reports and feature requests.
- For other requests, including security issues, please contact us at [smartnoise@opendp.org](mailto:smartnoise@opendp.org).

## Releases and Contributing

Please let us know if you encounter a bug by [creating an issue](https://github.com/opendp/smartnoise-sdk/issues).

We appreciate all contributions. Please review the [contributors guide](../contributing.rst). We welcome pull requests with bug-fixes without prior discussion.

If you plan to contribute new features, utility functions or extensions to this system, please first open an issue and discuss the feature with us.
