# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['snsynth',
 'snsynth.aggregate_seeded',
 'snsynth.models',
 'snsynth.mst',
 'snsynth.pytorch',
 'snsynth.pytorch.nn',
 'snsynth.pytorch.nn.ctgan',
 'snsynth.transform']

package_data = \
{'': ['*']}

install_requires = \
['diffprivlib>=0.5.2,<0.6.0',
 'opacus>=0.14.0,<0.15.0',
 'pac-synth>=0.0.5,<0.0.6',
 'smartnoise-sql>=0.2.5,<0.3.0']

setup_kwargs = {
    'name': 'smartnoise-synth',
    'version': '0.3.0',
    'description': 'Differentially Private Synthetic Data',
    'long_description': '[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://www.python.org/)\n\n<a href="https://smartnoise.org"><img src="https://github.com/opendp/smartnoise-sdk/raw/main/images/SmartNoise/SVG/Logo%20Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>\n\n# SmartNoise Synthesizers\n\nDifferentially private synthesizers for tabular data.  Package includes:\n* MWEM\n* MST\n* QUAIL\n* DP-CTGAN\n* PATE-CTGAN\n* PATE-GAN\n\n## Installation\n\n```\npip install smartnoise-synth\n```\n\n## Using\n\nPlease see the [SmartNoise synthesizers documentation](https://docs.smartnoise.org/synth/index.html) for usage examples.\n\n## Note on Inputs\n\nMWEM and MST require columns to be categorical. If you have columns with continuous values, you should discretize them before fitting.  Take care to discretize in a way that does not reveal information about the distribution of the data.\n\n## Communication\n\n- You are encouraged to join us on [GitHub Discussions](https://github.com/opendp/opendp/discussions/categories/smartnoise)\n- Please use [GitHub Issues](https://github.com/opendp/smartnoise-sdk/issues) for bug reports and feature requests.\n- For other requests, including security issues, please contact us at [smartnoise@opendp.org](mailto:smartnoise@opendp.org).\n\n## Releases and Contributing\n\nPlease let us know if you encounter a bug by [creating an issue](https://github.com/opendp/smartnoise-sdk/issues).\n\nWe appreciate all contributions. Please review the [contributors guide](../contributing.rst). We welcome pull requests with bug-fixes without prior discussion.\n\nIf you plan to contribute new features, utility functions or extensions to this system, please first open an issue and discuss the feature with us.\n',
    'author': 'SmartNoise Team',
    'author_email': 'smartnoise@opendp.org',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://smartnoise.org',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<3.11',
}


setup(**setup_kwargs)
