# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['snsynth',
 'snsynth.models',
 'snsynth.preprocessors',
 'snsynth.pytorch',
 'snsynth.pytorch.nn']

package_data = \
{'': ['*']}

install_requires = \
['ctgan==0.2.2.dev0', 'numpy>=1.21.2,<2.0.0', 'opacus==0.11.0', 'torch==1.7.1']

setup_kwargs = {
    'name': 'smartnoise-synth',
    'version': '0.2.1',
    'description': 'Differentially Private Synthetic Data',
    'long_description': '[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://www.python.org/)\n\n<a href="https://smartnoise.org"><img src="https://github.com/opendp/smartnoise-sdk/raw/main/images/SmartNoise/SVG/Logo%20Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>\n\n## SmartNoise Synthesizers\n\nDifferentially private synthesizers for tabular data.  Package includes:\n* MWEM\n* QUAIL\n* PATE-CTGAN\n* DP-CTGAN\n* PATE-GAN\n\n## Installation\n\n```\npip install smartnoise-synth\n```\n\n## Using\n\n```python\nimport snsynth\nimport pandas as pd\n\nsynth = snsynth.MWEMSynthesizer(1.0)  # epsilon=1.0\nfit = synth.fit(my_data)  # learn the distribution of the real data\nsample = synth.sample(10) # synthesize 10 rows\n```\n\n## Communication\n\n- You are encouraged to join us on [GitHub Discussions](https://github.com/opendp/opendp/discussions/categories/smartnoise)\n- Please use [GitHub Issues](https://github.com/opendp/smartnoise-sdk/issues) for bug reports and feature requests.\n- For other requests, including security issues, please contact us at [smartnoise@opendp.org](mailto:smartnoise@opendp.org).\n\n## Releases and Contributing\n\nPlease let us know if you encounter a bug by [creating an issue](https://github.com/opendp/smartnoise-sdk/issues).\n\nWe appreciate all contributions. Please review the [contributors guide](../contributing.rst). We welcome pull requests with bug-fixes without prior discussion.\n\nIf you plan to contribute new features, utility functions or extensions to this system, please first open an issue and discuss the feature with us.',
    'author': 'SmartNoise Team',
    'author_email': 'smartnoise@opendp.org',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://smartnoise.org',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7.1,<=3.11',
}


setup(**setup_kwargs)
