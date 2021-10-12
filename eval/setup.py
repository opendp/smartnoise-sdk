# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['sneval',
 'sneval.benchmarking',
 'sneval.evaluator',
 'sneval.explorer',
 'sneval.learner',
 'sneval.metrics',
 'sneval.params',
 'sneval.privacyalgorithm',
 'sneval.report']

package_data = \
{'': ['*']}

install_requires = \
['opendp>=0.3.0,<0.4.0', 'smartnoise-sql>=0.2,<0.3']

setup_kwargs = {
    'name': 'smartnoise-eval',
    'version': '0.2.0',
    'description': 'Differential Privacy Stochastic Evaluator',
    'long_description': '[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://www.python.org/)\n\n<a href="https://smartnoise.org"><img src="https://github.com/opendp/smartnoise-sdk/raw/main/images/SmartNoise/SVG/Logo%20Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>\n\n## SmartNoise Stochastic Evaluator\n\nTests differential privacy algorithms for privacy, accuracy, and bias.  Privacy tests are based on the method described in [section 5.3 of this paper](https://arxiv.org/pdf/1909.01917.pdf).\n\n## Installation\n\n```\npip install smartnoise-eval\n```\n\n## Communication\n\n- You are encouraged to join us on [GitHub Discussions](https://github.com/opendp/opendp/discussions/categories/smartnoise)\n- Please use [GitHub Issues](https://github.com/opendp/smartnoise-sdk/issues) for bug reports and feature requests.\n- For other requests, including security issues, please contact us at [smartnoise@opendp.org](mailto:smartnoise@opendp.org).\n\n## Releases and Contributing\n\nPlease let us know if you encounter a bug by [creating an issue](https://github.com/opendp/smartnoise-sdk/issues).\n\nWe appreciate all contributions. We welcome pull requests with bug-fixes without prior discussion.\n\nIf you plan to contribute new features, utility functions or extensions to this system, please first open an issue and discuss the feature with us.',
    'author': 'SmartNoise Team',
    'author_email': 'smartnoise@opendp.org',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://smartnoise.org',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7.1,<=3.9',
}


setup(**setup_kwargs)
