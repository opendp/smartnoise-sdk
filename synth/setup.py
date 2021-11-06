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
['ctgan>=0.4.3,<0.5.0', 'opacus>=0.14.0,<0.15.0']

setup_kwargs = {
    'name': 'smartnoise-synth',
    'version': '0.2.3',
    'description': 'Differentially Private Synthetic Data',
    'long_description': '[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://www.python.org/)\n\n<a href="https://smartnoise.org"><img src="https://github.com/opendp/smartnoise-sdk/raw/main/images/SmartNoise/SVG/Logo%20Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>\n\n# SmartNoise Synthesizers\n\nDifferentially private synthesizers for tabular data.  Package includes:\n* MWEM\n* QUAIL\n* DP-CTGAN\n* PATE-CTGAN\n* PATE-GAN\n\n## Installation\n\n```\npip install smartnoise-synth\n```\n\n## Using\n\n### MWEM\n\n```python\nimport pandas as pd\nimport numpy as np\n\npums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/\npums = pums.drop([\'income\'], axis=1)\nnf = pums.to_numpy().astype(int)\n\nsynth = snsynth.MWEMSynthesizer(epsilon=1.0, split_factor=nf.shape[1]) \nsynth.fit(nf)\n\nsample = synth.sample(10)\nprint(sample)\n```\n### DP-CTGAN\n\n```python\nimport pandas as pd\nimport numpy as np\nfrom snsynth.pytorch.nn import DPCTGAN\nfrom snsynth.pytorch import PytorchDPSynthesizer\n\npums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/\npums = pums.drop([\'income\'], axis=1)\n\nsynth = PytorchDPSynthesizer(1.0, DPCTGAN(), None)\nsynth.fit(pums, categorical_columns=pums.columns)\n\nsample = synth.sample(10) # synthesize 10 rows\nprint(sample)\n```\n\n### PATE-CTGAN\n\n```python\nimport pandas as pd\nimport numpy as np\nfrom snsynth.pytorch.nn import PATECTGAN\nfrom snsynth.pytorch import PytorchDPSynthesizer\n\npums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/\npums = pums.drop([\'income\'], axis=1)\n\nsynth = PytorchDPSynthesizer(1.0, PATECTGAN(regularization=\'dragan\'), None)\nsynth.fit(pums, categorical_columns=pums.columns)\n\nsample = synth.sample(10) # synthesize 10 rows\nprint(sample)\n```\n\n## Communication\n\n- You are encouraged to join us on [GitHub Discussions](https://github.com/opendp/opendp/discussions/categories/smartnoise)\n- Please use [GitHub Issues](https://github.com/opendp/smartnoise-sdk/issues) for bug reports and feature requests.\n- For other requests, including security issues, please contact us at [smartnoise@opendp.org](mailto:smartnoise@opendp.org).\n\n## Releases and Contributing\n\nPlease let us know if you encounter a bug by [creating an issue](https://github.com/opendp/smartnoise-sdk/issues).\n\nWe appreciate all contributions. Please review the [contributors guide](../contributing.rst). We welcome pull requests with bug-fixes without prior discussion.\n\nIf you plan to contribute new features, utility functions or extensions to this system, please first open an issue and discuss the feature with us.',
    'author': 'SmartNoise Team',
    'author_email': 'smartnoise@opendp.org',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://smartnoise.org',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.6.8,<3.9',
}


setup(**setup_kwargs)
