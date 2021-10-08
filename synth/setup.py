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
    'version': '0.2.0',
    'description': 'Differentially private synthetic data',
    'long_description': None,
    'author': 'SmartNoise Team',
    'author_email': 'smartnoise@opendp.org',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7.1,<=3.11',
}


setup(**setup_kwargs)
