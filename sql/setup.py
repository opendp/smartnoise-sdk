# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['snsql',
 'snsql._ast',
 'snsql._ast.expressions',
 'snsql.metadata',
 'snsql.reader',
 'snsql.sql',
 'snsql.sql._mechanisms',
 'snsql.sql.parser',
 'snsql.sql.reader',
 'snsql.xpath',
 'snsql.xpath.parser']

package_data = \
{'': ['*']}

install_requires = \
['PyYAML>=5.4.1,<6.0.0',
 'antlr4-python3-runtime==4.8',
 'graphviz>=0.17,<0.18',
 'numpy>=1.21.2,<2.0.0',
 'opendp>=0.3.0,<0.4.0',
 'pandas>=1.3.3,<2.0.0',
 'pandasql>=0.7.3,<0.8.0']

setup_kwargs = {
    'name': 'smartnoise-sql',
    'version': '0.2.0',
    'description': 'Differentially private SQL',
    'long_description': None,
    'author': 'SmartNoise Team',
    'author_email': 'smartnoise@opendp.org',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7.1,<=3.9',
}


setup(**setup_kwargs)
