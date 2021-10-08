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
    'version': '0.2.0.dev0',
    'description': 'Differentially Private SQL Queries',
    'long_description': '[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://www.python.org/)\n\n<a href="https://smartnoise.org"><img src="https://github.com/opendp/smartnoise-sdk/raw/main/images/SmartNoise/SVG/Logo%20Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>\n\n## SmartNoise SQL\n\nDifferentially private SQL queries.  Tested with:\n* PostgreSQL\n* SQL Server\n* Spark\n* Pandas (SQLite)\n* PrestoDB\n\nSmartNoise is intended for scenarios where the analyst is trusted by the data owner.  SmartNoise uses the [OpenDP](https://github.com/opendp/opendp) library of differential privacy algorithms.\n\n## Installation\n\n```\npip install smartnoise-sql\n```\n\n## Using\n\n```python\nimport snsql\nfrom snsql import Privacy\nimport pandas as pd\n\ncsv_path = \'PUMS.csv\'\nmeta_path = \'PUMS.yaml\'\n\ndata = pd.read_csv(csv_path)\nprivacy = Privacy(epsilon=1.0, delta=0.01)\nreader = snsql.from_connection(data, privacy=privacy, metadata=meta_path)\n\nresult = reader.execute(\'SELECT sex, AVG(age) AS age FROM PUMS.PUMS GROUP BY sex\')\n\nprint(result)\n```\n\n## Communication\n\n- You are encouraged to join us on [GitHub Discussions](https://github.com/opendp/opendp/discussions/categories/smartnoise)\n- Please use [GitHub Issues](https://github.com/opendp/smartnoise-sdk/issues) for bug reports and feature requests.\n- For other requests, including security issues, please contact us at [smartnoise@opendp.org](mailto:smartnoise@opendp.org).\n\n## Releases and Contributing\n\nPlease let us know if you encounter a bug by [creating an issue](https://github.com/opendp/smartnoise-sdk/issues).\n\nWe appreciate all contributions. We welcome pull requests with bug-fixes without prior discussion.\n\nIf you plan to contribute new features, utility functions or extensions to this system, please first open an issue and discuss the feature with us.',
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
