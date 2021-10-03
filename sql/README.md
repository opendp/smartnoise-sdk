[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://www.python.org/)

<a href="https://smartnoise.org"><img src="https://github.com/opendp/smartnoise-sdk/raw/main/images/SmartNoise/SVG/Logo%20Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>

## SmartNoise SQL

Differentially private SQL queries.  Tested with:
* PostgreSQL
* SQL Server
* Spark
* Pandas (SQLite)
* PrestoDB

SmartNoise is intended for scenarios where the analyst is trusted by the data owner.  SmartNoise uses the [OpenDP](https://github.com/opendp/opendp) library of differential privacy algorithms.

## Installation

```
pip install smartnoise-sql
```

## Using

```python
import snsql
from snsql import Privacy
import pandas as pd

csv_path = 'PUMS.csv'
meta_path = 'PUMS.yaml'

data = pd.read_csv(csv_path)
privacy = Privacy(epsilon=1.0, delta=0.01)
reader = snsql.from_connection(data, privacy=privacy, metadata=meta_path)

result = reader.execute('SELECT sex, AVG(age) AS age FROM PUMS.PUMS GROUP BY sex')

print(result)
```

## Communication

- You are encouraged to join us on [GitHub Discussions](https://github.com/opendp/opendp/discussions/categories/smartnoise)
- Please use [GitHub Issues](https://github.com/opendp/smartnoise-sdk/issues) for bug reports and feature requests.
- For other requests, including security issues, please contact us at [smartnoise@opendp.org](mailto:smartnoise@opendp.org).

## Releases and Contributing

Please let us know if you encounter a bug by [creating an issue](https://github.com/opendp/smartnoise-sdk/issues).

We appreciate all contributions. We welcome pull requests with bug-fixes without prior discussion.

If you plan to contribute new features, utility functions or extensions to this system, please first open an issue and discuss the feature with us.