[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a href="https://smartnoise.org"><img src="images/SmartNoise/SVG/Logo Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>

# SmartNoise SDK: Tools for Differential Privacy on Tabular Data


The SmartNoise SDK includes 2 packages:
* [smartnoise-sql](sql/): Run differentially private SQL queries
* [smartnoise-synth](synth/): Generate differentially private synthetic data

To get started, see the examples below. Click into each project for more detailed examples.

## SQL

[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C3.9%20%7C%203.10-blue)](https://www.python.org/)

### Install

```bash
pip install smartnoise-sql
```

### Query

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

`PUMS.csv` and `PUMS.yaml` can be found in the [datasets](datasets/) folder.

See the [SQL project](sql/README.md)

## Synthesizers

[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)

### Install
```
pip install smartnoise-synth
```

### MWEM

```python
import pandas as pd
import numpy as np

pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/
pums = pums.drop(['income'], axis=1)
nf = pums.to_numpy().astype(int)

synth = snsynth.MWEMSynthesizer(epsilon=1.0, split_factor=nf.shape[1]) 
synth.fit(nf)

sample = synth.sample(10) # get 10 synthetic rows
print(sample)
```

### PATE-CTGAN

```python
import pandas as pd
import numpy as np
from snsynth.pytorch.nn import PATECTGAN
from snsynth.pytorch import PytorchDPSynthesizer

pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/
pums = pums.drop(['income'], axis=1)

synth = PytorchDPSynthesizer(1.0, PATECTGAN(regularization='dragan'), None)
synth.fit(pums, categorical_columns=pums.columns.values.tolist())

sample = synth.sample(10) # synthesize 10 rows
print(sample)
```

See the [Synthesizers project](synth/README.md)

## Communication

- You are encouraged to join us on [GitHub Discussions](https://github.com/opendp/opendp/discussions/categories/smartnoise)
- Please use [GitHub Issues](https://github.com/opendp/smartnoise-sdk/issues) for bug reports and feature requests.
- For other requests, including security issues, please contact us at [smartnoise@opendp.org](mailto:smartnoise@opendp.org).

## Releases and Contributing

Please let us know if you encounter a bug by [creating an issue](https://github.com/opendp/smartnoise-sdk/issues).

We appreciate all contributions. Please review the [contributors guide](contributing.rst).  We welcome pull requests with bug-fixes without prior discussion.

If you plan to contribute new features, utility functions or extensions to this system, please first open an issue and discuss the feature with us.
