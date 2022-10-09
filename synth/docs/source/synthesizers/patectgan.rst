
Conditional tabular GAN using Private Aggregation of Teacher Ensembles.  From "`PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees <https://openreview.net/pdf?id=S1zk9iRqF7>`_" and "`Modeling Tabular data using Conditional GAN <https://arxiv.org/abs/1907.00503>`_".

.. code-block:: python

  import snsynth
  import pandas as pd
  import numpy as np

  from snsynth.pytorch.nn import PATECTGAN
  from snsynth.pytorch import PytorchDPSynthesizer
  from snsynth.preprocessors.data_transformer import BaseTransformer

  pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/
  pums = pums.drop(['income'], axis=1)

  synth = PytorchDPSynthesizer(1.0, PATECTGAN(regularization='dragan'))
  synth.fit(pums, categorical_columns=list(pums.columns), transformer=BaseTransformer)

  sample = synth.sample(10) # synthesize 10 rows
  print(sample)


Parameters
----------

.. autoclass:: snsynth.pytorch.nn.PATECTGAN