========================================================
Differentially Private Conditional Tabular GAN (DPCTGAN)
========================================================

Conditional tabular GAN with differentially private stochastic gradient descent.  From "`Modeling Tabular data using Conditional GAN <https://arxiv.org/abs/1907.00503>`_".

.. code-block:: python

  import snsynth
  import pandas as pd
  import numpy as np

  from snsynth.pytorch.nn import DPCTGAN
  from snsynth.pytorch import PytorchDPSynthesizer
  from snsynth.preprocessors.data_transformer import BaseTransformer

  pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/
  pums = pums.drop(['income'], axis=1)

  synth = PytorchDPSynthesizer(1.0, DPCTGAN())
  synth.fit(pums, categorical_columns=list(pums.columns), transformer=BaseTransformer)

  sample = synth.sample(10) # synthesize 10 rows
  print(sample)


Parameters
----------

.. autoclass:: snsynth.pytorch.nn.dpctgan.DPCTGAN
