SmartNoise Synthesizers
=======================


API Reference
-------------
.. toctree::
  :glob:
  :maxdepth: 2

  API index <api/index>

Getting Started
===============

MWEM
----

Multiplicative Weights Exponential Mechanism.

.. code-block:: python

  import pandas as pd
  import numpy as np

  pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/
  pums = pums.drop(['income'], axis=1)
  nf = pums.to_numpy().astype(int)

  synth = snsynth.MWEMSynthesizer(epsilon=1.0, split_factor=nf.shape[1]) 
  synth.fit(nf)

  sample = synth.sample(10)
  print(sample)

DP-CTGAN
--------

Conditional tabular GAN with differentially private stochastic gradient descent.

.. code-block:: python

  import pandas as pd
  import numpy as np
  from snsynth.pytorch.nn import DPCTGAN
  from snsynth.pytorch import PytorchDPSynthesizer

  pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/
  pums = pums.drop(['income'], axis=1)

  synth = PytorchDPSynthesizer(1.0, DPCTGAN(), None)
  synth.fit(pums, categorical_columns=pums.columns)

  sample = synth.sample(10) # synthesize 10 rows
  print(sample)

PATE-CTGAN
----------

Conditional tabular GAN using Private Aggregation of Teacher Ensembles.

.. code-block:: python

  import pandas as pd
  import numpy as np
  from snsynth.pytorch.nn import PATECTGAN
  from snsynth.pytorch import PytorchDPSynthesizer

  pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/
  pums = pums.drop(['income'], axis=1)

  synth = PytorchDPSynthesizer(1.0, PATECTGAN(regularization='dragan'), None)
  synth.fit(pums, categorical_columns=pums.columns)

  sample = synth.sample(10) # synthesize 10 rows
  print(sample)

This is version |version| of the guides, last built on |today|.

.. |opendp-logo| image:: _static/images/opendp-logo.png
   :class: img-responsive
