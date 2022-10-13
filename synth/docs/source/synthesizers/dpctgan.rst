========================================================
Differentially Private Conditional Tabular GAN (DPCTGAN)
========================================================

Conditional tabular GAN with differentially private stochastic gradient descent.  From "`Modeling Tabular data using Conditional GAN <https://arxiv.org/abs/1907.00503>`_".

.. code-block:: python

  import pandas as pd
  from snsynth import Synthesizer

  pums = pd.read_csv("PUMS.csv")

  synth = Synthesizer.create("dpctgan", epsilon=3.0, verbose=True)
  synth.fit(pums, preprocessor_eps=1.0)
  pums_synth = synth.sample(1000)


Parameters
----------

.. autoclass:: snsynth.pytorch.nn.dpctgan.DPCTGAN
