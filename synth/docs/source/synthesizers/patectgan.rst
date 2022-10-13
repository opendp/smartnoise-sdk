
Conditional tabular GAN using Private Aggregation of Teacher Ensembles.  From "`PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees <https://openreview.net/pdf?id=S1zk9iRqF7>`_" and "`Modeling Tabular data using Conditional GAN <https://arxiv.org/abs/1907.00503>`_".

.. code-block:: python

  import pandas as pd
  from snsynth import Synthesizer

  pums = pd.read_csv("PUMS.csv")

  synth = Synthesizer.create("patectgan", epsilon=3.0, verbose=True)
  synth.fit(pums, preprocessor_eps=1.0)
  pums_synth = synth.sample(1000)


Parameters
----------

.. autoclass:: snsynth.pytorch.nn.PATECTGAN