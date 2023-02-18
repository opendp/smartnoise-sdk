===========================
Maximum Spanning Tree (MST)
===========================

MST achieves state of the art results for marginals over categorical data, and does well even with small source data.  From McKenna et al. "`Winning the NIST Contest: A scalable and general approach to differentially private synthetic data <https://arxiv.org/abs/2108.04978>`_"

Before using MST, install `Private-PGM <https://github.com/ryan112358/private-pgm.git>`_ :

.. code-block:: bash

  pip install git+https://github.com/ryan112358/private-pgm.git

And call like this:

.. code-block:: python

  import pandas as pd
  from snsynth import Synthesizer

  pums = pd.read_csv("PUMS.csv")

  synth = Synthesizer.create("mst", epsilon=3.0, verbose=True)
  synth.fit(pums, preprocessor_eps=1.0)
  pums_synth = synth.sample(1000)

For more, see the `sample notebook <https://github.com/opendp/smartnoise-sdk/tree/main/synth/samples/mst_sample>`_


Parameters
----------

.. autoclass:: snsynth.mst.mst.MSTSynthesizer
