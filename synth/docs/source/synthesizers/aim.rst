==================================
Adaptive Iterative Mechanism (AIM)
==================================

 `AIM <https://arxiv.org/abs/2201.12677>`_ is a workload-adaptive algorithm, within the paradigm of algorithms that first selects a set of queries, then privately measures those queries, and finally generates synthetic data from the noisy measurements. It uses a set of innovative features to iteratively select the most useful measurements, reflecting both their relevance to the workload and their value in approximating the input data. AIM consistently outperforms a wide variety of existing mechanisms across a variety of experimental settings.

Before using AIM, install `Private-PGM <https://github.com/ryan112358/private-pgm.git>`_ :

.. code-block:: bash

  pip install git+https://github.com/ryan112358/private-pgm.git

And call like this:

.. code-block:: python

  import pandas as pd
  from snsynth import Synthesizer

  pums = pd.read_csv("PUMS.csv")

  synth = Synthesizer.create("aim", epsilon=3.0, verbose=True)
  synth.fit(pums, preprocessor_eps=1.0)
  pums_synth = synth.sample(1000)


Parameters
----------

.. autoclass:: snsynth.aim.aim.AIMSynthesizer
