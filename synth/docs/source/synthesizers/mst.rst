===========================
Maximum Spanning Tree (MST)
===========================

MST achieves state of the art results for marginals over categorical data, and does well even with small source data.  From McKenna et al. "`Winning the NIST Contest: A scalable and general approach to differentially private synthetic data <https://arxiv.org/abs/2108.04978>`_"

Before using MST, install `Private-PGM <pip install git+https://github.com/ryan112358/private-pgm.git>`_ :

.. code-block:: bash

  pip install git+https://github.com/ryan112358/private-pgm.git

And call like this:

.. code-block:: python

  import snsynth
  import pandas as pd
  import numpy as np

  pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/
  pums = pums.drop(['income'], axis=1)

  Domains = {"pums": "samples/mst_sample/pums-domain.json"} # in samples/mst_sample
  synth = snsynth.MSTSynthesizer(domains_dict=Domains, domain="pums", epsilon=1.0)
  synth.fit(pums)

  sample = synth.sample(10) # synthesize 10 rows
  print(sample)

For more, see the `sample notebook <https://github.com/opendp/smartnoise-sdk/tree/main/synth/samples/mst_sample>`_



Parameters
----------

.. autoclass:: snsynth.mst.mst.MSTSynthesizer
