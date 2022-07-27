=======================
SmartNoise Synthesizers
=======================

.. contents:: Table of Contents
  :local:
  :depth: 3

API Reference
=============
.. toctree::
  :glob:
  :maxdepth: 2

  API index <api/index>

Getting Started
===============

MWEM
----

Multiplicative Weights Exponential Mechanism.  From "`A Simple and Practical Algorithm for Differentially Private Data Release <https://www.cs.huji.ac.il/~katrina//papers/mwem-nips.pdf>`_".

.. code-block:: python

  import snsynth
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

PATE-CTGAN
----------

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


MST
---

MST achieves state of the art results for marginals over categorical data, and does well even with small source data.  From McKenna et al. "`Winning the NIST Contest: A scalable and general approach to differentially private synthetic data <https://arxiv.org/abs/2108.04978>`_"

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


This is version |version| of the guides, last built on |today|.

.. |opendp-logo| image:: _static/images/opendp-logo.png
   :class: img-responsive
