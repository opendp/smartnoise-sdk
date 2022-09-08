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

  synth = snsynth.MWEMSynthesizer(debug=True)
  synth.fit(nf)

  print(f"MWEM spent epsilon {synth.spent}")
  sample = synth.sample(100)
  print(sample)

MWEM maintains an in-memory copy of the full joint distribution, initialized to a uniform distribution, and updated with each iteration.  The size of the joint distribution in memory is the product of the cardinalities of columns of the input data, which may be much larger than the number of rows.  For example, in the PUMS dataset with income column dropped, the size of the in-memory histogram is 29,184 cells.  The size of the histogram can explode rapidly with multiple columns with high cardinality.  You can provide splits to divide the columns into independent subsets, which may dramatically reduce the memory requirement.

The MWEM algorithm operates by alternating between a step that selects a poorly-performing query via exponential mechanism, and then updating the estimated joint distribution via the laplace mechanism, to perform better on the selected query.  Over multiple iterations, the estimated joint distribution will become better at answering the selected workload of queries.

Because of this, the performance of MWEM is highly dependent on the quality of the candidate query workload.  The implementation tries to generate a query workload that will perform well.  You can provide some hints to influence the candidate queries.  By default, MWEM will generate workloads with all one-way and two-way marginals.  If you want to ensure three-way or higher marginals are candidates, you can use ``marginal_width``.  In cases where columns contain ordinal data, particularly in columns that are binned from continuous values, you can use ``add_ranges`` to ensure that candidate queries include range queries.  If range queries are more important that two-way marginals, for example, you can combine ``add_ranges`` with a ``marginal_width=1`` to suppress two-way marginals.

Each iteration spends a fraction of the budgeted epsilon.  Very large numbers of iterations may divide the epsilon too small, resulting in large measurement error on each measurement.  Conversely, using too few iterations will reduce error of individual measurements, but may interefere with the algorithm converging.  For example, if your data set has 15 total one-way and two-way cuboids, and you use iterations=15, every cuboid will be measured with the maximum possible epsilon, but the fit may be poor.

By default, the implementation chooses a number of iterations based on the size of the data.  The number of candidate queries generated will be based on the number of iterations, and will focus on marginals, unless ``add_ranges`` is specified.  You can also specify a ``q_count`` to ensure that a certain number of candidate queries is always generated, regardless of the number of iterations.  This can be useful when you care more about range queries than marginals, because range queries each touch a single slice, while marginal cuboids may update hundreds or thousands of slices.

If you are confident that the candidate query workload is high quality, you can use ``measure_only`` to skip the query selection step, and just sample uniformly from the workload.  This can double the budget available to measure the queries, but typically is not useful for anything but the smallest datasets.

The ``debug`` flag prints detailed diagnostics about the query workload generated and convergence status.

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


Aggregate Seeded
----------------

A differentially-private synthesizer that relies on DP Marginals to build synthetic data. It will compute DP Marginals (called aggregates) for your dataset up to and including a specified reporting length, and synthesize data based on the computed aggregated counts.

Based on the `Synthetic Data Showcase project <https://github.com/microsoft/synthetic-data-showcase>`_. DP documentation available `here <https://github.com/microsoft/synthetic-data-showcase/tree/main/docs/dp>`_.

Before using Aggregate Seeded, install `pac-synth <pip install pac-synth>`_ :

.. code-block:: bash

  pip install pac-synth

And call like this:

.. code-block:: python

  from snsynth.aggregate_seeded import AggregateSeededSynthesizer

  # this generates a random pandas data frame with 5000 records
  # replace this with your own data
  sensitive_df = gen_data_frame(5000)

  # build synthesizer
  synth = AggregateSeededSynthesizer(epsilon=0.5)
  synth.fit(sensitive_df)

  # sample 5000 records and build a data frame
  synthetic_df = sensitive_df.sample(5000)

  # show 10 example records
  print(synthetic_df.sample(10))

  # this will output
  #      H1 H2  H3 H4 H5 H6 H7 H8 H9 H10
  # 2791  2         1  0  1  1  1  0   1
  # 2169  1  3   4  1  0  1  0  1  1   0
  # 4607     4   7  1  1  0  1  1  1   0
  # 4803  1      8  0  0  0  1  1  1   1
  # 2635         8  0  1  1  1  0  1   0
  # 537   1         1  1  1  1  1  0   0
  # 3495     6   7  0  0  1  0  0  1   0
  # 2009  1  3   3  0  0  1  0  1  1   0
  # 3214  1  5      1  1  1  1  1  0   1
  # 4879  2  5  10  0  1  1  1  1  1   1

For more, see the `samples notebook <https://github.com/opendp/smartnoise-sdk/tree/main/synth/samples/aggregate_seeded_sample>`_.


This is version |version| of the guides, last built on |today|.

.. |opendp-logo| image:: _static/images/opendp-logo.png
   :class: img-responsive
