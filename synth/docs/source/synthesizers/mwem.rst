===================================================
Multiplicative Weights Exponential Mechanism (MWEM)
===================================================

Multiplicative Weights Exponential Mechanism.  From "`A Simple and Practical Algorithm for Differentially Private Data Release <https://www.cs.huji.ac.il/~katrina//papers/mwem-nips.pdf>`_".

.. code-block:: python

  import pandas as pd
  from snsynth import Synthesizer

  pums = pd.read_csv("PUMS.csv")
  synth = Synthesizer.create("mwem", epsilon=3.0, verbose=True)
  synth.fit(pums, preprocessor_eps=1.0)
  pums_synth = synth.sample(1000)

MWEM maintains an in-memory copy of the full joint distribution, initialized to a uniform distribution, and updated with each iteration.  The size of the joint distribution in memory is the product of the cardinalities of columns of the input data, which may be much larger than the number of rows.  For example, in the code above, the dimensionality inferred will be about 300,000 cells, and training will take several minutes.  In the PUMS dataset with income column dropped, the size of the in-memory histogram is 29,184 cells.  The size of the histogram can explode rapidly with multiple columns with high cardinality.  You can provide splits to divide the columns into independent subsets, which may dramatically reduce the memory requirement. In the code below, MWEM will split the data into multiple disjoint cubes with 3 columns each (per the ``split_factor`` argument), and train a separate model for each cube.  The size of the in-memory histogram will be lower than 3,000 cells, and training will be relatively fast.

.. code-block:: python

  import pandas as pd
  from snsynth import Synthesizer

  pums = pd.read_csv("PUMS.csv")
  synth = Synthesizer.create("mwem", epsilon=3.0, split_factor=3, verbose=True)
  synth.fit(pums, preprocessor_eps=1.0)
  pums_synth = synth.sample(1000)

The MWEM algorithm operates by alternating between a step that selects a poorly-performing query via exponential mechanism, and then updating the estimated joint distribution via the laplace mechanism, to perform better on the selected query.  Over multiple iterations, the estimated joint distribution will become better at answering the selected workload of queries.

Because of this, the performance of MWEM is highly dependent on the quality of the candidate query workload.  The implementation tries to generate a query workload that will perform well.  You can provide some hints to influence the candidate queries.  By default, MWEM will generate workloads with all one-way and two-way marginals.  If you want to ensure three-way or higher marginals are candidates, you can use ``marginal_width``.  In cases where columns contain ordinal data, particularly in columns that are binned from continuous values, you can use ``add_ranges`` to ensure that candidate queries include range queries.  If range queries are more important that two-way marginals, for example, you can combine ``add_ranges`` with a ``marginal_width=1`` to suppress two-way marginals.

Each iteration spends a fraction of the budgeted epsilon.  Very large numbers of iterations may divide the epsilon too small, resulting in large measurement error on each measurement.  Conversely, using too few iterations will reduce error of individual measurements, but may interefere with the algorithm converging.  For example, if your data set has 15 total one-way and two-way cuboids, and you use iterations=15, every cuboid will be measured with the maximum possible epsilon, but the fit may be poor.

By default, the implementation chooses a number of iterations based on the size of the data.  The number of candidate queries generated will be based on the number of iterations, and will focus on marginals, unless ``add_ranges`` is specified.  You can also specify a ``q_count`` to ensure that a certain number of candidate queries is always generated, regardless of the number of iterations.  This can be useful when you care more about range queries than marginals, because range queries each touch a single slice, while marginal cuboids may update hundreds or thousands of slices.

If you are confident that the candidate query workload is high quality, you can use ``measure_only`` to skip the query selection step, and just sample uniformly from the workload.  This can double the budget available to measure the queries, but typically is not useful for anything but the smallest datasets.

Parameters
----------

.. autoclass:: snsynth.mwem.MWEMSynthesizer

