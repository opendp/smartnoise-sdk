================
Aggregate Seeded
================

A differentially-private synthesizer that relies on DP Marginals to build synthetic data. It will compute DP Marginals (called aggregates) for your dataset up to and including a specified reporting length, and synthesize data based on the computed aggregated counts.

Based on the `Synthetic Data Showcase project <https://github.com/microsoft/synthetic-data-showcase>`_. DP documentation available `here <https://github.com/microsoft/synthetic-data-showcase/tree/main/docs/dp>`_.

Call like this:

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

Parameters
----------

.. autoclass:: snsynth.aggregate_seeded.AggregateSeededSynthesizer
