SmartNoise SQL
==============



API Reference
-------------
.. toctree::
  :glob:
  :titlesonly:
  :maxdepth: 1

  API index <api/index>

Getting Started
===============

.. code-block:: python

  import snsql
  import pandas as pd
  privacy = snsql.Privacy(epsilon=1.0, delta=0.01)

  csv_path = 'PUMS.csv'
  meta_path = 'PUMS.yaml'

  pums = pd.read_csv(csv_path)
  reader = snsql.from_df(pums, privacy=privacy, metadata=meta_path)

  result = reader.execute('SELECT sex, AVG(age) AS age FROM PUMS.PUMS GROUP BY sex')

Metadata
========

.. toctree::
  :glob:
  :titlesonly:
  :maxdepth: 1

  Metadata <metadata.rst>

This is version |version| of the guides, last built on |today|.

.. |opendp-logo| image:: _static/images/opendp-logo.png
   :class: img-responsive
