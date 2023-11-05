=======================
SmartNoise Synthesizers
=======================

.. contents:: Table of Contents
  :local:
  :depth: 3


Getting Started
===============

Create a synthesizer with the ``Synthesizer.create()`` method, passing in the name of the sythesizer you want to create, along with a privacy budget and any synthesizer-specific hyperparameters.  To see a list of available synthesizers, use the ``Synthesizer.list_synthesizers()`` method or read the `Synthesizer Reference <synthesizers/index.html>`_.

Each synthesizer has a ``fit()`` method that fits the synthesizer to a private data set, and a ``sample()`` method that generates synthetic data from the fitted synthesizer.  Each synthesizer also has a ``fit_sample()`` helper method that combines the ``fit()`` and ``sample()`` methods into a single call. By using the ``sample_conditional()`` method one can generate samples that satisfy certain conditions. It performs rejection sampling and can enable analytics without prior retention of the synthetic data.

.. code-block:: python

  from snsynth import Synthesizer
  import pandas as pd

  pums = pd.read_csv('PUMS.csv', index_col=None) # in datasets/
  pums = pums.drop(['income', 'age'], axis=1)

  synth = Synthesizer.create('mwem', epsilon=1.0)
  sample = synth.fit_sample(pums)
  print(sample)

  sample_conditional = synth.sample_conditional(100, "age < 50 AND income > 1000")
  print(sample_conditional)

Preprocessing Privacy Budget
----------------------------

The synthesizer will attempt to automatically prepocess the data into a format suitable for that synthesizer.  For example, the ``mwem`` synthesizer requires that categorical variables be encoded as integers, and the ``dpctgan`` synthesizer requires categories to be one-hot encoded.  In some cases, pre-processing will consume some privacy budget.  For example, binning or scaling continuous columns requires bounds, and approximate bounds will be computed with some privacy cost if no bounds are provided by the analyst.  To specify the amount of budget to be used for preprocessing,  you can pass a ``preprocessor_eps`` argument to the ``Synthesizer.create()`` method.

.. code-block:: python

  from snsynth import Synthesizer
  import pandas as pd

  pums = pd.read_csv('PUMS.csv', index_col=None) # in datasets/
  synth = Synthesizer.create('dpctgan', epsilon=1.0, verbose=True)
  sample = synth.fit_sample(pums, preprocessor_eps=0.5)
  print(sample)

In the above, the ``dpctgan`` synthesizer will use 0.5 privacy budget for preprocessing, and 0.5 privacy budget for synthesizing the data.  In the above, we also set the synthesizer to be ``verbose``, so we can see if the preprocessor budget was actually spent.  If, for example, mwem decided that all columns in the data were categorical, then no budget would be spent for preprocessing, and the full epsilon of 1.0 would be available to fit the synthesizer.

Preprocessor Hints
------------------

When inferring a preprocessor, the synthesizer first attempts to determine whether each column is categorical, ordinal, or continuous.  To skip this step and tell the synthesizer how to treat each column, you can pass in the ``categorical_columns``, ``ordinal_columns``, and ``continuous_columns`` arguments to the ``fit()`` method.  Additionally, if you know that columns can have missing values, you can specify ``nullable=True``.

.. note::

  Before using the MST synthesizer, please install ``mbi`` by running ``pip install git+https://github.com/ryan112358/private-pgm.git``.

.. code-block:: python

  from snsynth import Synthesizer
  import pandas as pd

  pums = pd.read_csv('PUMS_null.csv', index_col=None) # in datasets/
  pums.drop(['pid'], axis=1, inplace=True)
  categorical_columns = list(pums.columns)
  categorical_columns.remove(['income', 'age')
  synth = Synthesizer.create('mst', epsilon=1.0, verbose=True)
  sample = synth.fit_sample(
    pums, 
    categorical_columns=categorical_columns,
    continuous_columns=['income', 'age'],
    preprocessor_eps=0.5,
    nullable=True
  )
  print(sample)

In the above, we tell the synthesizer that all columns are categorical, except for ``income`` and ``age``, which are continuous.  We also tell the synthesizer that the data may contain missing values, so the synthesizer will use a special preprocessor that can handle missing values.

Data Transforms
===============

Even with preprocessing hints, the preprocessor inferred by the synthesizer may not be exactly what you want.  For example, the ``mwem`` synthesizer will automatically bin continuous columns into 10 bins.  And spending epsilon to infer bounds is wasteful and reduces accuracy when you already have public bounds for continuous columns.  In most cases, you will get the best performance by manually specifying the preprocessor you want to use.  Prepreocessing is done by a ``TableTransformer`` object, which implements a differentially private reversible data transform.

In this example, we provide fixed bounds for age and income, and log-transform income before scaling.  We use the one-hot encoding style, because we will be using this transformer in a GAN:

.. code-block:: python

    from snsynth import Synthesizer
    from snsynth.transform import *

    pums = pd.read_csv('PUMS.csv', index_col=None) # in datasets/

    tt = TableTransformer([
        MinMaxTransformer(lower=18, upper=70), # age
        ChainTransformer([LabelTransformer(), OneHotEncoder()]), # sex
        ChainTransformer([LabelTransformer(), OneHotEncoder()]), # educ
        ChainTransformer([LabelTransformer(), OneHotEncoder()]), # race
        ChainTransformer([
            ClampTransformer(lower=1),
            LogTransformer(),
            MinMaxTransformer(lower=0, upper=np.log(420000)) # income
        ]),
        ChainTransformer([LabelTransformer(), OneHotEncoder()]), # married
    ])

    synth = Synthesizer.create('dpctgan', epsilon=1.0, verbose=True)
    sample = synth.fit_sample(pums, transformer=tt, preprocessor_eps=0.0)

    assert (synth.odometer.spent == (0.0, 0.0))

For more information on the different transforms, see the `Data Transformers Reference <transforms/index.html>`_.

Synthesizers Reference
======================
.. toctree::
  :glob:
  :maxdepth: 3

  Synthesizers Index <synthesizers/index>


Data Transformers Reference
===========================
.. toctree::
  :glob:
  :maxdepth: 2

  Data Transformers Index <transforms/index>

This is version |version| of the guides, last built on |today|.

.. |opendp-logo| image:: _static/images/opendp-logo.png
   :class: img-responsive
