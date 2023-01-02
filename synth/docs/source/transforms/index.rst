=================
Data Transformers
=================

.. contents:: Table of Contents
  :local:
  :depth: 3

All synthesizers take an optional ``transformer`` argument, which accepts a ``TableTransformer`` object.  The transformer is used to transform the data before synthesis and then reverse the transformation after synthesis.  A ``TableTransformer`` manages a list of ``ColumnTransformer`` objects, one for each column in the table.  Multiple transformations of a column can be chained together with a ``ChainTransformer``.

Using Data Transformers
=======================

Inferring a TableTransformer
----------------------------

The ``create`` factory method can be used to create a ``TableTransformer`` based on a data set, which can be a pandas dataframe, a numpy array, or a list of tuples.  The following exmaple creates a transformer that converts categorical columns to one-hot encoding.

.. code-block:: python

    from snsynth.transform.table import TableTransformer

    pums = pd.read_csv('PUMS.csv', index_col=None) # in datasets/
    pums = pums.drop(['income', 'age'], axis=1)
    cat_cols = list(pums.columns)

    tt = TableTransformer.create(pums, style='gan', categorical_columns=cat_cols)
    pums_encoded = tt.fit_transform(pums)

    # 26 columns in one-hot encodind
    assert(len(pums_encoded[0]) == 26)
    assert(len(pums_encoded) == len(pums))

    # round-trip
    pums_decoded = tt.inverse_transform(pums_encoded)
    assert(pums.equals(pums_decoded))
    
The default one-hot style is useful for neural networks, but is wasteful for cube-style synthesizer, such as MWEW, MST, and PAC-SYNTH.  The ``style`` argument can be used to specify a different style.  The following example creates a transformer that converts categorical columns into sequential label encoding.

.. code-block:: python

    from snsynth.transform.table import TableTransformer

    pums = pd.read_csv('PUMS.csv', index_col=None) # in datasets/
    pums = pums.drop(['income', 'age'], axis=1)
    cat_cols = list(pums.columns)

    tt = TableTransformer.create(pums, style='cube', categorical_columns=cat_cols)
    pums_encoded = tt.fit_transform(pums)

    # 4 columns in sequential label encoding
    assert(len(pums_encoded[0]) == 4)
    assert(len(pums_encoded) == len(pums))

    # round-trip
    pums_decoded = tt.inverse_transform(pums_encoded)
    assert(pums.equals(pums_decoded))

Inferring Bounds for Continuous Columns
---------------------------------------

In the examples above, we used only categorical columns, since continuous values need a min and max value to be transformed.  The ``create`` method can infer the min and max values from the data set.  Inferring the min and max requires some privacy budget, specified by the ``epsilon`` argument.

.. code-block:: python

    from snsynth.transform.table import TableTransformer

    pums = pd.read_csv('PUMS.csv', index_col=None) # in datasets/
    cat_cols = list(pums.columns)
    cat_cols.remove('income')
    cat_cols.remove('age')

    tt = TableTransformer.create(pums, style='cube', categorical_columns=cat_cols, continuous_columns=['age', 'income'])
    pums_encoded = tt.fit_transform(pums, epsilon=3.0)

    # 6 columns in sequential label encoding
    assert(len(pums_encoded[0]) == 6)
    assert(len(pums_encoded) == len(pums))

    # round-trip
    pums_decoded = tt.inverse_transform(pums_encoded)
    assert(round(pums['age'].mean()) == round(pums_decoded['age'].mean()))
    print(f"We used {tt.odometer.spent} when fitting the transformer")

Declaring a TableTransformer
----------------------------

In the above example, the transformer used some privacy budget to infer approximate bounds for the two continuous columns.  When bounds are known in advance, this is wasteful and can impact the accuracy of the synthesizer.  In cases where you want maximum control, you can specify your ``TableTransformer`` declaratively:

.. code-block:: python

    from snsynth.transform import *

    pums = pd.read_csv('PUMS.csv', index_col=None) # in datasets/

    tt = TableTransformer([
        MinMaxTransformer(lower=18, upper=70), # age
        LabelTransformer(), # sex
        LabelTransformer(), # educ
        LabelTransformer(), # race
        MinMaxTransformer(lower=0, upper=420000), # income
        LabelTransformer(), # married
    ])

    pums_encoded = tt.fit_transform(pums)

    # no privacy budget used
    assert(tt.odometer.spent == (0.0, 0.0)) 

    # round-trip
    pums_decoded = tt.inverse_transform(pums_encoded)
    assert(round(pums['age'].mean()) == round(pums_decoded['age'].mean()))


Individual column transformers can be chained together with a ``ChainTransformer``.  For example, we might want to convert each categorical column to a sequential label encoding, but then convert the resulting columns to one-hot encoding.  And we might want to log-transform the income column.  The following example shows how to do this:

.. code-block:: python

    from snsynth.transform import *

    pums = pd.read_csv('PUMS.csv', index_col=None) # in datasets/

    tt = TableTransformer([
        MinMaxTransformer(lower=18, upper=70), # age
        ChainTransformer([LabelTransformer(), OneHotEncoder()]), # sex
        ChainTransformer([LabelTransformer(), OneHotEncoder()]), # educ
        ChainTransformer([LabelTransformer(), OneHotEncoder()]), # race
        ChainTransformer([
            LogTransformer(),
            MinMaxTransformer(lower=0, upper=np.log(420000)) # income
        ]),
        ChainTransformer([LabelTransformer(), OneHotEncoder()]), # married
    ])

    pums_encoded = tt.fit_transform(pums)

    # no privacy budget used
    assert(tt.odometer.spent == (0.0, 0.0)) 

    # round-trip
    pums_decoded = tt.inverse_transform(pums_encoded)
    assert(round(pums['age'].mean()) == round(pums_decoded['age'].mean()))


Default TableTransformer
------------------------

If this argument is not provided, the synthesizer will attempt to infer the most appropriate transformer to map the data into the format expected by the synthesizer.

.. code-block:: python

    from snsynth.pytorch.nn import DPCTGAN
    from snsynth.mwem import MWEMSynthesizer
    import pandas as pd

    pums_csv_path = "PUMS.csv"
    pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/
    pums = pums.drop(['income', 'age'], axis=1)
    cat_cols = list(pums.columns)

    mwem = MWEMSynthesizer(epsilon=2.0)
    mwem.fit(pums, categorical_columns=cat_cols)
    print(f"MWEM inferred a cube transformer with {mwem._transformer.output_width} columns")

    dpctgan = DPCTGAN(epsilon=2.0)
    dpctgan.fit(pums, categorical_columns=cat_cols)
    print(f"DPCTGAN inferred a onehot transformer with {dpctgan._transformer.output_width} columns")


Anonymize personally identifiable information (PII)
---------------------------------------------------

To prevent leakage of sensitive information PII can be anonymized by generating fake data. The ``AnonymizationTransformer`` can be used with builtin methods of the `Faker <https://github.com/joke2k/faker>`_ library or with a custom callable. By default, existing values are discarded and new values will be generated during inverse transformation. If ``fake_inbound=True`` is provided, the new values are injected during transformation.

.. code-block:: python

    import random
    from snsynth.transform import *

    # example data set with columns: user ID, email, age
    pii_data = [(1, "email_1", 29), (2, "email_2", 42), (3, "email_3", 18)]

    tt = TableTransformer([
        AnonymizationTransformer(lambda: random.randint(0, 1_000)),  # generate random user ID
        AnonymizationTransformer("email"),  # fake email
        ChainTransformer([
            AnonymizationTransformer(lambda: random.randint(0, 100), fake_inbound=True),  # generate random age
            MinMaxTransformer(lower=0, upper=100)  # then use another transformer
        ])
    ])

    pii_data_transformed = tt.fit_transform(pii_data)
    assert all(len(t) == 1 for t in pii_data_transformed)  # only the faked age column could be used by a synthesizer

    pii_data_inversed = tt.inverse_transform(pii_data_transformed)
    assert all(a != b for a, b in zip(pii_data, pii_data_inversed))

Mixing Inferred and Declared Transformers
-----------------------------------------

In many cases, the inferred transformers will be mostly acceptable, with only a few columns requiring special handling.  In this case, the ``TableTransformer`` can set constraints on the inference to make sure that specific columns are handled differently.  For example, the following code will use the inferred transformer for all columns except for the ``income`` column, which will be transformed using ``LogTransformer`` and ``BinTransformer``:

.. code-block:: python

    import pandas as pd
    import math
    from snsynth.transform import *

    pums = pd.read_csv('PUMS_pid.csv', index_col=None)

    tt = TableTransformer.create(
        pums, 
        style='cube',
        constraints={
            'income': 
                ChainTransformer([
                    LogTransformer(),
                    BinTransformer(bins=20, lower=0, upper=math.log(400_000))
                ])
        }
    )
    tt.fit(pums, epsilon=1.0)
    print(tt.odometer.spent)
    income = tt.transformers[4]
    assert(isinstance(income, ChainTransformer))
    pid = tt.transformers[6]
    assert(isinstance(pid, AnonymizationTransformer))
    print(f"ID column is a {pid.fake} anonymization")

In the above example, the budget spent will be 0.0, because the bounds were specified for the income column.  All other columns are correctly inferred, with the identifier column using a sequence faker.

Note that the inferred columns will use ``cube`` style, so we use a ``BinTransformer`` to discretize the income column.  If we had used ``gan`` style, we would have used something more appropriate for a GAN, such as a ``OneHotEncoder`` or ``MinMaxTransformer``.

Constraints can also be specified with shortcut strings.  For example, if we want the identifier column to be faked with a random GUID rather than an integer sequence, we could manually construct the ``AnonymizationTransformer`` similar to the above, or we can just use the string ``"uuid4"``:

.. code-block:: python

    tt = TableTransformer.create(
        pums, 
        style='cube',
        constraints={
            'pid': 'uuid4'
        }
    )

Because this is a faker, it works the same regardless of the style.  Likewise, constraints can be specified as ``ordinal``, ``categorical``, or ``continuous`` to use the appropriate transformer for the column regardless of style.

In the below example, we use the ``"drop"`` constraint to drop the identifier column entirely.  We also specify that ``educ`` should be treated as continuous, even though it in an integer with only 13 levels.  This will cause the inferred transformer to use continuous transformers rather than discrete.  Both of these constraints will do the right thing regardless of the style.

.. code-block:: python

    tt = TableTransformer.create(
        pums, 
        style='cube',
        constraints={
            'educ': 'continuous',
            'pid': 'drop'
        }
    )


TableTransformer API
====================

.. autoclass:: snsynth.transform.table.TableTransformer
    :members:

.. autoclass:: snsynth.transform.table.NoTransformer

Column Transformers Reference
=============================

LabelTransformer
----------------

.. autoclass:: snsynth.transform.label.LabelTransformer

OneHotEncoder
-------------

.. autoclass:: snsynth.transform.onehot.OneHotEncoder

MinMaxTransformer
-----------------

.. autoclass:: snsynth.transform.minmax.MinMaxTransformer

StandardScaler
--------------

.. autoclass:: snsynth.transform.standard.StandardScaler

BinTransformer
--------------

.. autoclass:: snsynth.transform.bin.BinTransformer

LogTransformer
--------------

.. autoclass:: snsynth.transform.log.LogTransformer

ChainTransformer
----------------

.. autoclass:: snsynth.transform.chain.ChainTransformer

ClampTransformer
----------------

.. autoclass:: snsynth.transform.clamp.ClampTransformer

AnonymizationTransformer
------------------------

.. autoclass:: snsynth.transform.anonymization.AnonymizationTransformer

DropTransformer
------------------------

.. autoclass:: snsynth.transform.drop.DropTransformer