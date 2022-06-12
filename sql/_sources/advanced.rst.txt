##############
Advanced Usage
##############

This section covers some advanced usage scenarios.

Overriding Mechanisms
---------------------

You can override the default mechanisms used for differentially private summary statistics.  The mechanisms are specified in the ``mechanisms.map`` dictionary on the ``Privacy`` object.

.. code-block:: python

    from snsql import Privacy, Stat, Mechanism

    privacy = Privacy(epsilon=1.0)
    print(f"We default to using {privacy.mechanisms.map[Stat.count]} for counts.")
    print("Switching to use gaussian")
    privacy.mechanisms.map[Stat.count] = Mechanism.gaussian

The list of statistics that can be mapped is in the ``Stat`` enumeration, and the mechanisms available are listed in the ``Mechanism`` enumeration.  The ``AVG`` sumamry statistic is computed from a sum and a count, each of which can be overriden.

For integer sums, you can specify ``Stat.sum_int``, and specify ``Stat.sum_large_int`` separately for large integer sums.  A "large" integer sum is any sum that is greater than the value set in ``mechanisms.large``, which defaults to 1000.  This is primarily useful when using geometric mechanism for integer sums, since the geometric mechanism slows down for large integer sums.

pre_aggregated
--------------

By default, ``execute()`` uses the underlying database engine to compute exact aggregates.
You can pass in exact aggregates from a different source, using the ``pre_aggregated`` parameter.

.. code-block:: python

    query = 'SELECT sex, COUNT(*) AS n, AVG(age) AS age FROM PUMS.PUMS GROUP BY sex ORDER BY sex'

    pre_agg = [
        ['n', 'sex', 'sum_age'],
        [510, '1', 23500],
        [490, '0', 21000]
    ]

    for _ in range(3):
        res = reader.execute(query, pre_aggregated=pre_agg)
        print(res)

.. code-block:: bash

    [['sex', 'n', 'age'], ['0', 493, 42.33581999011567], ['1', 511, 44.43093430694777]]
    [['sex', 'n', 'age'], ['0', 487, 43.40306305599861], ['1', 511, 45.90357659295852]]
    [['sex', 'n', 'age'], ['0', 488, 43.11059968286903], ['1', 506, 46.21407160686184]]

In this example, the query will skip the database and add noise to ``pre_agg`` directly.
Each run will add different noise.  You can pass in any iterable over tuples.

Note that the exact aggregates provide a SUM instead of an AVG.  This is because SmartNoise computes AVG
from a noisy COUNT and a noisy SUM.  The ``execute()`` method expects the ``pre_aggregated`` values
to match what would be obtained by running the query against an underlying database engine.

You can see which columns are expected by checking the AST of the rewritten query:

.. code-block:: python

    query = 'SELECT sex, COUNT(*) AS n, AVG(age) AS age FROM PUMS.PUMS GROUP BY sex ORDER BY sex'

    subquery, _ = reader._rewrite(query)
    expressions = [
        f'{str(col.expression)} AS {col.name}' 
        for col in subquery.select.namedExpressions
    ]
    print(', '.join(expressions))

.. code-block:: bash

    ['COUNT ( * )', 'sex', 'SUM ( age )']

Here we see that the first column is ``COUNT(*)``, the second is ``sex`` and the third is ``SUM(age)``

postprocess
-----------

Several query operations do not affect privacy, because they happen after noise has been applied.
Examples include clamping negative counts, TOP/LIMIT, HAVING, and ORDER BY.  Computations such as AVG
are also performed on noisy values in post-processing.  Since the post-processing happens
after noise addition, caching layers may wish to extend budget by caching results
immediately before post-processing.

.. code-block:: python

    query = 'SELECT sex, COUNT(*) AS n FROM PUMS.PUMS GROUP BY sex'
    no_pp = reader.execute(query, postprocess=False)
    print(list(no_pp))

.. code-block::

    [[485.4821800946391, 0], [511.22682440467884, 1]]

Here we see that the counts have not been clamped to be integers, and the columns are ordered differently from the outer SELECT.  However, the counts are noisy, and ``censor_dims`` has been applied, so this result is suitable for caching and using in post-processing.

Note that the noisy counts are floats, because ``censor_dims`` is ``True`` by default.  If the metadata
had specified ``censor_dims=False``, then the geometric mechanism would be used for these specific counts, and the values would be integers.

Here is a more complex example:

.. code-block:: python

    query = 'SELECT TOP 2 educ, AVG(age) AS age FROM PUMS.PUMS GROUP BY educ ORDER BY age DESC'

    no_pp = reader.execute(query, postprocess=False)
    print(list(no_pp))

.. code-block::

    [[34.429285994199816, 1, 1679], [13.966503517008807, 2, 640], [39.30000608265984, 3, 1757], [17.211438317953128, 4, 888], [24.727002841061243, 5, 845], [18.247455233675588, 6, 869], [28.619036170132635, 7, 776], [50.41413180280105, 8, 2067], [200.23507699829014, 9, 8954], [58.75871160176575, 10, 2483], [165.14751392246907, 11, 7151], [75.87011805331791, 12, 3326], [178.57055363635266, 13, 8737], [52.596166495791834, 14, 2650], [23.02440993754067, 15, 1311], [14.743632346849909, 16, 305]]

Here we notice several things.  The counts are noisy and the ``educ`` values are not sorted in descending order.
The third column has a SUM instead of an AVG.  And the LIMIT is not applied.  But this rowset is differentially private, and
has everything necessary for post-processing.

The output when ``postprocess=False`` is the same as the input required for ``pre_aggregated``.
This allows patterns like the following:

.. code-block:: python

    query = 'SELECT TOP 2 educ, AVG(age) AS age FROM PUMS.PUMS GROUP BY educ ORDER BY age DESC'

    no_pp = reader.execute(query, postprocess=False)

    for _ in range(3):
        res = reader.execute(query, pre_aggregated=list(no_pp)) # postprocess=True
        print(res)

.. code-block::

    [['educ', 'age'], ['2', 69.56647634418115], ['4', 63.23184623593364]]
    [['educ', 'age'], ['4', 50.32885468901986], ['16', 49.724923251737366]]
    [['educ', 'age'], ['6', 54.17627519853133], ['1', 52.93913290533175]]

In this example, ``no_pp`` holds differentially private values, so ``pre_aggregated`` is
not actually the exact aggregates, but instead can be thought of as a simulated version of the
exact aggregates.  The loop runs multiple releases, adding noise each time, without
affecting the privacy cost of the original query.  This can be useful in cases
where you want to estimate error ranges via simulation without querying the exact aggregates
repeatedly.  It can also be useful when caching results to avoid spending budget.  For example,
in the above, the caller could do sequential queries with different LIMIT or ORDER BY,
without spending additional budget.  Queries for SUM (not requested in the original query) could
also be answered with no additional privacy cost.

Note that the result of ``postprocess=False`` will ensure that rare dimensions are censored,
to ensure that the result is differentially private.  Passing this result back in as
``pre_aggregated`` could result in additional dimensions near the threshold being
censored, because noise will be added again.  This may or may not be desirable, depending on your
application.  For example, if you are trying to estimate error ranges, you may want
to set ``censor_dims=False`` when generating the ``postprocess=False`` result, and then
set ``censor_dims=True`` on each of the simulated runs.
