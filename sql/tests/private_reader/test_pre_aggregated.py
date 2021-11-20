from snsql import *
import pandas as pd
import numpy as np

privacy = Privacy(epsilon=3.0, delta=0.1)


class TestPreAggregatedSuccess:
    # Test input checks for pre_aggregated
    def test_list_success(self, test_databases):
        # pass in properly formatted list
        pre_aggregated = [
            ('keycount', 'sex', 'count_star'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]
        query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="pandas"
        )
        if priv:
            res = priv.execute(query, pre_aggregated=pre_aggregated)
            assert(str(res[1][0]) == '1') # it's sorted
    def test_pandas_success(self, test_databases):
        # pass in properly formatted dataframe
        pre_aggregated = [
            ('keycount', 'sex', 'count_star'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]
        colnames = pre_aggregated[0]
        pre_aggregated = pd.DataFrame(data=pre_aggregated[1:], index=None)
        pre_aggregated.columns = colnames

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="pandas"
        )
        if priv:
            query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'
            res = priv.execute(query, pre_aggregated=pre_aggregated)
            assert(str(res[1][0]) == '1') # it's sorted
    def test_pandas_success_df(self, test_databases):
        # pass in properly formatted dataframe
        pre_aggregated = [
            ('keycount', 'sex', 'count_star'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]
        colnames = pre_aggregated[0]
        pre_aggregated = pd.DataFrame(data=pre_aggregated[1:], index=None)
        pre_aggregated.columns = colnames

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="pandas"
        )
        if priv:
            query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'
            res = priv.execute_df(query, pre_aggregated=pre_aggregated)
            assert(str(res['sex'][0]) == '1') # it's sorted
    def test_np_ndarray_success(self, test_databases):
        # pass in properly formatted dataframe
        pre_aggregated = [
            ('keycount', 'sex', 'count_star'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]
        colnames = pre_aggregated[0]
        pre_aggregated = pd.DataFrame(data=pre_aggregated[1:], index=None)
        pre_aggregated.columns = colnames
        pre_aggregated = pre_aggregated.to_numpy()

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="pandas"
        )
        if priv:
            query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'
            res = priv.execute(query, pre_aggregated=pre_aggregated)
            assert(str(res[1][0]) == '1') # it's sorted
    def test_np_array_success(self, test_databases):
        # pass in properly formatted dataframe
        pre_aggregated = [
            ('keycount', 'sex', 'count_star'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]
        pre_aggregated = np.array(pre_aggregated[1:])

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="pandas"
        )
        if priv:
            query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'
            res = priv.execute(query, pre_aggregated=pre_aggregated)
            assert(str(res[1][0]) == '1') # it's sorted
    def test_spark_df_success(self, test_databases):
        # pass in properly formatted dataframe
        pre_aggregated = [
            ('keycount', 'sex', 'count_star'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="spark"
        )
        if priv:
            pre_aggregated = priv.reader.api.createDataFrame(pre_aggregated[1:], pre_aggregated[0])
            query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'
            res = priv.execute(query, pre_aggregated=pre_aggregated)
            res = test_databases.to_tuples(res)
            assert(str(res[1][0]) == '1') # it's sorted
    def test_spark_df_success_df(self, test_databases):
        # pass in properly formatted dataframe
        pre_aggregated = [
            ('keycount', 'sex', 'count_star'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="spark"
        )
        if priv:
            pre_aggregated = priv.reader.api.createDataFrame(pre_aggregated[1:], pre_aggregated[0])
            query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'
            res = priv.execute_df(query, pre_aggregated=pre_aggregated)
            assert(str(res['sex'][0]) == '1') # it's sorted
    def test_spark_rdd_success(self, test_databases):
        # pass in properly formatted dataframe
        pre_aggregated = [
            ('keycount', 'sex', 'count_star'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="spark"
        )
        if priv:
            pre_aggregated = priv.reader.api.createDataFrame(pre_aggregated[1:], pre_aggregated[0])
            pre_aggregated = pre_aggregated.rdd
            query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'
            res = priv.execute(query, pre_aggregated=pre_aggregated)
            res = test_databases.to_tuples(res)
            assert(str(res[1][0]) == '1') # it's sorted

class TestPreAggregatedColumnFail:
    # Test input checks for pre_aggregated
    def test_list_col_fail(self, test_databases):
        # pass in wrongly formatted list
        pre_aggregated = [
            ('count_star', 'sex', 'count_age'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]
        query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="pandas"
        )
        if priv:
            try:
                _ = priv.execute(query, pre_aggregated=pre_aggregated)
            except ValueError:
                return
            raise AssertionError("execute should have raised an exception")
    def test_pandas_col_fail(self, test_databases):
        # pass in wrongly formatted dataframe
        pre_aggregated = [
            ('count_star', 'sex', 'count_age'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]
        colnames = pre_aggregated[0]
        pre_aggregated = pd.DataFrame(data=pre_aggregated[1:], index=None)
        pre_aggregated.columns = colnames

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="pandas"
        )
        if priv:
            query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'
            try:
                _ = priv.execute(query, pre_aggregated=pre_aggregated)
            except ValueError:
                return
            raise AssertionError("execute should have raised an exception")
    def test_pandas_col_fail_2(self, test_databases):
        # pass in wrongly formatted dataframe
        pre_aggregated = [
            ('sex', 'count_star'),
            (2, 2000),
            (1, 2000)
        ]
        colnames = pre_aggregated[0]
        pre_aggregated = pd.DataFrame(data=pre_aggregated[1:], index=None)
        pre_aggregated.columns = colnames

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="pandas"
        )
        query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'
        if priv:
            try:
                _ = priv.execute(query, pre_aggregated=pre_aggregated)
            except ValueError:
                return
            raise AssertionError("execute should have raised an exception")
    def test_spark_df_col_fail(self, test_databases):
        # pass in wrongly formatted dataframe
        pre_aggregated = [
            ('keycount', 'age', 'count_star'),
            (1000, 2, 2000),
            (1000, 1, 2000)
        ]

        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="spark"
        )
        if priv:
            pre_aggregated = priv.reader.api.createDataFrame(pre_aggregated[1:], pre_aggregated[0])
            query = 'SELECT sex, COUNT(*) AS n, COUNT(*) AS foo FROM PUMS.PUMS GROUP BY sex ORDER BY sex'
            try:
                _ = priv.execute(query, pre_aggregated=pre_aggregated)
            except ValueError:
                return
            raise AssertionError("execute should have raised an exception")

