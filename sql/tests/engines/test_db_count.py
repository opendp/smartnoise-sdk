import pytest
import sys
from snsql import metadata

from snsql.sql.privacy import Privacy

privacy = Privacy(alphas=[0.01, 0.05], epsilon=30.0, delta=0.1)

overrides = {'censor_dims': False}

class TestDbCounts:
    @pytest.mark.skipif(sys.version_info < (3, 8), reason="Skip because older PRNG")
    def test_db_counts(self, test_databases):
        # Actual is 1000
        for dbname in ['PUMS', 'PUMS_pid', 'PUMS_large', 'PUMS_dup', 'PUMS_null' ]:
            readers = test_databases.get_private_readers(privacy=privacy, database=dbname, overrides=overrides)
            for reader in readers:
                tablename = 'PUMS' if dbname != 'PUMS_large' else 'PUMS_large'
                query = f'SELECT COUNT(age) AS n FROM PUMS.{tablename}'
                res = reader.execute(query)
                res = test_databases.to_tuples(res)
                n = res[1][0]
                lower = 980
                upper = 1020
                if dbname == 'PUMS_null':
                    # Actual is ~926
                    # Reservoir sampling increases variance
                    lower = 890
                    upper = 950
                if dbname == 'PUMS_large':
                    lower = 1223900
                    upper = 1224000
                print(f"Table {dbname}.PUMS.{tablename} has {n} COUNT(age) rows in {reader.engine}")
                assert(n > lower and n < upper)
    def test_db_counts_star(self, test_databases):
        # Actual is 1000
        for dbname in ['PUMS', 'PUMS_pid', 'PUMS_large', 'PUMS_dup', 'PUMS_null']:
            readers = test_databases.get_private_readers(privacy=privacy, database=dbname, overrides=overrides)
            for reader in readers:
                tablename = 'PUMS' if dbname != 'PUMS_large' else 'PUMS_large'
                query = f'SELECT COUNT(*) AS n FROM PUMS.{tablename}'
                res = reader.execute(query)
                res = test_databases.to_tuples(res)
                n = res[1][0]
                lower = 980
                upper = 1020
                if dbname == 'PUMS_null':
                    # actual is ~978
                    lower = 940
                    upper = 995
                if dbname == 'PUMS_large':
                    lower = 1223900
                    upper = 1224000
                print(f"Table {dbname}.PUMS.{tablename} has {n} COUNT(*) rows in {reader.engine}")
                assert(n > lower and n < upper)
    def test_db_counts_no_max_ids(self, test_databases):
        # Actual is ~1690 or ~1950 depending on PRNG
        for dbname in ['PUMS_dup', 'PUMS_null']:
            overrides = {'max_ids': 9, 'censor_dims': False}
            readers = test_databases.get_private_readers(privacy=privacy, database=dbname, overrides=overrides)
            for reader in readers:
                tablename = 'PUMS' if dbname != 'PUMS_large' else 'PUMS_large'
                query = f'SELECT COUNT(*) AS n FROM PUMS.{tablename}'
                res = reader.execute(query)
                res = test_databases.to_tuples(res)
                n = res[1][0]
                lower = 1650
                upper = 2200
                print(f"Table {dbname}.PUMS.{tablename} has {n} COUNT(*) rows in {reader.engine} with no max_ids")
                assert(n > lower and n < upper)
    @pytest.mark.skipif(sys.version_info < (3, 8), reason="Skip because older PRNG")
    def test_db_counts_distinct_pid(self, test_databases):
        for dbname in ['PUMS_pid', 'PUMS_dup', 'PUMS_null']:
            overrides = {'max_ids': 9, 'censor_dims': False}
            readers = test_databases.get_private_readers(privacy=privacy, database=dbname, overrides=overrides)
            for reader in readers:
                tablename = 'PUMS'
                query = f'SELECT COUNT(DISTINCT pid) AS n FROM PUMS.{tablename}'
                res = reader.execute(query)
                res = test_databases.to_tuples(res)
                n = res[1][0]
                # Actual is 1000
                lower = 985
                upper = 1010
                if dbname == 'PUMS_null':
                    # this is more variable with max_ids
                    # Actual is ~977
                    lower = 945
                    upper = 990
                print(f"Table {dbname}.PUMS.{tablename} has {n} COUNT(DISTINCT pid) rows in {reader.engine}")
                assert(n > lower and n < upper)
    @pytest.mark.skipif(sys.version_info < (3, 8), reason="Skip because older PRNG")
    def test_count_null_impute(self, test_databases):
        # Replace missing values for age, so count should equal count(*)
        # Actual is ~1000
        for dbname in ['PUMS_null']:
            overrides = {'max_ids': 1, 'censor_dims': False}
            readers = test_databases.get_private_readers(privacy=privacy, database=dbname, overrides=overrides)
            for reader in readers:
                metadata = reader.metadata
                first_key = list(metadata.m_tables.keys())[0]
                metadata[first_key]['age'].missing_value = 30
                tablename = 'PUMS'
                query = f'SELECT COUNT(age) AS n FROM PUMS.{tablename}'
                res = reader.execute(query)
                res = test_databases.to_tuples(res)
                n = res[1][0]
                lower = 950
                upper = 1050
                print(f"Table {dbname}.PUMS.{tablename} has {n} COUNT(*) rows in {reader.engine}")
                assert(n > lower and n < upper)