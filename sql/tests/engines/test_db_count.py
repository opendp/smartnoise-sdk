from snsql.sql.privacy import Privacy

privacy = Privacy(alphas=[0.01, 0.05], epsilon=10.0, delta=0.1)

overrides = {'censor_dims': False}

class TestDbCounts:
    def test_db_counts(self, test_databases):
        for dbname in ['PUMS', 'PUMS_pid', 'PUMS_large', 'PUMS_dup', 'PUMS_null' ]:
            readers = test_databases.get_private_readers(privacy=privacy, database=dbname, overrides=overrides)
            for reader in readers:
                tablename = 'PUMS' if dbname != 'PUMS_large' else 'PUMS_large'
                query = f'SELECT COUNT(age) AS n FROM PUMS.{tablename}'
                res = reader.execute(query)
                res = test_databases.to_tuples(res)
                n = res[1][0]
                lower = 950
                upper = 1050
                if dbname == 'PUMS_large':
                    lower = 1223900
                    upper = 1224000
                elif dbname == 'PUMS_null':
                    lower = 920
                    upper = 980
                print(f"Table {dbname}.PUMS.{tablename} has {n} COUNT(age) rows in {reader.engine}")
                assert(n > lower and n < upper)
    def test_db_counts_star(self, test_databases):
        for dbname in ['PUMS', 'PUMS_pid', 'PUMS_large', 'PUMS_dup', 'PUMS_null']:
            readers = test_databases.get_private_readers(privacy=privacy, database=dbname, overrides=overrides)
            for reader in readers:
                tablename = 'PUMS' if dbname != 'PUMS_large' else 'PUMS_large'
                query = f'SELECT COUNT(*) AS n FROM PUMS.{tablename}'
                res = reader.execute(query)
                res = test_databases.to_tuples(res)
                n = res[1][0]
                lower = 950
                upper = 1050
                if tablename == 'PUMS_large':
                    lower = 1223900
                    upper = 1224000
                print(f"Table {dbname}.PUMS.{tablename} has {n} COUNT(*) rows in {reader.engine}")
                assert(n > lower and n < upper)
    def test_db_counts_no_max_ids(self, test_databases):
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
                upper = 1800
                print(f"Table {dbname}.PUMS.{tablename} has {n} COUNT(*) rows in {reader.engine} with no max_ids")
                assert(n > lower and n < upper)

