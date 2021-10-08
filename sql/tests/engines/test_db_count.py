from snsql.sql.privacy import Privacy

privacy = Privacy(alphas=[0.01, 0.05], epsilon=10.0, delta=0.1)

overrides = {'censor_dims': False}

class TestDbCounts:
    def test_db_counts(self, test_databases):
        for dbname in ['PUMS', 'PUMS_pid', 'PUMS_large', 'PUMS_dup']:
            print(dbname)
            readers = test_databases.get_private_readers(privacy=privacy, database=dbname, overrides=overrides)
            for reader in readers:
                print("\t" + reader.engine)
                tablename = dbname
                if dbname == 'PUMS_pid':
                    tablename = 'PUMS'
                query = f'SELECT COUNT(age) AS n FROM PUMS.{tablename}'
                res = reader.execute(query)
                res = test_databases.to_tuples(res)
                print("\t\t" + str(res))
                print(res[0])
                n = res[1][0]
                lower = 950
                upper = 1050
                if tablename == 'PUMS_large':
                    lower = 1223900
                    upper = 1224000
                assert(n > lower and n < upper)
