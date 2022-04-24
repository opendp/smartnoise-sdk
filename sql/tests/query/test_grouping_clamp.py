from snsql.sql.privacy import Privacy

privacy = Privacy(epsilon=30.0)

class TestGroupingClamp:
    def test_clamp_on(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy)
        assert(len(readers) > 0)
        for reader in readers:
            meta = reader.metadata
            first_key = list(meta.m_tables.keys())[0]
            meta[first_key]["income"].upper = 100
            query = "SELECT AVG(income) AS income FROM PUMS.PUMS"
            res = test_databases.to_tuples(reader.execute(query))
            assert(res[1][0] < 150.0)

    def test_clamp_off(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy)
        assert(len(readers) > 0)
        for reader in readers:
            meta = reader.metadata
            first_key = list(meta.m_tables.keys())[0]
            meta[first_key]["income"].upper = 100
            query = "SELECT income, COUNT(pid) AS n FROM PUMS.PUMS GROUP BY income"
            res = test_databases.to_tuples(reader.execute(query))
            assert(len(res) > 40)
