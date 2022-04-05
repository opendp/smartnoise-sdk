from snsql.sql.privacy import Privacy

privacy = Privacy(epsilon=30.0)

class TestThreePartObjectName:
    def test_clamp_on(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS', privacy=privacy)
        assert(len(readers) > 0)
        for reader in readers:
            if reader.engine == 'postgres':
                query = "SELECT COUNT(*) FROM PUMS.PUMS.PUMS"
                reader.execute(query)
