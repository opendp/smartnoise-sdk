import pytest
from snsql import Privacy

privacy = Privacy(epsilon=10.0)

class TestHavingExpressions:
    def test_flter_star(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # use expression with * and no column name
            q = "SELECT educ, COUNT(*) FROM pums GROUP BY educ HAVING COUNT(*) > 50"
            res = test_databases.to_tuples(reader.execute(q))
            assert(len(res) > 5)
            assert(len(res) < 11)
    def test_filter_distinct(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # use expression with distinct
            q = "SELECT educ, COUNT(DISTINCT pid) AS n FROM pums GROUP BY educ HAVING COUNT(DISTINCT pid) > 50"
            res = test_databases.to_tuples(reader.execute(q))
            assert(len(res) > 5)
            assert(len(res) < 11)
    def test_filter_alias(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # use expression with distinct and alias.  Not SQL-92 but works
            q = "SELECT educ, COUNT(DISTINCT pid) AS n FROM pums GROUP BY educ HAVING n > 50"
            res = test_databases.to_tuples(reader.execute(q))
            assert(len(res) > 5)
            assert(len(res) < 11)
    def test_mismatch_expr(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # count expressions don't match: error
            q = "SELECT educ, COUNT(DISTINCT pid) AS n FROM pums GROUP BY educ HAVING COUNT(*) > 50"
            with pytest.raises(Exception):
                res = test_databases.to_tuples(reader.execute(q))
    def test_filter_complex_expr(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # use expression with * and combine columns on having
            q = "SELECT educ, COUNT(*) FROM pums GROUP BY educ HAVING (COUNT(*) / CAST(educ AS INTEGER)) > 5"
            res = test_databases.to_tuples(reader.execute(q))
            assert(len(res) > 7)
            assert(len(res) < 12)
    def test_swapa_1(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # swap columns and make sure original takes precedence
            # length should be 0
            q = "SELECT sex as educ, educ as sex, COUNT(*) AS n FROM pums GROUP BY sex, educ HAVING sex > 2"
            res = test_databases.to_tuples(reader.execute(q))
            assert(len(res) < 2)
    def test_swapa_2(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # swap columns and make sure original takes precedence
            # length should be 0
            q = "SELECT educ as sex, sex as educ, COUNT(*) AS n FROM pums GROUP BY sex, educ HAVING sex > 2"
            res = test_databases.to_tuples(reader.execute(q))
            assert(len(res) < 2)
    def test_swapb_1(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # swap columns and make sure original takes precedence
            # length should be > 10
            q = "SELECT sex as educ, educ as sex, COUNT(*) AS n FROM pums GROUP BY sex, educ HAVING educ > 2"
            res = test_databases.to_tuples(reader.execute(q))
            assert(len(res) > 10)
    def test_swapb_2(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # swap columns and make sure original takes precedence
            # length should be > 10
            q = "SELECT educ as sex, sex as educ, COUNT(*) AS n FROM pums GROUP BY sex, educ HAVING educ > 2"
            res = test_databases.to_tuples(reader.execute(q))
            assert(len(res) > 10)