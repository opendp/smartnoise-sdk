import pytest
from snsql import Privacy

privacy = Privacy(epsilon=10.0)

class TestSortExpressions:
    def test_skip_masked_out_col(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            q = "SELECT sex AS gender, educ FROM pums GROUP BY sex, educ ORDER by sex DESC"
            res = test_databases.to_tuples(reader.execute(q))
            gender = [str(c[0]).strip() for c in res[1:]]
            assert(all([g == '1' for g in gender[:5]]))
            assert(all([g == '0' for g in gender[-5:]]))
    def test_on_masked_out_col(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # output column name masks input column name, and sort uses masked name
            q = "SELECT sex AS gender, educ AS sex FROM pums GROUP BY sex, educ ORDER by gender DESC"
            res = test_databases.to_tuples(reader.execute(q))
            gender = [str(c[0]).strip() for c in res[1:]]
            assert(all([g == '1' for g in gender[:5]]))
            assert(all([g == '0' for g in gender[-5:]]))
    def test_nonstandard_masked_out_col(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # using an expression on the output column name works here, though it's not valid SQL-92
            q = "SELECT sex AS gender, educ AS sex FROM pums GROUP BY sex, educ ORDER by (CAST(gender AS INTEGER) + 5) DESC"
            res = test_databases.to_tuples(reader.execute(q))
            gender = [str(c[0]).strip() for c in res[1:]]
            assert(all([g == '1' for g in gender[:5]]))
            assert(all([g == '0' for g in gender[-5:]]))
    def test_swapped_masked_out_col(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # output column name masks input column name, and sort uses masked name
            # note that 'sex' in this case sorts by 'educ'
            q = "SELECT sex AS gender, educ AS sex FROM pums GROUP BY sex, educ ORDER by sex DESC"
            res = test_databases.to_tuples(reader.execute(q))
            educ = [str(c[1]).strip() for c in res[1:]]
            educ_reversed = reversed(sorted(educ))
            assert(all([e1 == e2 for e1, e2 in zip(educ, educ_reversed)]))
    def test_missing_column(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            q = "SELECT educ FROM pums GROUP BY educ ORDER by sex DESC"
            with pytest.raises(Exception):
                test_databases.to_tuples(reader.execute(q))
    def test_expr_on_masked_out_col(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # output column name masks input column name, and sort uses masked name
            q = "SELECT sex AS gender, educ AS sex FROM pums GROUP BY sex, educ ORDER by (CAST(gender AS INTEGER) * -1) DESC"
            res = test_databases.to_tuples(reader.execute(q))
            gender = [str(c[0]).strip() for c in res[1:]]
            assert(all([g == '0' for g in gender[:5]]))
            assert(all([g == '1' for g in gender[-5:]]))
    def test_mult_expr_on_masked_out_col(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            # output column name masks input column name, and sort uses masked name
            q = "SELECT sex AS gender, educ AS sex FROM pums GROUP BY sex, educ ORDER by (CAST(gender AS INTEGER) * -1) DESC, educ ASC"
            res = test_databases.to_tuples(reader.execute(q))
            gender = [str(c[0]).strip() for c in res[1:]]
            educ = [str(c[1]).strip() for c in res[1:]]
            assert(all([g == '0' for g in gender[:5]]))
            assert(all([g == '1' for g in gender[-5:]]))
            educ = educ[:5]
            educ_sorted = sorted(educ)
            assert(all([e1 == e2 for e1, e2 in zip(educ, educ_sorted)]))
    def test_cast_numeric_sort(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            q = "SELECT educ FROM pums GROUP BY educ ORDER by CAST(educ AS INTEGER) DESC"
            res = test_databases.to_tuples(reader.execute(q))
            educ = [int(str(c[0]).strip()) for c in res[1:]]
            educ_reversed = reversed(sorted(educ))
            assert(all([e1 == e2 for e1, e2 in zip(educ, educ_reversed)]))
    def test_cast_string_sort(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            q = "SELECT educ FROM pums GROUP BY educ ORDER by CAST(educ AS VARCHAR) DESC"
            res = test_databases.to_tuples(reader.execute(q))
            educ = [str(c[0]).strip() for c in res[1:]]
            educ_reversed = reversed(sorted(educ))
            assert(all([e1 == e2 for e1, e2 in zip(educ, educ_reversed)]))
    def test_missing_expr(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            q = "SELECT educ, COUNT(DISTINCT pid) AS n FROM pums GROUP BY educ ORDER by COUNT(pid) DESC"
            with pytest.raises(Exception):
                test_databases.to_tuples(reader.execute(q))
