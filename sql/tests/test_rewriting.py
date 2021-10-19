import os
import subprocess
from snsql.metadata import Metadata
from snsql.sql.private_rewriter import Rewriter
from snsql.sql.parse import QueryParser


queries = [
        "SELECT COUNT(*) AS c FROM PUMS.PUMS",
        "SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c",
        "SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC",
        "SELECT COUNT(*) * 5 AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC",
        "SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 90 AND educ = '8'",
        "SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 80 GROUP BY educ",
        "SELECT COUNT(*) as c FROM PUMS.PUMS WHERE age > 100",
        "SELECT SUM(age) as c FROM PUMS.PUMS WHERE age > 100",
        "SELECT SUM(age) as age_total FROM PUMS.PUMS",
        "SELECT POWER(SUM(age), 2) as age_total FROM PUMS.PUMS",
        "SELECT (SUM(age)) as age_total FROM PUMS.PUMS",
        "SELECT ((SUM(age))) as age_total FROM PUMS.PUMS",
        "SELECT COUNT(*) AS c FROM PUMS.PUMS GROUP BY married",
        "SELECT AVG(age) as age_total FROM PUMS.PUMS",
        "SELECT VARIANCE(age) as age_total FROM PUMS.PUMS",
        "SELECT STDDEV(age) as age_total FROM PUMS.PUMS",
        "SELECT COUNT(*) AS count_all FROM PUMS.PUMS",
        "SELECT AVG(age) FROM PUMS.PUMS",
        "SELECT COUNT(*) FROM PUMS.PUMS",
        "SELECT age, SUM(income) AS sum_income FROM PUMS.PUMS GROUP BY age HAVING age > 5 AND COUNT(*) > 7 OR NOT ( age < 5)",
        "SELECT age, sex, SUM(income) AS sum_income FROM PUMS.PUMS GROUP BY age, sex ORDER BY age, sum_income, sex DESC LIMIT 10",
        "SELECT COUNT(*) AS count_all, SUM(income) AS sum_income, AVG(income) AS avg_income, VARIANCE(income) AS var_income, STDDEV(income) AS stddev_income FROM PUMS.PUMS",
        "SELECT (VAR(age)) AS my_var, married FROM PUMS.PUMS GROUP BY married",
        "SELECT age, sex, SUM(income) AS my_sum, SUM(age) AS sum_age FROM PUMS.PUMS GROUP BY age, sex HAVING NOT ( sex = True ) ORDER BY my_sum, sex DESC LIMIT 10",
        "SELECT married, SUM(income) AS my_sum, SUM(age) AS sum_age FROM PUMS.PUMS GROUP BY married HAVING married",
        "SELECT SUM(age) FROM PUMS.PUMS",
        "SELECT COUNT(*), VAR(age) AS var_age FROM PUMS.PUMS",
        "SELECT married, educ, income, AVG(age) FROM PUMS.PUMS GROUP BY married, educ, income",
        "SELECT TOP 5 educ, AVG(income) FROM PUMS.PUMS GROUP BY educ",
        "SELECT COUNT(married), AVG(age) FROM PUMS.PUMS",
        "SELECT 3 AS three, married, educ, AVG(income) FROM PUMS.PUMS GROUP BY educ, married",
        "SELECT educ FROM PUMS.PUMS GROUP BY educ",
        "SELECT SUM(age) AS age FROM PUMS.PUMS",
        "SELECT (SUM(age)) AS age FROM PUMS.PUMS",
        "SELECT SUM(age) * 1 AS age FROM PUMS.PUMS",
        "SELECT SUM(age), COUNT(*) FROM PUMS.PUMS",
        "SELECT PI(), RAND(), POWER(AVG(age), 2) FROM PUMS.PUMS",
        "SELECT SUM(income) AS income FROM PUMS.PUMS WHERE sex = True",
        "SELECT AVG(income) AS income FROM PUMS.PUMS GROUP BY married",
        "SELECT educ, SUM(income) AS total_income, AVG(income) AS avg_income, AVG(age) FROM PUMS.PUMS WHERE married = TRUE GROUP BY educ",
        "SELECT educ, SUM(income) AS income FROM PUMS.PUMS WHERE educ IN (2, 3) GROUP BY educ ORDER BY educ DESC",
        "SELECT COUNT(DISTINCT pid) AS n, age, income FROM PUMS.PUMS WHERE age > 55 GROUP BY age, income",
        "SELECT COUNT(DISTINCT pid) AS pid FROM PUMS.PUMS GROUP BY married",
        "SELECT COUNT(DISTINCT pid) AS pid FROM PUMS.PUMS",
    ]

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
metadata = Metadata.from_file(meta_path)

metadata['PUMS.PUMS']['']

def test_rewriting():
    for query in queries:
        try:
            query = QueryParser(metadata).query(str(query))
            dp_query = Rewriter(metadata).query(query)
        except:
            raise ValueError(f"Query parse and rewrite failed: {query}")
        parsed_dp_query = QueryParser(metadata).query(str(dp_query))
        assert dp_query == parsed_dp_query
