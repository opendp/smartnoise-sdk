import os
import subprocess
from snsql.sql.odometer import Odometer, OdometerHeterogeneous
from snsql.sql.privacy import Privacy
from snsql.sql.private_reader import PrivateReader
import pandas as pd
import numpy as np
import pytest


from snsql.metadata import Metadata

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))
pums = pd.read_csv(csv_path)
privacy = Privacy(epsilon=1.0)

meta_obj = Metadata.from_(meta_path)

class TestOdometer:
    def test_two_var(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        with pytest.raises(ValueError):
            res = priv.execute("SELECT VAR(age), VAR(income) FROM PUMS.PUMS GROUP BY sex")
            assert(priv.odometer.k == 5)
    def test_count_pid_query(self):
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_path)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(DISTINCT pid) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 1)
    def test_count_query(self):
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_path)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 2)
    def test_count_row_privacy(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(*) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 1)
    def test_count_row_privacy_col(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 2)
    def test_count_row_privacy_col_quantifier(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(DISTINCT age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 2)
    def test_count_row_privacy_no_censor(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = False
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(DISTINCT age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 1)
    def test_variance(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv.execute("SELECT VAR(age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 4)
    def test_std(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv.execute("SELECT STD(age) FROM PUMS.PUMS GROUP BY sex")
        assert(priv.odometer.k == 4)
    def test_avg(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv.execute("SELECT AVG(age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 3)
    def test_sum(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv.execute("SELECT SUM(age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 2)
    def test_odo_hom(self):
        privacy = Privacy(epsilon=0.1, delta = 1/(1000))
        odo = Odometer(privacy)
        for _ in range(300):
            odo.spend()
        eps, delt = odo.spent
        assert(np.isclose(eps, 8.2519))
        assert(np.isclose(delt, 0.2596633))
    def test_odo_het(self):
        privacy = Privacy(epsilon=0.1, delta = 1/(1000))
        odo = OdometerHeterogeneous(privacy)
        for _ in range(300):
            odo.spend()
        eps, delt = odo.spent
        assert(np.isclose(eps, 8.2519))
        assert(np.isclose(delt, 0.2596633))
    def test_odo_het_alternate(self):
        privacy = Privacy(epsilon=0.1, delta = 1/(1000))
        odo = OdometerHeterogeneous(privacy)
        for _ in range(300):
            odo.spend(privacy)
        eps, delt = odo.spent
        assert(np.isclose(eps, 8.2519))
        assert(np.isclose(delt, 0.2596633))

class TestMultiplier:
    def test_count_pid_query(self):
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_path)
        res = priv._get_mechanism_costs("SELECT COUNT(DISTINCT pid) FROM PUMS.PUMS GROUP BY educ")
        res = [r for r in res if r]
        assert(len (res) == 1)
    def test_count_query(self):
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_path)
        res = priv._get_mechanism_costs("SELECT COUNT(age) FROM PUMS.PUMS GROUP BY educ")
        res = [r for r in res if r]
        assert(len(res) == 2)
    def test_count_row_privacy(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv._get_mechanism_costs("SELECT COUNT(*) FROM PUMS.PUMS GROUP BY educ")
        res = [r for r in res if r]
        assert(len(res) == 1)
    def test_count_row_privacy_col(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv._get_mechanism_costs("SELECT COUNT(age) FROM PUMS.PUMS GROUP BY educ")
        res = [r for r in res if r]
        assert(len(res) == 2)
    def test_count_row_privacy_col_quantifier(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv._get_mechanism_costs("SELECT COUNT(DISTINCT age) FROM PUMS.PUMS GROUP BY educ")
        res = [r for r in res if r]
        assert(len(res) == 2)
    def test_count_row_privacy_no_censor(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = False
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv._get_mechanism_costs("SELECT COUNT(DISTINCT age) FROM PUMS.PUMS GROUP BY educ")
        res = [r for r in res if r]
        assert(len(res) == 1)
    def test_variance(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv._get_mechanism_costs("SELECT VAR(age) FROM PUMS.PUMS GROUP BY educ")
        res = [r for r in res if r]
        assert(len(res) == 4)
    def test_std(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv._get_mechanism_costs("SELECT STD(age) FROM PUMS.PUMS GROUP BY sex")
        res = [r for r in res if r]
        assert(len(res) == 4)
    def test_avg(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv._get_mechanism_costs("SELECT AVG(age) FROM PUMS.PUMS GROUP BY educ")
        res = [r for r in res if r]
        assert(len(res) == 3)
    def test_sum(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv._get_mechanism_costs("SELECT SUM(age) FROM PUMS.PUMS GROUP BY educ")
        res = [r for r in res if r]
        assert(len(res) == 2)
    def test_two_var(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        query = "SELECT VAR(age), VAR(income) FROM PUMS.PUMS GROUP BY sex"
        with pytest.raises(ValueError):
            res = priv._get_mechanism_costs(query)
            res = [r for r in res if r]
            assert(len(res) == 5)
            eps, _ = priv.get_privacy_cost(query)
            assert(eps <= 5.0)


@pytest.mark.parametrize(
    "epsilon, delta, k, expected",
    [
        # Each alternative in Theorem 3.4 is the minimum in one of these cases.
        (1.0, 0.01, 1, (1.0, 0.01495)),
        (0.1, 0.1, 10, (0.7135807026501802, 0.668755481905)),
        (0.1, 0.001, 300, (8.251929371461469, 0.2596633213599786)),
    ],
    ids=["basic", "log-e", "log-inverse-slack"],
)
def test_homogeneous_composition_reference_values(epsilon, delta, k, expected):
    # Regression for GH-643, reported by marcpare. Reference values use
    # Kairouz et al., Theorem 3.4 (https://arxiv.org/abs/1311.0776).
    # They were evaluated separately with 60-digit decimal arithmetic.
    odo = Odometer(Privacy(epsilon=epsilon, delta=delta))
    odo.spend(k)

    assert odo.spent == pytest.approx(expected, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize(
    "epsilon, delta, k",
    [(0.01, 1e-6, 100), (0.1, 0.001, 300), (0.1, 0.1, 10), (1.0, 0.1, 10)],
)
def test_homogeneous_matches_identical_heterogeneous_spends(epsilon, delta, k):
    privacy = Privacy(epsilon=epsilon, delta=delta)
    homogeneous = Odometer(privacy)
    heterogeneous = OdometerHeterogeneous(privacy)
    homogeneous.spend(k)
    for _ in range(k):
        heterogeneous.spend()

    assert homogeneous.spent == pytest.approx(heterogeneous.spent)


def test_homogeneous_delta_does_not_decrease_with_more_spends():
    odo = Odometer(Privacy(epsilon=0.1, delta=0.1))
    deltas = [odo.spent[1]]
    for _ in range(30):
        odo.spend()
        deltas.append(odo.spent[1])

    assert deltas == sorted(deltas)
    assert all(0 <= delta <= 1 for delta in deltas)


def test_homogeneous_reset_and_batch_spends():
    odo = Odometer(Privacy(epsilon=0.1, delta=0.1))
    assert odo.spent == (0.0, 0.0)
    odo.spend(10)
    spent = odo.spent
    odo.reset()
    assert odo.spent == (0.0, 0.0)
    assert odo.k == 0
    for _ in range(10):
        odo.spend()
    assert odo.spent == spent
