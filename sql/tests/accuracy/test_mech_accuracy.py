from snsql.sql._mechanisms import *

n_trials = 500

"""
Sample each mechanism 500 times for a combination of epsilon, delta,
alpha, and sensitivity.  The sampled values must fall within +/- 3%
of the reported accuracy, to account for slack due to small sample size.
"""
class TestMechAccuracy:
    def test_lap(self):
        for epsilon in [0.1, 2.0]:
            for delta in [0.0, 0.01]:
                for alpha in [0.01, 0.05]:
                    for sensitivity in [1.0, 33.0]:
                        mech = Laplace(epsilon, delta=delta, sensitivity=sensitivity)
                        acc = mech.accuracy(alpha)
                        print(f"{epsilon}, {delta}, {alpha}, sens={sensitivity} +/-{acc}")
                        zeros = [0.0] * n_trials
                        vals = mech.release(zeros)
                        inside = [v >= -acc and v <= acc for v in vals]
                        percent_in = sum(inside) / n_trials
                        percentile = 1.0 - alpha
                        assert(percent_in >= percentile - 0.03 and percent_in <= percentile + 0.03)
    def test_gauss(self):
        for epsilon in [0.1, 2.0]:
            for delta in [10E-6, 0.01]:
                for alpha in [0.01, 0.05]:
                    for sensitivity in [1.0, 33.0]:
                        mech = Gaussian(epsilon, delta=delta, sensitivity=sensitivity)
                        acc = mech.accuracy(alpha)
                        print(f"{epsilon}, {delta}, {alpha}, sens={sensitivity} +/-{acc}")
                        zeros = [0.0] * n_trials
                        vals = mech.release(zeros)
                        inside = [v >= -acc and v <= acc for v in vals]
                        percent_in = sum(inside) / n_trials
                        percentile = 1.0 - alpha
                        assert(percent_in >= percentile - 0.03 and percent_in <= percentile + 0.03)
    def test_geom(self):
        for epsilon in [0.1, 1.0]:
            for delta in [0.0]:
                for alpha in [0.01, 0.05]:
                    for sensitivity in [1, 33]:
                        mech = Geometric(epsilon, delta=delta, sensitivity=sensitivity)
                        acc = mech.accuracy(alpha)
                        print(f"{epsilon}, {delta}, {alpha}, sens={sensitivity} +/-{acc}")
                        zeros = [0.0] * n_trials
                        vals = mech.release(zeros)
                        inside = [v >= -acc and v <= acc for v in vals]
                        percent_in = sum(inside) / n_trials
                        percentile = 1.0 - alpha
                        assert(percent_in >= percentile - 0.03 and percent_in <= percentile + 0.035)
                        